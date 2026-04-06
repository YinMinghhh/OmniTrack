# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import os.path as osp
import pickle
import shutil
import tempfile
import time

import mmcv
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

from mmdet.core import encode_mask_results


import mmcv
import numpy as np
import pycocotools.mask as mask_util


def custom_encode_mask_results(mask_results):
    """Encode bitmap mask to RLE code. Semantic Masks only
    Args:
        mask_results (list | tuple[list]): bitmap mask results.
            In mask scoring rcnn, mask_results is a tuple of (segm_results,
            segm_cls_score).
    Returns:
        list | tuple: RLE encoded mask.
    """
    cls_segms = mask_results
    num_classes = len(cls_segms)
    encoded_mask_results = []
    for i in range(len(cls_segms)):
        encoded_mask_results.append(
            mask_util.encode(
                np.array(
                    cls_segms[i][:, :, np.newaxis], order="F", dtype="uint8"
                )
            )[0]
        )  # encoded with RLE
    return [encoded_mask_results]


def _extract_sample_keys(data, batch_size):
    img_metas = data.get("img_metas")
    if img_metas is None or not hasattr(img_metas, "data"):
        return None
    meta_list = img_metas.data[0]
    if not isinstance(meta_list, list):
        return None

    sample_keys = []
    for meta in meta_list:
        if not isinstance(meta, dict):
            return None
        sample_idx = meta.get("sample_idx")
        if sample_idx is None:
            return None
        sample_keys.append(sample_idx)

    if len(sample_keys) != batch_size:
        return None
    return sample_keys


def _build_rank_sample_plan(data_loader, world_size):
    sampler = getattr(data_loader, "sampler", None)
    if sampler is None:
        return None
    if not hasattr(sampler, "dataset") or not hasattr(sampler, "num_replicas"):
        return None
    if sampler.num_replicas != world_size:
        return None

    sampler_cls = type(sampler)
    sampler_kwargs = dict(
        dataset=sampler.dataset,
        num_replicas=sampler.num_replicas,
        shuffle=getattr(sampler, "shuffle", False),
        seed=getattr(sampler, "seed", 0),
    )
    try:
        return [
            len(list(iter(sampler_cls(rank=rank, **sampler_kwargs))))
            for rank in range(world_size)
        ]
    except Exception:
        return None


def _planned_global_batch(rank_sample_plan, iteration, batch_size):
    processed = 0
    start = iteration * batch_size
    for num_samples in rank_sample_plan:
        if start >= num_samples:
            continue
        processed += min(batch_size, num_samples - start)
    return processed


def _merge_ordered_results(part_list, size, dataset):
    if dataset is None or not hasattr(dataset, "data_infos"):
        return None

    key_to_idx = {}
    for idx, info in enumerate(dataset.data_infos[:size]):
        sample_key = info.get("token")
        if sample_key is None or sample_key in key_to_idx:
            return None
        key_to_idx[sample_key] = idx

    ordered_results = [None] * size
    for part in part_list:
        sample_keys = part.get("sample_keys")
        part_results = part.get("results")
        if sample_keys is None or part_results is None:
            return None
        if len(sample_keys) != len(part_results):
            return None
        for sample_key, result in zip(sample_keys, part_results):
            sample_idx = key_to_idx.get(sample_key)
            if sample_idx is None or ordered_results[sample_idx] is not None:
                return None
            ordered_results[sample_idx] = result

    if any(result is None for result in ordered_results):
        return None
    return ordered_results


def custom_multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.
    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.
    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
    Returns:
        list: The prediction results.
    """
    model.eval()
    bbox_results = []
    mask_results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
        rank_sample_plan = _build_rank_sample_plan(data_loader, world_size)
    else:
        rank_sample_plan = None
    default_batch_size = getattr(data_loader, "batch_size", 1)
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    have_mask = False
    sample_keys_part = []
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            # encode mask results
            if isinstance(result, dict):
                if "bbox_results" in result.keys():
                    bbox_result = result["bbox_results"]
                    batch_size = len(result["bbox_results"])
                    bbox_results.extend(bbox_result)
                if (
                    "mask_results" in result.keys()
                    and result["mask_results"] is not None
                ):
                    mask_result = custom_encode_mask_results(
                        result["mask_results"]
                    )
                    mask_results.extend(mask_result)
                    have_mask = True
            else:
                batch_size = len(result)
                bbox_results.extend(result)

        sample_keys = _extract_sample_keys(data, batch_size)
        if sample_keys is None:
            sample_keys_part.extend([None] * batch_size)
        else:
            sample_keys_part.extend(sample_keys)

        if rank == 0:
            if rank_sample_plan is not None:
                planned_batch = _planned_global_batch(
                    rank_sample_plan, i, default_batch_size
                )
            else:
                planned_batch = batch_size
            planned_batch = min(
                planned_batch, len(dataset) - prog_bar.completed
            )
            for _ in range(planned_batch):
                prog_bar.update()

    # collect results from all ranks
    keyed_collect = (
        not have_mask
        and len(sample_keys_part) == len(bbox_results)
        and all(sample_key is not None for sample_key in sample_keys_part)
    )
    if gpu_collect:
        bbox_results = collect_results_gpu(
            bbox_results,
            len(dataset),
            sample_keys_part=sample_keys_part if keyed_collect else None,
            dataset=dataset if keyed_collect else None,
        )
        if have_mask:
            mask_results = collect_results_gpu(mask_results, len(dataset))
        else:
            mask_results = None
    else:
        bbox_results = collect_results_cpu(
            bbox_results,
            len(dataset),
            tmpdir,
            sample_keys_part=sample_keys_part if keyed_collect else None,
            dataset=dataset if keyed_collect else None,
        )
        tmpdir = tmpdir + "_mask" if tmpdir is not None else None
        if have_mask:
            mask_results = collect_results_cpu(
                mask_results, len(dataset), tmpdir
            )
        else:
            mask_results = None

    if mask_results is None:
        return bbox_results
    return {"bbox_results": bbox_results, "mask_results": mask_results}


def collect_results_cpu(
    result_part, size, tmpdir=None, sample_keys_part=None, dataset=None
):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full(
            (MAX_LEN,), 32, dtype=torch.uint8, device="cuda"
        )
        if rank == 0:
            mmcv.mkdir_or_exist(".dist_test")
            tmpdir = tempfile.mkdtemp(dir=".dist_test")
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device="cuda"
            )
            dir_tensor[: len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    payload = dict(results=result_part, sample_keys=sample_keys_part)
    mmcv.dump(payload, osp.join(tmpdir, f"part_{rank}.pkl"))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f"part_{i}.pkl")
            part_list.append(mmcv.load(part_file))
        ordered_results = _merge_ordered_results(part_list, size, dataset)
        if ordered_results is None:
            ordered_results = []
            for part in part_list:
                ordered_results.extend(list(part["results"]))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(
    result_part, size, sample_keys_part=None, dataset=None
):
    rank, world_size = get_dist_info()
    payload = dict(results=result_part, sample_keys=sample_keys_part)
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(payload)), dtype=torch.uint8, device="cuda"
    )
    shape_tensor = torch.tensor(part_tensor.shape, device="cuda")
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device="cuda")
    part_send[: shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    dist.all_gather(part_recv_list, part_send)

    if rank != 0:
        return None

    part_list = []
    for recv, shape in zip(part_recv_list, shape_list):
        part_result = pickle.loads(recv[: shape[0]].cpu().numpy().tobytes())
        if part_result["results"]:
            part_list.append(part_result)

    ordered_results = _merge_ordered_results(part_list, size, dataset)
    if ordered_results is None:
        ordered_results = []
        for part in part_list:
            ordered_results.extend(list(part["results"]))
    return ordered_results[:size]
