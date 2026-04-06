import math

import torch
from torch.utils.data import DistributedSampler as _DistributedSampler
from .sampler import SAMPLER

import pdb
import sys


class ForkedPdb(pdb.Pdb):
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open("/dev/stdin")
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


def set_trace():
    ForkedPdb().set_trace(sys._getframe().f_back)


@SAMPLER.register_module()
class DistributedSampler(_DistributedSampler):
    def __init__(
        self, dataset=None, num_replicas=None, rank=None, shuffle=True, seed=0
    ):
        super().__init__(
            dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle
        )
        # for the compatibility from PyTorch 1.3+
        self.seed = seed if seed is not None else 0

    def __iter__(self):
        # deterministically shuffle based on epoch
        assert not self.shuffle
        if "data_infos" in dir(self.dataset):
            timestamps = [
                x["timestamp"] / 1e6 for x in self.dataset.data_infos
            ]
            vehicle_idx = [
                x["lidar_path"].split("/")[-1][:4]
                if "lidar_path" in x
                else None
                for x in self.dataset.data_infos
            ]
        else:
            timestamps = [
                x["timestamp"] / 1e6
                for x in self.dataset.datasets[0].data_infos
            ] * len(self.dataset.datasets)
            vehicle_idx = [
                x["lidar_path"].split("/")[-1][:4]
                if "lidar_path" in x
                else None
                for x in self.dataset.datasets[0].data_infos
            ] * len(self.dataset.datasets)

        sequence_splits = []
        for i in range(len(timestamps)):
            if i == 0 or (
                abs(timestamps[i] - timestamps[i - 1]) > 4
                or vehicle_idx[i] != vehicle_idx[i - 1]
            ):
                sequence_splits.append([i])
            else:
                sequence_splits[-1].append(i)

        target_lengths = [
            len(self.dataset) // self.num_replicas for _ in range(self.num_replicas)
        ]
        for i in range(len(self.dataset) % self.num_replicas):
            target_lengths[i] += 1

        indices_per_rank = [[] for _ in range(self.num_replicas)]
        rank = 0
        current_length = 0

        for split in sequence_splits:
            split_length = len(split)
            if rank < self.num_replicas - 1 and current_length > 0:
                current_target = target_lengths[rank]
                keep_cost = abs((current_length + split_length) - current_target)
                move_cost = abs(current_length - current_target)
                if keep_cost > move_cost:
                    rank += 1
                    current_length = 0

            indices_per_rank[rank].extend(split)
            current_length += split_length

        indices = indices_per_rank[self.rank]

        self.num_samples = len(indices)
        return iter(indices)
