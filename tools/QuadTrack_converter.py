import os
import numpy as np
import pickle as pkl
from tqdm import tqdm


class QuadTrackConverter(object):
    '''
    Convert QuadTrack (MOT format) to OmniTrack pkl file.
    '''

    def __init__(self, data_root: str, save_root: str):
        self.data_root = data_root  # .../OmniTrack/data
        self.quad_root = os.path.join(data_root, 'QuadTrack')
        self.save_root = save_root

        self.train_pkl = []
        self.val_pkl = []
        self.test_pkl = []

        # 定义验证集序列 (这里简单选取最后两个序列作为验证集，你可以根据需要修改)
        # 获取所有训练序列并排序
        train_seq_dir = os.path.join(self.quad_root, 'train')
        if os.path.exists(train_seq_dir):
            all_seqs = sorted(os.listdir(train_seq_dir))
            # 简单的 90/10 切分，或者至少留2个做验证
            val_count = max(2, int(len(all_seqs) * 0.1))
            self.val_seqs = all_seqs[-val_count:]
            self.train_seqs = all_seqs[:-val_count]
        else:
            self.val_seqs = []
            self.train_seqs = []

        print(f"Train Seqs: {len(self.train_seqs)}, Val Seqs: {len(self.val_seqs)}")

    def generate_pkl(self):
        # Handle Train/Val
        print("Processing Train/Val sets...")
        train_root = os.path.join(self.quad_root, 'train')
        if os.path.exists(train_root):
            for seq in tqdm(self.train_seqs, desc="Train"):
                self.handle_seq(seq, 'train', train_root)
            for seq in tqdm(self.val_seqs, desc="Val"):
                self.handle_seq(seq, 'val', train_root)

        # Handle Test
        print("Processing Test set...")
        test_root = os.path.join(self.quad_root, 'test')
        if os.path.exists(test_root):
            test_seqs = sorted(os.listdir(test_root))
            for seq in tqdm(test_seqs, desc="Test"):
                self.handle_seq(seq, 'test', test_root)

        self.save_pkl()

    def handle_seq(self, seq_name, split, split_root):
        seq_path = os.path.join(split_root, seq_name)
        img_dir = os.path.join(seq_path, 'img1')
        gt_path = os.path.join(seq_path, 'gt', 'gt.txt')

        # 获取所有图片
        if not os.path.exists(img_dir):
            print(f"Warning: {img_dir} does not exist. Skipping.")
            return

        imgs = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])

        # 加载 GT
        gt_data = {}  # frame_id -> list of boxes
        if split in ['train', 'val'] and os.path.exists(gt_path):
            # MOT format: frame, id, left, top, width, height, conf, class, vis
            raw_gt = np.loadtxt(gt_path, delimiter=',')
            for line in raw_gt:
                frame_idx = int(line[0])
                # 假设 QuadTrack 图片 000000.jpg 对应 frame 0 或 1，需对齐
                # 通常 MOT 挑战赛 frame 从 1 开始，我们这里存储时按 int 存

                # Filter: 只保留行人 (class 1) 或者根据需要。
                # 假设 QuadTrack gt.txt 只有行人，或者 class 位于第8列(index 7)
                # 这里的 filter 逻辑如果不确定 class id，可以先全部保留

                # Convert xywh (top-left) to cxywh (center)
                x1, y1, w, h = line[2], line[3], line[4], line[5]
                cx = x1 + w / 2
                cy = y1 + h / 2

                bbox = [cx, cy, w, h]
                obj_id = int(line[1])

                if frame_idx not in gt_data:
                    gt_data[frame_idx] = {'bboxes': [], 'ids': []}
                gt_data[frame_idx]['bboxes'].append(bbox)
                gt_data[frame_idx]['ids'].append(obj_id)

        # 处理每一帧
        for img_name in imgs:
            frame_dict = self.handle_single_frame(img_name, seq_name, split, gt_data, split_root)

            if split == 'train':
                self.train_pkl.append(frame_dict)
            elif split == 'val':
                self.val_pkl.append(frame_dict)
            elif split == 'test':
                self.test_pkl.append(frame_dict)

    def handle_single_frame(self, img_name, seq_name, split, gt_data, split_root):
        # 解析帧号
        frame_id = int(img_name.split('.')[0])

        # 路径处理
        # 目标: ./data/QuadTrack/train/0015/img1/000000.jpg
        # self.data_root usually is .../OmniTrack/data
        # We need relative path starting from project root usually ./data/...

        # 构造相对于 OmniTrack/data 的路径
        # split_root is .../data/QuadTrack/train
        # rel_split = QuadTrack/train

        rel_path = os.path.join('data', 'QuadTrack', os.path.basename(split_root), seq_name, 'img1', img_name)

        # 伪造时间戳 (microseconds)，假设 10FPS (100ms per frame) -> 100,000 us
        # 这对于时序模块很重要，保证顺序即可
        timestamp = frame_id * 100000

        cams = {
            'image_stitched': {
                'data_path': rel_path,  # 这里的路径会被 Dataset 类读取
                'type': 'image_stitched',
                'sample_data_token': f"{seq_name}_{frame_id:06d}",
                'timestamp': timestamp
            }
        }

        ego_bboxes = []
        instance_ids = []
        gt_names = []

        # MOT gt.txt 很多时候是从 Frame 1 开始，而文件名可能是 000000.jpg
        # 我们尝试匹配 frame_id (0-based) 和 frame_id + 1 (1-based)
        # 通常 000000.jpg 对应 gt 中的 0 或者 1。
        # 这里做一个简单的启发式尝试：如果 gt 中有 0 帧数据，用 0，否则尝试 frame_id + 1

        search_id = frame_id
        if frame_id not in gt_data and (frame_id + 1) in gt_data:
            search_id = frame_id + 1

        if (split in ['train', 'val']) and (search_id in gt_data):
            ego_bboxes = np.array(gt_data[search_id]['bboxes'])
            instance_ids = np.array(gt_data[search_id]['ids'])
            # 假设全都是行人
            gt_names = np.array(['pedestrian'] * len(ego_bboxes))
        else:
            ego_bboxes = np.array([])
            instance_ids = np.array([])
            gt_names = np.array([])

        infos_frame = {
            'cams': cams,
            'sweeps': [],
            'gt_boxes': ego_bboxes,
            'gt_names': gt_names,
            'gt_velocity': [],
            'instance_inds': instance_ids,
            'timestamp': timestamp,
            'token': f"{seq_name}_{frame_id:06d}",
        }
        return infos_frame

    def save_pkl(self):
        if not os.path.exists(self.save_root):
            os.makedirs(self.save_root)

        def _save(data, name):
            if len(data) > 0:
                info = {'infos': data, 'metadata': {'version': 'QuadTrack_v1.0'}}
                with open(os.path.join(self.save_root, name), 'wb') as f:
                    pkl.dump(info, f)
                print(f"Saved {name} with {len(data)} frames.")

        _save(self.train_pkl, 'QuadTrack_infos_train.pkl')
        _save(self.val_pkl, 'QuadTrack_infos_val.pkl')
        _save(self.test_pkl, 'QuadTrack_infos_test.pkl')


if __name__ == '__main__':
    # 假设脚本在 tools/ 下运行
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_root = os.path.join(project_root, 'data')

    # 结果保存路径
    save_root = os.path.join(data_root, 'QuadTrack', 'info')

    converter = QuadTrackConverter(data_root, save_root)
    converter.generate_pkl()