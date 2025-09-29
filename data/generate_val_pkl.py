import os
import pickle
import numpy as np

class ThreeDMatchDataset:
    def __init__(self, root_dir, split='train', voxel_size=0.05, val_scenes=None):
        self.root_dir = root_dir  # 3DMatch 数据集的根目录
        self.split = split        # 数据集分割 ('train', 'val', 'test')
        self.voxel_size = voxel_size  # 体素大小
        self.val_scenes = val_scenes  # 验证集的场景（从训练集中挑选）
        self.data_pairs = []      # 保存点云对的元数据

        # 加载对应 split 的数据对信息
        self.load_data_pairs()

    def load_data_pairs(self):
        """
        加载数据集中的点云对及其元数据。此处假设验证集是从训练集中选取的场景。
        """
        pairs_file = os.path.join(self.root_dir, f'{self.split}_pairs.txt')
        assert os.path.exists(pairs_file), f"{pairs_file} 不存在。"

        with open(pairs_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            # 假设每行是 点云1 路径, 点云2 路径, ground truth 变换矩阵
            p1, p2, transform = line.strip().split()
            
            # 根据选定的验证集场景进行筛选
            if self.val_scenes is None or any(scene in p1 for scene in self.val_scenes):
                self.data_pairs.append({
                    'src': p1,  # 源点云路径
                    'tgt': p2,  # 目标点云路径
                    'transform': np.fromstring(transform, sep=' ').reshape(4, 4)  # 位姿矩阵
                })

    def process_and_save(self):
        """
        处理点云对并生成 val.pkl 文件
        """
        processed_data = []
        for pair in self.data_pairs:
            src_path = os.path.join(self.root_dir, pair['src'])
            tgt_path = os.path.join(self.root_dir, pair['tgt'])

            # 加载点云
            src_pc = self.load_point_cloud(src_path)
            tgt_pc = self.load_point_cloud(tgt_path)

            # 下采样点云
            src_pc_down = self.voxel_down_sample(src_pc, self.voxel_size)
            tgt_pc_down = self.voxel_down_sample(tgt_pc, self.voxel_size)

            processed_data.append({
                'src': src_pc_down,       # 下采样后的源点云
                'tgt': tgt_pc_down,       # 下采样后的目标点云
                'transform': pair['transform'],  # ground truth 变换矩阵
            })

        # 保存到 val.pkl 文件
        with open(f'val.pkl', 'wb') as f:
            pickle.dump(processed_data, f)
        print(f'val.pkl 文件生成完毕。')

    def load_point_cloud(self, file_path):
        """
        加载单个点云文件 (假设点云格式为 .ply 或 .txt)
        """
        return np.loadtxt(file_path)

    def voxel_down_sample(self, point_cloud, voxel_size):
        """
        对点云进行体素下采样，减少点数
        """
        indices = np.floor(point_cloud / voxel_size).astype(np.int32)
        _, unique_indices = np.unique(indices, axis=0, return_index=True)
        return point_cloud[unique_indices]


if __name__ == '__main__':
    # 验证集场景假设是 3DMatch 训练集中部分场景，例如 '7-scenes-redkitchen' 和 'sun3d-hotel-umd'
    val_scenes = ['7-scenes-redkitchen', 'sun3d-hotel-umd']

    # 3DMatch 数据集的根目录
    root_dir = './3dmatch'

    # 初始化数据集并生成 val.pkl
    dataset = ThreeDMatchDataset(root_dir=root_dir, split='train', voxel_size=0.05, val_scenes=val_scenes)
    dataset.process_and_save()
