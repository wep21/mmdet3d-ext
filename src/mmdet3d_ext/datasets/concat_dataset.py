from mmdet3d.registry import DATASETS
from mmengine.dataset.dataset_wrapper import ConcatDataset as _ConcatDataset


@DATASETS.register_module()
class ConcatDataset(_ConcatDataset):
    def get_cat_ids(self, idx: int) -> list[int]:
        dataset_idx, sample_idx = self._get_ori_dataset_idx(idx)
        return self.datasets[dataset_idx].get_cat_ids(sample_idx)