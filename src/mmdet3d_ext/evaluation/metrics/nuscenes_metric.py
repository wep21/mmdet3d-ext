import tempfile
from os import path as osp
from typing import Dict, List, Optional, Sequence, Tuple, Union

import mmengine
import numpy as np
import pyquaternion
from mmdet3d.evaluation.metrics import NuScenesMetric as _NuScenesMetric
from mmdet3d.evaluation.metrics.nuscenes_metric import output_to_nusc_box
from mmdet3d.structures import CameraInstance3DBoxes, LiDARInstance3DBoxes
from mmengine import load
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from nuscenes.eval.common.config import DetectionConfig
from nuscenes.eval.detection.constants import DETECTION_NAMES
from nuscenes.utils.data_classes import Box as NuScenesBox

from mmdet3d_ext.registry import METRICS


@METRICS.register_module()
class NuScenesMetric(_NuScenesMetric):
    """Nuscenes evaluation metric.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        metric (str or List[str]): Metrics to be evaluated. Defaults to 'bbox'.
        modality (dict): Modality to specify the sensor data used as input.
            Defaults to dict(use_camera=False, use_lidar=True).
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix will
            be used instead. Defaults to None.
        format_only (bool): Format the output results without perform
            evaluation. It is useful when you want to format the result to a
            specific format and submit it to the test server.
            Defaults to False.
        jsonfile_prefix (str, optional): The prefix of json files including the
            file path and the prefix of filename, e.g., "a/b/prefix".
            If not specified, a temp file will be created. Defaults to None.
        eval_version (str): Configuration version of evaluation.
            Defaults to 'detection_cvpr_2019'.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """

    NameMapping = {
        'movable_object.barrier': 'barrier',
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.car': 'car',
        'vehicle.construction': 'construction_vehicle',
        'vehicle.motorcycle': 'motorcycle',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'movable_object.trafficcone': 'traffic_cone',
        'vehicle.trailer': 'trailer',
        'vehicle.truck': 'truck',
    }
    DefaultAttribute = {
        'car': 'vehicle.parked',
        'pedestrian': 'pedestrian.moving',
        'trailer': 'vehicle.parked',
        'truck': 'vehicle.parked',
        'bus': 'vehicle.moving',
        'motorcycle': 'cycle.without_rider',
        'construction_vehicle': 'vehicle.parked',
        'bicycle': 'cycle.without_rider',
        'barrier': '',
        'traffic_cone': '',
    }
    # https://github.com/nutonomy/nuscenes-devkit/blob/57889ff20678577025326cfc24e57424a829be0a/python-sdk/nuscenes/eval/detection/evaluate.py#L222
    ErrNameMapping = {
        'trans_err': 'mATE',
        'scale_err': 'mASE',
        'orient_err': 'mAOE',
        'vel_err': 'mAVE',
        'attr_err': 'mAAE',
    }

    CLASSES = ('car', 'truck', 'bus', 'bicycle', 'pedestrian')

    CLASS_DETECTION_RANGE = {
        'car': 75,
        'truck': 75,
        'trailer': 75,
        'bus': 75,
        'construction_vehicle': 75,
        'bicycle': 75,
        'motorcycle': 75,
        'pedestrian': 75,
        'traffic_cone': 0,
        'barrier': 0,
    }

    def __init__(
        self,
        data_root: str,
        ann_files: list[str],
        lidar_key: str = 'LIDAR_TOP',
        metric: Union[str, List[str]] = 'bbox',
        modality: dict = dict(use_camera=False, use_lidar=True),
        prefix: Optional[str] = None,
        format_only: bool = False,
        jsonfile_prefix: Optional[str] = None,
        eval_version: str = 'detection_cvpr_2019',
        collect_device: str = 'cpu',
        backend_args: Optional[dict] = None,
    ) -> None:
        self.default_prefix = 'NuScenes metric'
        BaseMetric.__init__(self, collect_device=collect_device, prefix=prefix)
        if modality is None:
            modality = dict(use_camera=False, use_lidar=True)
        self.ann_files = ann_files
        self.lidar_key = lidar_key
        self.data_root = data_root
        self.modality = modality
        self.format_only = format_only
        if self.format_only:
            assert jsonfile_prefix is not None, 'jsonfile_prefix must be not '
            'None when format_only is True, otherwise the result files will '
            'be saved to a temp directory which will be cleanup at the end.'

        self.jsonfile_prefix = jsonfile_prefix
        self.backend_args = backend_args

        self.metrics = metric if isinstance(metric, list) else [metric]

        # self.eval_version = eval_version
        # self.eval_detection_configs = config_factory(self.eval_version)
        eval_config_dict = {
            'class_names': self.CLASSES,
            'class_range': self.CLASS_DETECTION_RANGE,
            'dist_fcn': 'center_distance',
            'dist_ths': [0.5, 1.0, 2.0, 4.0],
            'dist_th_tp': 2.0,
            'min_recall': 0.1,
            'min_precision': 0.1,
            'max_boxes_per_sample': 500,
            'mean_ap_weight': 5,
        }
        self.eval_detection_configs = DetectionConfig.deserialize(eval_config_dict)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            result = dict()
            pred_3d = data_sample['pred_instances_3d']
            pred_2d = data_sample['pred_instances']
            for attr_name in pred_3d:
                pred_3d[attr_name] = pred_3d[attr_name].to('cpu')
            result['pred_instances_3d'] = pred_3d
            for attr_name in pred_2d:
                pred_2d[attr_name] = pred_2d[attr_name].to('cpu')
            result['pred_instances'] = pred_2d
            sample_idx = data_sample['sample_idx']
            result['sample_idx'] = sample_idx
            self.results.append(result)

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (List[dict]): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        classes = self.dataset_meta['classes']
        self.version = self.dataset_meta['version']
        # load annotations
        self.data_infos = []
        for ann_file in self.ann_files:
            self.data_infos.extend(
                load(osp.join(self.data_root, ann_file), backend_args=self.backend_args)['data_list']
            )
        result_dict, tmp_dir = self.format_results(results, classes, self.jsonfile_prefix)

        metric_dict = {}

        if self.format_only:
            logger.info(f'results are saved in {osp.basename(self.jsonfile_prefix)}')
            return metric_dict

        for metric in self.metrics:
            ap_dict = self.nus_evaluate(result_dict, classes=classes, metric=metric, logger=logger)
            for result in ap_dict:
                metric_dict[result] = ap_dict[result]

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return metric_dict

    def nus_evaluate(
        self,
        result_dict: dict,
        metric: str = 'bbox',
        classes: Optional[List[str]] = None,
        logger: Optional[MMLogger] = None,
    ) -> Dict[str, float]:
        """Evaluation in Nuscenes protocol.

        Args:
            result_dict (dict): Formatted results of the dataset.
            metric (str): Metrics to be evaluated. Defaults to 'bbox'.
            classes (List[str], optional): A list of class name.
                Defaults to None.
            logger (MMLogger, optional): Logger used for printing related
                information during evaluation. Defaults to None.

        Returns:
            Dict[str, float]: Results of each evaluation metric.
        """
        metric_dict = dict()
        scene_ids = [osp.splitext(osp.basename(annfile))[0].split('_')[-1] for annfile in self.ann_files]
        for name in result_dict:
            print(f'Evaluating bboxes of {name}')
            ret_dict = self._evaluate_single(result_dict[name], scene_ids, classes=classes, result_name=name)
            metric_dict.update(ret_dict)
        return metric_dict

    def _evaluate_single(
        self,
        result_path: str,
        scene_ids: list[str],
        classes: Optional[List[str]] = None,
        result_name: str = 'pred_instances_3d',
    ) -> Dict[str, float]:
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            classes (List[str], optional): A list of class name.
                Defaults to None.
            result_name (str): Result name in the metric prefix.
                Defaults to 'pred_instances_3d'.

        Returns:
            Dict[str, float]: Dictionary of evaluation details.
        """
        from nuscenes import NuScenes

        from ..functional.nusc_eval import NuScenesEval

        output_dir = osp.join(*osp.split(result_path)[:-1])
        nuscs = dict()
        for id in scene_ids:
            nusc = NuScenes(version='annotation', dataroot=osp.join(self.data_root, id), verbose=False)
            nuscs[nusc.scene[0]['token']] = nusc
        nusc_eval = NuScenesEval(
            nuscs,
            config=self.eval_detection_configs,
            result_path=result_path,
            output_dir=output_dir,
            lidar_key=self.lidar_key,
            verbose=False,
        )
        nusc_eval.main(render_curves=False)

        # record metrics
        metrics = mmengine.load(osp.join(output_dir, 'metrics_summary.json'))
        detail = dict()
        metric_prefix = f'{result_name}_NuScenes'
        for name in classes:
            for k, v in metrics['label_aps'][name].items():
                val = float(f'{v:.4f}')
                detail[f'{metric_prefix}/{name}_AP_dist_{k}'] = val
            for k, v in metrics['label_tp_errors'][name].items():
                val = float(f'{v:.4f}')
                detail[f'{metric_prefix}/{name}_{k}'] = val
            for k, v in metrics['tp_errors'].items():
                val = float(f'{v:.4f}')
                detail[f'{metric_prefix}/{self.ErrNameMapping[k]}'] = val

        detail[f'{metric_prefix}/NDS'] = metrics['nd_score']
        detail[f'{metric_prefix}/mAP'] = metrics['mean_ap']
        return detail

    def format_results(
        self, results: List[dict], classes: Optional[List[str]] = None, jsonfile_prefix: Optional[str] = None
    ) -> Tuple[dict, Union[tempfile.TemporaryDirectory, None]]:
        """Format the mmdet3d results to standard NuScenes json file.

        Args:
            results (List[dict]): Testing results of the dataset.
            classes (List[str], optional): A list of class name.
                Defaults to None.
            jsonfile_prefix (str, optional): The prefix of json files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Defaults to None.

        Returns:
            tuple: Returns (result_dict, tmp_dir), where ``result_dict`` is a
            dict containing the json filepaths, ``tmp_dir`` is the temporal
            directory created for saving json files when ``jsonfile_prefix`` is
            not specified.
        """
        assert isinstance(results, list), 'results must be a list'

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_dict = dict()
        sample_idx_list = [result['sample_idx'] for result in results]

        for name in results[0]:
            if 'pred' in name and '3d' in name and name[0] != '_':
                print(f'\nFormating bboxes of {name}')
                results_ = [out[name] for out in results]
                tmp_file_ = osp.join(jsonfile_prefix, name)
                box_type_3d = type(results_[0]['bboxes_3d'])
                if box_type_3d == LiDARInstance3DBoxes:
                    result_dict[name] = self._format_lidar_bbox(results_, sample_idx_list, classes, tmp_file_)
                elif box_type_3d == CameraInstance3DBoxes:
                    result_dict[name] = self._format_camera_bbox(results_, sample_idx_list, classes, tmp_file_)

        return result_dict, tmp_dir
    

    def _format_lidar_bbox(self,
                           results: List[dict],
                           sample_idx_list: List[int],
                           classes: Optional[List[str]] = None,
                           jsonfile_prefix: Optional[str] = None) -> str:
        """Convert the results to the standard format.

        Args:
            results (List[dict]): Testing results of the dataset.
            sample_idx_list (List[int]): List of result sample idx.
            classes (List[str], optional): A list of class name.
                Defaults to None.
            jsonfile_prefix (str, optional): The prefix of the output jsonfile.
                You can specify the output directory/filename by modifying the
                jsonfile_prefix. Defaults to None.

        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}

        print('Start to convert detection format...')
        for i, det in enumerate(mmengine.track_iter_progress(results)):
            annos = []
            boxes, attrs = output_to_nusc_box(det)
            sample_idx = sample_idx_list[i]
            sample_token = self.data_infos[sample_idx]['token']
            boxes = lidar_nusc_box_to_global(self.data_infos[sample_idx],
                                             boxes, classes,
                                             self.eval_detection_configs)
            for i, box in enumerate(boxes):
                name = classes[box.label]
                if np.sqrt(box.velocity[0]**2 + box.velocity[1]**2) > 0.2:
                    if name in [
                            'car',
                            'construction_vehicle',
                            'bus',
                            'truck',
                            'trailer',
                    ]:
                        attr = 'vehicle.moving'
                    elif name in ['bicycle', 'motorcycle']:
                        attr = 'cycle.with_rider'
                    else:
                        attr = self.DefaultAttribute[name]
                else:
                    if name in ['pedestrian']:
                        attr = 'pedestrian.standing'
                    elif name in ['bus']:
                        attr = 'vehicle.stopped'
                    else:
                        attr = self.DefaultAttribute[name]

                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                    detection_name=name,
                    detection_score=box.score,
                    attribute_name=attr)
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        }
        mmengine.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        print(f'Results writes to {res_path}')
        mmengine.dump(nusc_submissions, res_path)
        return res_path

def lidar_nusc_box_to_global(
        info: dict, boxes: List[NuScenesBox], classes: List[str],
        eval_configs: DetectionConfig) -> List[NuScenesBox]:
    """Convert the box from ego to global coordinate.

    Args:
        info (dict): Info for a specific sample data, including the calibration
            information.
        boxes (List[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        classes (List[str]): Mapped classes in the evaluation.
        eval_configs (:obj:`DetectionConfig`): Evaluation configuration object.

    Returns:
        List[:obj:`DetectionConfig`]: List of standard NuScenesBoxes in the
        global coordinate.
    """
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        lidar2ego = np.array(info['lidar_points']['lidar2ego'])
        box.rotate(
            pyquaternion.Quaternion(matrix=lidar2ego, rtol=1e-05, atol=1e-07))
        box.translate(lidar2ego[:3, 3])
        # filter det in ego.
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        if DETECTION_NAMES[box.label] not in classes:
            continue
        box.label = classes.index(DETECTION_NAMES[box.label])
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue
        # Move box to global coord system
        ego2global = np.array(info['ego2global'])
        box.rotate(
            pyquaternion.Quaternion(matrix=ego2global, rtol=1e-05, atol=1e-07))
        box.translate(ego2global[:3, 3])
        box_list.append(box)
    return box_list
