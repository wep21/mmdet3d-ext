import copy
import os.path as osp
import pickle
from pathlib import Path

import mmengine
import numpy as np
from mmdet3d.registry import DATASETS
from mmdet3d.structures.ops import box_np_ops
from mmengine import track_iter_progress
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from tools.dataset_converters.nuscenes_converter import obtain_sensor2top
from tools.dataset_converters.update_infos_to_v2 import (
    clear_data_info_unused_keys,
    clear_instance_unused_keys,
    convert_quaternion_to_matrix,
    # generate_nuscenes_camera_instances,
    get_empty_img_info,
    get_empty_instance,
    get_empty_standard_data_info,
    get_single_lidar_sweep,
)

from mmdet3d_ext.datasets import MonoNuScenesDataset  # noqa

NameMapping = {
    'vehicle.car': 'car',
    'vehicle.construction': 'truck',
    'vehicle.emergency (ambulance & police)': 'car',
    'vehicle.motorcycle': 'bicycle',
    'vehicle.trailer': 'truck',
    'vehicle.truck': 'truck',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus (bendy & rigid)': 'bus',
    'pedestrian.adult': 'pedestrian',
    'pedestrian.child': 'pedestrian',
    'pedestrian.construction_worker': 'pedestrian',
    'pedestrian.personal_mobility': 'pedestrian',
    'pedestrian.police_officer': 'pedestrian',
    'pedestrian.stroller': 'pedestrian',
    'pedestrian.wheelchair': 'pedestrian',
    'movable_object.barrier': 'barrier',
    'movable_object.debris': 'debris',
    'movable_object.pushable_pullable': 'pushable_pullable',
    'movable_object.trafficcone': 'traffic_cone',
    'movable_object.traffic_cone': 'traffic_cone',
    'animal': 'animal',
    'static_object.bicycle_rack': 'bicycle_rack',
    'car': 'car',
    'truck': 'truck',
    'bus': 'bus',
    'trailer': 'truck',
    'motorcycle': 'bicycle',
    'bicycle': 'bicycle',
    'police_car': 'car',
    'pedestrian': 'pedestrian',
    'police_officer': 'pedestrian',
    'forklift': 'car',
    'construction_worker': 'pedestrian',
    'stroller': 'pedestrian',
}


def create_mono_nuscenes_infos(
    root_path: str,
    dataset_id: str,
    info_prefix: str = 'nuscenes',
    max_sweeps: int = 10,
    lidar_type: str = 'LIDAR_TOP',
    camera_types: list[str] = [
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_FRONT_LEFT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT',
        ]
):
    """Create info file of nuscene dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str, optional): Version of the data.
            Default: 'v1.0-trainval'.
        max_sweeps (int, optional): Max number of sweeps.
            Default: 10.
    """
    nusc = NuScenes(version='annotation', dataroot=osp.join(root_path, dataset_id), verbose=True)
    infos = fill_info(nusc, max_sweeps=max_sweeps, lidar_type=lidar_type, camera_types=camera_types)
    metadata = {'version': dataset_id}
    
    print(
        f'sample: {len(infos)}'
    )
    data = {'infos': infos, 'metadata': metadata}
    info_path = osp.join(root_path, f'{info_prefix}_infos_{dataset_id}.pkl')
    mmengine.dump(data, info_path)
    update_mono_nuscenes_infos(info_path, root_path, camera_types=camera_types)


def fill_info(nusc: NuScenes, max_sweeps: int = 10, lidar_type: str = 'LIDAR_TOP', camera_types: list[str] = [
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_FRONT_LEFT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT',
        ]) -> list[dict]:
    """Generate the train/val infos from the raw data.

    Args:
        nusc (:obj:`NuScenes`): Dataset class in the nuScenes dataset.
        train_scenes (list[str]): Basic information of training scenes.
        val_scenes (list[str]): Basic information of validation scenes.
        test (bool, optional): Whether use the test mode. In test mode, no
            annotations can be accessed. Default: False.
        max_sweeps (int, optional): Max number of sweeps. Default: 10.

    Returns:
        tuple[list[dict]]: Information of training set and validation set
            that will be saved to the info file.
    """
    infos: list[dict] = []
    for sample in mmengine.track_iter_progress(nusc.sample):
        if not sample['data']:
            continue
        lidar_token = sample['data'][lidar_type]
        sd_rec = nusc.get('sample_data', sample['data'][lidar_type])
        cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

        mmengine.check_file_exist(lidar_path)

        info = {
            'lidar_path': lidar_path,
            'num_features': 5,
            'token': sample['token'],
            'sweeps': [],
            'cams': {},
            'lidar2ego_translation': cs_record['translation'],
            'lidar2ego_rotation': cs_record['rotation'],
            'ego2global_translation': pose_record['translation'],
            'ego2global_rotation': pose_record['rotation'],
            'timestamp': sample['timestamp'],
        }

        l2e_r = info['lidar2ego_rotation']
        l2e_t = info['lidar2ego_translation']
        e2g_r = info['ego2global_rotation']
        e2g_t = info['ego2global_translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        for cam in camera_types:
            cam_token = sample['data'][cam]
            cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
            cam_info = obtain_sensor2top(
                nusc, cam_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, cam
            )
            cam_info.update(cam_intrinsic=cam_intrinsic)
            info['cams'].update({cam: cam_info})

        # obtain sweeps for a single key-frame
        sd_rec = nusc.get('sample_data', sample['data'][lidar_type])
        sweeps = []
        while len(sweeps) < max_sweeps:
            if not sd_rec['prev'] == '':
                sweep = obtain_sensor2top(
                    nusc, sd_rec['prev'], l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, 'lidar'
                )
                sweeps.append(sweep)
                sd_rec = nusc.get('sample_data', sd_rec['prev'])
            else:
                break
        info['sweeps'] = sweeps

        annotations = [
            nusc.get('sample_annotation', token) for token in sample['anns']
        ]
        locs = np.array([b.center for b in boxes]).reshape(-1, 3)
        dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
        rots = np.array([b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(
            -1, 1
        )
        velocity = np.array(
            [nusc.box_velocity(token)[:2] for token in sample['anns']]
        )
        valid_flag = np.array(
            [
                (anno['num_lidar_pts'] + anno['num_radar_pts']) > 0
                for anno in annotations
            ],
            dtype=bool,
        ).reshape(-1)
        # convert velo from global to lidar
        for i in range(len(boxes)):
            velo = np.array([*velocity[i], 0.0])
            velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
            velocity[i] = velo[:2]

        names = [b.name for b in boxes]
        for i in range(len(names)):
            if names[i] in NameMapping:
                names[i] = NameMapping[names[i]]
        names = np.array(names)
        # we need to convert box size to
        # the format of our lidar coordinate system
        # which is x_size, y_size, z_size (corresponding to l, w, h)
        gt_boxes = np.concatenate([locs, dims[:, [1, 0, 2]], rots], axis=1)
        assert len(gt_boxes) == len(
            annotations
        ), f'{len(gt_boxes)}, {len(annotations)}'
        info['gt_boxes'] = gt_boxes
        info['gt_names'] = names
        info['gt_velocity'] = velocity.reshape(-1, 2)
        info['num_lidar_pts'] = np.array([a['num_lidar_pts'] for a in annotations])
        info['num_radar_pts'] = np.array([a['num_radar_pts'] for a in annotations])
        info['valid_flag'] = valid_flag

        if 'lidarseg' in nusc.table_names:
            info['pts_semantic_mask_path'] = osp.join(
                nusc.dataroot, nusc.get('lidarseg', lidar_token)['filename']
            )

        infos.append(info)

    return infos

def update_mono_nuscenes_infos(pkl_path: str, out_dir: str, camera_types: list[str] = [
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_FRONT_LEFT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_BACK_RIGHT',
    ], classes: tuple[str] = ('car', 'truck', 'bus', 'bicycle', 'pedestrian')):
    print(f'{pkl_path} will be modified.')
    if out_dir in pkl_path:
        print(f'Warning, you may overwriting '
              f'the original data {pkl_path}.')
    print(f'Reading from input file: {pkl_path}.')
    data_list = mmengine.load(pkl_path)
    METAINFO = {
        'classes': classes,
    }
    NuScenes(
        version='annotation',
        dataroot=osp.join(osp.dirname(pkl_path), data_list['metadata']['version']),
        verbose=True)

    print('Start updating:')
    converted_list = []
    for i, ori_info_dict in enumerate(
            mmengine.track_iter_progress(data_list['infos'])):
        temp_data_info = get_empty_standard_data_info(
            camera_types=camera_types)
        temp_data_info['sample_idx'] = i
        temp_data_info['token'] = ori_info_dict['token']
        temp_data_info['ego2global'] = convert_quaternion_to_matrix(
            ori_info_dict['ego2global_rotation'],
            ori_info_dict['ego2global_translation'])
        temp_data_info['lidar_points']['num_pts_feats'] = ori_info_dict.get(
            'num_features', 5)
        temp_data_info['lidar_points']['lidar_path'] = Path(
            ori_info_dict['lidar_path']).name
        temp_data_info['lidar_points'][
            'lidar2ego'] = convert_quaternion_to_matrix(
                ori_info_dict['lidar2ego_rotation'],
                ori_info_dict['lidar2ego_translation'])
        # bc-breaking: Timestamp has divided 1e6 in pkl infos.
        temp_data_info['timestamp'] = ori_info_dict['timestamp'] / 1e6
        for ori_sweep in ori_info_dict['sweeps']:
            temp_lidar_sweep = get_single_lidar_sweep()
            temp_lidar_sweep['lidar_points'][
                'lidar2ego'] = convert_quaternion_to_matrix(
                    ori_sweep['sensor2ego_rotation'],
                    ori_sweep['sensor2ego_translation'])
            temp_lidar_sweep['ego2global'] = convert_quaternion_to_matrix(
                ori_sweep['ego2global_rotation'],
                ori_sweep['ego2global_translation'])
            lidar2sensor = np.eye(4)
            rot = ori_sweep['sensor2lidar_rotation']
            trans = ori_sweep['sensor2lidar_translation']
            lidar2sensor[:3, :3] = rot.T
            lidar2sensor[:3, 3:4] = -1 * np.matmul(rot.T, trans.reshape(3, 1))
            temp_lidar_sweep['lidar_points'][
                'lidar2sensor'] = lidar2sensor.astype(np.float32).tolist()
            temp_lidar_sweep['timestamp'] = ori_sweep['timestamp'] / 1e6
            temp_lidar_sweep['lidar_points']['lidar_path'] = ori_sweep[
                'data_path']
            temp_lidar_sweep['sample_data_token'] = ori_sweep[
                'sample_data_token']
            temp_data_info['lidar_sweeps'].append(temp_lidar_sweep)
        temp_data_info['images'] = {}
        for cam in ori_info_dict['cams']:
            empty_img_info = get_empty_img_info()
            empty_img_info['img_path'] = Path(
                ori_info_dict['cams'][cam]['data_path']).name
            empty_img_info['cam2img'] = ori_info_dict['cams'][cam][
                'cam_intrinsic'].tolist()
            empty_img_info['sample_data_token'] = ori_info_dict['cams'][cam][
                'sample_data_token']
            # bc-breaking: Timestamp has divided 1e6 in pkl infos.
            empty_img_info[
                'timestamp'] = ori_info_dict['cams'][cam]['timestamp'] / 1e6
            empty_img_info['cam2ego'] = convert_quaternion_to_matrix(
                ori_info_dict['cams'][cam]['sensor2ego_rotation'],
                ori_info_dict['cams'][cam]['sensor2ego_translation'])
            lidar2sensor = np.eye(4)
            rot = ori_info_dict['cams'][cam]['sensor2lidar_rotation']
            trans = ori_info_dict['cams'][cam]['sensor2lidar_translation']
            lidar2sensor[:3, :3] = rot.T
            lidar2sensor[:3, 3:4] = -1 * np.matmul(rot.T, trans.reshape(3, 1))
            empty_img_info['lidar2cam'] = lidar2sensor.astype(
                np.float32).tolist()
            temp_data_info['images'][cam] = empty_img_info
        ignore_class_name = set()
        if 'gt_boxes' in ori_info_dict:
            num_instances = ori_info_dict['gt_boxes'].shape[0]
            for i in range(num_instances):
                empty_instance = get_empty_instance()
                empty_instance['bbox_3d'] = ori_info_dict['gt_boxes'][
                    i, :].tolist()
                if ori_info_dict['gt_names'][i] in METAINFO['classes']:
                    empty_instance['bbox_label'] = METAINFO['classes'].index(
                        ori_info_dict['gt_names'][i])
                else:
                    ignore_class_name.add(ori_info_dict['gt_names'][i])
                    empty_instance['bbox_label'] = -1
                empty_instance['bbox_label_3d'] = copy.deepcopy(
                    empty_instance['bbox_label'])
                empty_instance['velocity'] = ori_info_dict['gt_velocity'][
                    i, :].tolist()
                empty_instance['num_lidar_pts'] = ori_info_dict[
                    'num_lidar_pts'][i]
                empty_instance['num_radar_pts'] = ori_info_dict[
                    'num_radar_pts'][i]
                empty_instance['bbox_3d_isvalid'] = ori_info_dict[
                    'valid_flag'][i]
                empty_instance = clear_instance_unused_keys(empty_instance)
                temp_data_info['instances'].append(empty_instance)
            # temp_data_info[
            #     'cam_instances'] = generate_nuscenes_camera_instances(
            #         ori_info_dict, nusc)
        if 'pts_semantic_mask_path' in ori_info_dict:
            temp_data_info['pts_semantic_mask_path'] = Path(
                ori_info_dict['pts_semantic_mask_path']).name
        temp_data_info, _ = clear_data_info_unused_keys(temp_data_info)
        converted_list.append(temp_data_info)
    pkl_name = Path(pkl_path).name
    out_path = osp.join(out_dir, pkl_name)
    print(f'Writing to output file: {out_path}.')
    print(f'ignore classes: {ignore_class_name}')

    metainfo = dict()
    metainfo['categories'] = {k: i for i, k in enumerate(METAINFO['classes'])}
    # if ignore_class_name:
    #     for ignore_class in ignore_class_name:
    #         metainfo['categories'][ignore_class] = -1
    metainfo['dataset'] = 'nuscenes'
    metainfo['version'] = data_list['metadata']['version']
    metainfo['info_version'] = '1.1'
    converted_data_info = dict(metainfo=metainfo, data_list=converted_list)

    mmengine.dump(converted_data_info, out_path, 'pkl')


def create_groundtruth_database(data_path,
                                dataset_ids,
                                lidar_type,
                                used_classes=None,
                                database_save_path=None,
                                db_info_save_path=None):
    """Given the raw data, generate the ground truth database.

    Args:
        dataset_class_name (str): Name of the input dataset.
        data_path (str): Path of the data.
        info_prefix (str): Prefix of the info file.
        info_path (str, optional): Path of the info file.
            Default: None.
        mask_anno_path (str, optional): Path of the mask_anno.
            Default: None.
        used_classes (list[str], optional): Classes have been used.
            Default: None.
        database_save_path (str, optional): Path to save database.
            Default: None.
        db_info_save_path (str, optional): Path to save db_info.
            Default: None.
        relative_path (bool, optional): Whether to use relative path.
            Default: True.
        with_mask (bool, optional): Whether to use mask.
            Default: False.
    """
    print('Create GT Database of MonoNuScenesDataset')
    datasets = []
    for dataset_id in dataset_ids:
        dataset_cfg = dict(
            type='MonoNuScenesDataset', _scope_='mmdet3d', data_root=data_path, ann_file=osp.join(data_path, f'nuscenes_infos_{dataset_id}.pkl'))

        dataset_cfg.update(
            metainfo=dict(version='dbinfo'),
            use_valid_flag=True,
            data_prefix=dict(
                pts=f'{dataset_id}/data/{lidar_type}', img='', sweeps=f'{dataset_id}/data/{lidar_type}'),
            pipeline=[
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=5,
                    use_dim=5),
                dict(
                    type='LoadPointsFromMultiSweeps',
                    sweeps_num=10,
                    use_dim=[0, 1, 2, 3, 4],
                    pad_empty_sweeps=True,
                    remove_close=True),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True)
            ])
        datasets.append(dataset_cfg)

    dataset = DATASETS.build(dict(type='ConcatDataset', datasets=datasets))

    if database_save_path is None:
        database_save_path = osp.join(data_path, 'nuscenes_gt_database')
    if db_info_save_path is None:
        db_info_save_path = osp.join(data_path, 'nuscenes_dbinfos_train.pkl')
    mmengine.mkdir_or_exist(database_save_path)
    all_db_infos = dict()

    group_counter = 0
    for j in track_iter_progress(list(range(len(dataset)))):
        data_info = dataset.get_data_info(j)
        example = dataset.datasets[0].pipeline(data_info)
        annos = example['ann_info']
        image_idx = example['sample_idx']
        points = example['points'].numpy()
        gt_boxes_3d = annos['gt_bboxes_3d'].numpy()
        names = [dataset.metainfo['classes'][i] for i in annos['gt_labels_3d']]
        group_dict = dict()
        if 'group_ids' in annos:
            group_ids = annos['group_ids']
        else:
            group_ids = np.arange(gt_boxes_3d.shape[0], dtype=np.int64)
        difficulty = np.zeros(gt_boxes_3d.shape[0], dtype=np.int32)
        if 'difficulty' in annos:
            difficulty = annos['difficulty']

        num_obj = gt_boxes_3d.shape[0]
        point_indices = box_np_ops.points_in_rbbox(points, gt_boxes_3d)

        for i in range(num_obj):
            filename = f'{image_idx}_{names[i]}_{i}.bin'
            abs_filepath = osp.join(database_save_path, filename)
            rel_filepath = osp.join('nuscenes_gt_database', filename)

            # save point clouds and image patches for each object
            gt_points = points[point_indices[:, i]]
            gt_points[:, :3] -= gt_boxes_3d[i, :3]

            with open(abs_filepath, 'w') as f:
                gt_points.tofile(f)

            if (used_classes is None) or names[i] in used_classes:
                db_info = {
                    'name': names[i],
                    'path': rel_filepath,
                    'image_idx': image_idx,
                    'gt_idx': i,
                    'box3d_lidar': gt_boxes_3d[i],
                    'num_points_in_gt': gt_points.shape[0],
                    'difficulty': difficulty[i],
                }
                local_group_id = group_ids[i]
                # if local_group_id >= 0:
                if local_group_id not in group_dict:
                    group_dict[local_group_id] = group_counter
                    group_counter += 1
                db_info['group_id'] = group_dict[local_group_id]
                if 'score' in annos:
                    db_info['score'] = annos['score'][i]
                if names[i] in all_db_infos:
                    all_db_infos[names[i]].append(db_info)
                else:
                    all_db_infos[names[i]] = [db_info]

    for k, v in all_db_infos.items():
        print(f'load {len(v)} {k} database infos')

    with open(db_info_save_path, 'wb') as f:
        pickle.dump(all_db_infos, f)


class GTDatabaseCreater:
    """Given the raw data, generate the ground truth database. This is the
    parallel version. For serialized version, please refer to
    `create_groundtruth_database`

    Args:
        data_path (str): Path of the data.
        mask_anno_path (str, optional): Path of the mask_anno.
            Default: None.
        used_classes (list[str], optional): Classes have been used.
            Default: None.
        database_save_path (str, optional): Path to save database.
            Default: None.
        db_info_save_path (str, optional): Path to save db_info.
            Default: None.
        relative_path (bool, optional): Whether to use relative path.
            Default: True.
        with_mask (bool, optional): Whether to use mask.
            Default: False.
        num_worker (int, optional): the number of parallel workers to use.
            Default: 8.
    """

    def __init__(self,
                 data_path,
                 dataset_ids,
                 lidar_type,
                 mask_anno_path=None,
                 used_classes=None,
                 database_save_path=None,
                 db_info_save_path=None,
                 relative_path=True,
                 add_rgb=False,
                 lidar_only=False,
                 bev_only=False,
                 coors_range=None,
                 with_mask=False,
                 num_worker=8) -> None:
        self.data_path = data_path
        self.dataset_ids = dataset_ids
        self.lidar_type = lidar_type
        self.mask_anno_path = mask_anno_path
        self.used_classes = used_classes
        self.database_save_path = database_save_path
        self.db_info_save_path = db_info_save_path
        self.relative_path = relative_path
        self.add_rgb = add_rgb
        self.lidar_only = lidar_only
        self.bev_only = bev_only
        self.coors_range = coors_range
        self.with_mask = with_mask
        self.num_worker = num_worker
        self.pipeline = None

    def create_single(self, input_dict):
        group_counter = 0
        single_db_infos = dict()
        example = self.pipeline(input_dict)
        annos = example['ann_info']
        # image_idx = example['sample_idx']
        image_idx = example['idx']
        points = example['points'].numpy()
        gt_boxes_3d = annos['gt_bboxes_3d'].numpy()
        names = [
            self.dataset.metainfo['classes'][i] for i in annos['gt_labels_3d']
        ]
        group_dict = dict()
        if 'group_ids' in annos:
            group_ids = annos['group_ids']
        else:
            group_ids = np.arange(gt_boxes_3d.shape[0], dtype=np.int64)
        difficulty = np.zeros(gt_boxes_3d.shape[0], dtype=np.int32)
        if 'difficulty' in annos:
            difficulty = annos['difficulty']

        num_obj = gt_boxes_3d.shape[0]
        point_indices = box_np_ops.points_in_rbbox(points, gt_boxes_3d)

        for i in range(num_obj):
            filename = f'{image_idx}_{names[i]}_{i}.bin'
            abs_filepath = osp.join(self.database_save_path, filename)
            rel_filepath = osp.join('nuscenes_gt_database',
                                    filename)

            # save point clouds and image patches for each object
            gt_points = points[point_indices[:, i]]
            gt_points[:, :3] -= gt_boxes_3d[i, :3]

            with open(abs_filepath, 'w') as f:
                gt_points.tofile(f)

            if (self.used_classes is None) or names[i] in self.used_classes:
                db_info = {
                    'name': names[i],
                    'path': rel_filepath,
                    'image_idx': image_idx,
                    'gt_idx': i,
                    'box3d_lidar': gt_boxes_3d[i],
                    'num_points_in_gt': gt_points.shape[0],
                    'difficulty': difficulty[i],
                }
                local_group_id = group_ids[i]
                # if local_group_id >= 0:
                if local_group_id not in group_dict:
                    group_dict[local_group_id] = group_counter
                    group_counter += 1
                db_info['group_id'] = group_dict[local_group_id]
                if 'score' in annos:
                    db_info['score'] = annos['score'][i]
                if names[i] in single_db_infos:
                    single_db_infos[names[i]].append(db_info)
                else:
                    single_db_infos[names[i]] = [db_info]

        return single_db_infos

    def create(self):
        print('Create GT Database of MonoNuScenesDataset')
        datasets = []
        for dataset_id in self.dataset_ids:
            dataset_cfg = dict(
                type='MonoNuScenesDataset', _scope_='mmdet3d', data_root=self.data_path, ann_file=osp.join(self.data_path, f'nuscenes_infos_{dataset_id}.pkl'))

            dataset_cfg.update(
                metainfo=dict(version='dbinfo'),
                use_valid_flag=True,
                data_prefix=dict(
                    pts=f'{dataset_id}/data/{self.lidar_type}', img='', sweeps=f'{dataset_id}/data/{self.lidar_type}'),
                pipeline=[
                    dict(
                        type='LoadPointsFromFile',
                        coord_type='LIDAR',
                        load_dim=5,
                        use_dim=5),
                    dict(
                        type='LoadPointsFromMultiSweeps',
                        sweeps_num=10,
                        use_dim=[0, 1, 2, 3, 4],
                        pad_empty_sweeps=True,
                        remove_close=True),
                    dict(
                        type='LoadAnnotations3D',
                        with_bbox_3d=True,
                        with_label_3d=True)
                ])
            datasets.append(dataset_cfg)

        self.dataset = DATASETS.build(dict(type='ConcatDataset', datasets=datasets))

        self.pipeline = self.dataset.datasets[0].pipeline
        if self.database_save_path is None:
            self.database_save_path = osp.join(
                self.data_path, 'nuscenes_gt_database')
        if self.db_info_save_path is None:
            self.db_info_save_path = osp.join(
                self.data_path, 'nuscenes_dbinfos_train.pkl')
        mmengine.mkdir_or_exist(self.database_save_path)

        def loop_dataset(i):
            input_dict = self.dataset.get_data_info(i)
            input_dict['box_type_3d'] = self.dataset.datasets[0].box_type_3d
            input_dict['box_mode_3d'] = self.dataset.datasets[0].box_mode_3d
            input_dict['idx'] = i 
            return input_dict

        multi_db_infos = mmengine.track_parallel_progress(
            self.create_single,
            ((loop_dataset(i)
              for i in range(len(self.dataset))), len(self.dataset)),
            self.num_worker)
        print('Make global unique group id')
        group_counter_offset = 0
        all_db_infos = dict()
        for single_db_infos in track_iter_progress(multi_db_infos):
            group_id = -1
            for name, name_db_infos in single_db_infos.items():
                for db_info in name_db_infos:
                    group_id = max(group_id, db_info['group_id'])
                    db_info['group_id'] += group_counter_offset
                if name not in all_db_infos:
                    all_db_infos[name] = []
                all_db_infos[name].extend(name_db_infos)
            group_counter_offset += (group_id + 1)

        for k, v in all_db_infos.items():
            print(f'load {len(v)} {k} database infos')

        with open(self.db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)