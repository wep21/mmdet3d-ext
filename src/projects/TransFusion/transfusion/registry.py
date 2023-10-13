from mmengine.registry import Registry
from mmdet3d.registry import MODELS as MMDET3D_MODELS

MODELS = Registry(
    'model', parent=MMDET3D_MODELS, scope='transfusion')