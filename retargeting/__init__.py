"""Asimov GMR retargeting package.

The IK engine derives from General Motion Retargeting (GMR), Copyright
2025 Yanjieze, MIT licensed — see NOTICE.
"""
from .params import IK_CONFIG_ROOT, ASSET_ROOT, ROBOT_XML_DICT, IK_CONFIG_DICT, ROBOT_BASE_DICT, VIEWER_CAM_DISTANCE_DICT
from .motion_retarget import GeneralMotionRetargeting
from .kinematics_model import KinematicsModel
