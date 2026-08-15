"""SMPL-X / SMPL+H loading and per-frame target extraction.

Based on General Motion Retargeting (GMR), Copyright 2025 Yanjieze, MIT
licensed — see NOTICE. The SMPL+H AMASS adapter (load_smplh_amass_file),
measured-height scaling and FRAMERATE_OVERRIDES are this project's.
"""
import os

import numpy as np
import smplx
import torch
from scipy.spatial.transform import Rotation as R
from smplx.joint_names import JOINT_NAMES
from scipy.interpolate import interp1d

from retargeting.utils.quat import quat_mul

# AMASS datasets whose packaged `mocap_framerate` is WRONG, mapped to the true
# capture rate. Keyed by the dataset directory name in the AMASS path.
# BMLhandball: the source publication (Helm/Troje/Munzert handball penalty
# throw database, Data in Brief 2017) states 240 Hz Vicon capture; AMASS ships
# it stamped 120, so every clip plays at exactly half speed. Verified
# numerically 2026-08-12: ballistic pelvis windows fit apparent gravity
# ~-1.4 m/s^2 (needs -9.81 at the claimed rate), expert arm peaks read 33%
# BELOW lay throwers, walking cadence ~0.75 steps/s. Releases up to tune_v7
# were built on the slow data.
FRAMERATE_OVERRIDES = {
    "BMLhandball": 240.0,
}


def true_frame_rate(smplh_file, claimed_rate):
    """The actual capture rate for this file: the packaged `mocap_framerate`,
    unless the file belongs to a dataset in FRAMERATE_OVERRIDES."""
    path = str(smplh_file).replace(os.sep, "/")
    for ds, rate in FRAMERATE_OVERRIDES.items():
        if f"/{ds}/" in path:
            return rate
    return claimed_rate

def load_smpl_file(smpl_file):
    smpl_data = np.load(smpl_file, allow_pickle=True)
    return smpl_data


def _forward_frames(body_model, betas, root_orient, pose_body, trans, chunk=512):
    """Run the body model over a clip in fixed-size chunks under no_grad.

    The previous monolithic body_model(...) call materialized every intermediate
    for ALL frames at once and, lacking no_grad, the autograd graph on top --
    inference needs neither. On long clips this ballooned a batch worker to
    ~14 GB RSS and the kernel OOM-killed it inside the interactive session's
    cgroup, tearing the whole session down (2026-07-26). Chunking caps the peak
    at ~chunk frames regardless of clip length (~1-2 GB/worker). Returns only
    the fields downstream reads: global_orient, full_pose, joints."""
    import types
    n = int(root_orient.shape[0])
    b = torch.as_tensor(np.asarray(betas), dtype=torch.float32).view(1, -1)
    go = torch.tensor(root_orient).float()
    bp = torch.tensor(pose_body).float()
    tr = torch.tensor(trans).float()
    full_pose, joints = [], []
    with torch.no_grad():
        for i in range(0, n, chunk):
            k = min(i + chunk, n) - i
            out = body_model(
                betas=b, global_orient=go[i:i + k], body_pose=bp[i:i + k],
                transl=tr[i:i + k],
                left_hand_pose=torch.zeros(k, 45), right_hand_pose=torch.zeros(k, 45),
                jaw_pose=torch.zeros(k, 3), leye_pose=torch.zeros(k, 3),
                reye_pose=torch.zeros(k, 3), return_full_pose=True,
            )
            full_pose.append(out.full_pose)
            joints.append(out.joints)
    return types.SimpleNamespace(global_orient=go, full_pose=torch.cat(full_pose),
                                 joints=torch.cat(joints))


def measured_height(body_model, betas, anchor=1.66):
    """Subject height MEASURED from the body model: T-pose the clip's betas and
    take the joint vertical span, anchored so the zero-betas mean shape of the
    same model reads `anchor` (the height the old 1.66 + 0.1*betas[0] heuristic
    assigned to the mean). Shape- and gender-correct by construction; the
    heuristic ignored every beta but the first and the gender model, and drifted
    badly at extreme shapes (CMU/105: heuristic 1.55 vs measured skeleton ~1.5)."""
    with torch.no_grad():
        def span(b):
            out = body_model(betas=b, global_orient=torch.zeros(1, 3),
                             body_pose=torch.zeros(1, 63))
            j = out.joints[0, :22, :]                     # body joints only
            return float(j[:, 1].max() - j[:, 1].min())   # canonical T-pose is y-up
        b = torch.as_tensor(np.asarray(betas, dtype=np.float32).reshape(-1)
                            ).view(1, -1)
        return anchor * span(b) / span(torch.zeros_like(b))

def load_smplx_file(smplx_file, smplx_body_model_path):
    smplx_data = np.load(smplx_file, allow_pickle=True)
    body_model = smplx.create(
        smplx_body_model_path,
        "smplx",
        gender=str(smplx_data["gender"]),
        use_pca=False,
    )
    # print(smplx_data["pose_body"].shape)
    # print(smplx_data["betas"].shape)
    # print(smplx_data["root_orient"].shape)
    # print(smplx_data["trans"].shape)
    
    smplx_output = _forward_frames(body_model, smplx_data["betas"],
                                   smplx_data["root_orient"], smplx_data["pose_body"],
                                   smplx_data["trans"])
    
    human_height = measured_height(body_model, smplx_data["betas"])

    return smplx_data, body_model, smplx_output, human_height


def load_smplh_amass_file(smplh_file, smplx_body_model_path):
    """Load an AMASS SMPL+H npz and adapt it to the SMPL-X pipeline.

    AMASS SMPL+H stores axis-angle in `poses (N,156)` = [global_orient(3),
    body_pose(63), ...hands]. The body kinematic tree (joints 0-21) is shared
    with SMPL-X, so we feed the sliced body pose through the SMPL-X body model.
    """
    raw = np.load(smplh_file, allow_pickle=True)
    poses = raw["poses"]                 # (N, 156)
    num_frames = poses.shape[0]
    root_orient = poses[:, :3]           # (N, 3)
    pose_body = poses[:, 3:66]           # (N, 63) -> 21 body joints
    trans = raw["trans"]                 # (N, 3)
    betas = np.asarray(raw["betas"]).reshape(-1)[:16]
    if betas.shape[0] < 16:
        betas = np.pad(betas, (0, 16 - betas.shape[0]))
    gender = str(raw["gender"])
    gender = gender.replace("b'", "").replace("'", "").strip() or "neutral"
    frame_rate = true_frame_rate(smplh_file, float(raw["mocap_framerate"]))

    body_model = smplx.create(
        smplx_body_model_path, "smplx", gender=gender, use_pca=False, num_betas=16,
    )
    smplx_output = _forward_frames(body_model, betas, root_orient, pose_body, trans)

    # dict exposing exactly the keys the downstream code reads
    smplx_data = {
        "pose_body": pose_body,
        "root_orient": root_orient,
        "trans": trans,
        "betas": betas,
        "gender": gender,
        "mocap_frame_rate": np.array(frame_rate),
    }
    human_height = measured_height(body_model, betas)
    return smplx_data, body_model, smplx_output, human_height


def load_gvhmr_pred_file(gvhmr_pred_file, smplx_body_model_path):
    gvhmr_pred = torch.load(gvhmr_pred_file)
    smpl_params_global = gvhmr_pred['smpl_params_global']
    # print(smpl_params_global['body_pose'].shape)
    # print(smpl_params_global['betas'].shape)
    # print(smpl_params_global['global_orient'].shape)
    # print(smpl_params_global['transl'].shape)
    
    betas = np.pad(smpl_params_global['betas'][0], (0,6))
    
    # correct rotations
    # rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    # rotation_quat = R.from_matrix(rotation_matrix).as_quat(scalar_first=True)
    
    # smpl_params_global['body_pose'] = smpl_params_global['body_pose'] @ rotation_matrix
    # smpl_params_global['global_orient'] = smpl_params_global['global_orient'] @ rotation_quat
    
    smplx_data = {
        'pose_body': smpl_params_global['body_pose'].numpy(),
        'betas': betas,
        'root_orient': smpl_params_global['global_orient'].numpy(),
        'trans': smpl_params_global['transl'].numpy(),
        "mocap_frame_rate": torch.tensor(30),
    }

    body_model = smplx.create(
        smplx_body_model_path,
        "smplx",
        gender="neutral",
        use_pca=False,
    )
    
    smplx_output = _forward_frames(body_model, smplx_data["betas"],
                                   smplx_data["root_orient"], smplx_data["pose_body"],
                                   smplx_data["trans"])
    
    human_height = measured_height(body_model, smplx_data['betas'])

    return smplx_data, body_model, smplx_output, human_height


def get_smplx_data(smplx_data, body_model, smplx_output, curr_frame):
    """
    Must return a dictionary with the following structure:
    {
        "Hips": (position, orientation),
        "Spine": (position, orientation),
        ...
    }
    """
    global_orient = smplx_output.global_orient[curr_frame].squeeze()
    full_body_pose = smplx_output.full_pose[curr_frame].reshape(-1, 3)
    joints = smplx_output.joints[curr_frame].detach().numpy().squeeze()
    joint_names = JOINT_NAMES[: len(body_model.parents)]
    parents = body_model.parents

    result = {}
    joint_orientations = []
    for i, joint_name in enumerate(joint_names):
        if i == 0:
            rot = R.from_rotvec(global_orient)
        else:
            rot = joint_orientations[parents[i]] * R.from_rotvec(
                full_body_pose[i].squeeze()
            )
        joint_orientations.append(rot)
        result[joint_name] = (joints[i], rot.as_quat(scalar_first=True))

  
    return result


def slerp(rot1, rot2, t):
    """Spherical linear interpolation between two rotations."""
    # Convert to quaternions
    q1 = rot1.as_quat()
    q2 = rot2.as_quat()
    
    # Normalize quaternions
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    
    # Compute dot product
    dot = np.sum(q1 * q2)
    
    # If the dot product is negative, slerp won't take the shorter path
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    
    # If the inputs are too close, linearly interpolate
    if dot > 0.9995:
        return R.from_quat(q1 + t * (q2 - q1))
    
    # Perform SLERP
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)
    
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    q = s0 * q1 + s1 * q2
    
    return R.from_quat(q)

def get_smplx_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=30):
    """
    Must return a dictionary with the following structure:
    {
        "Hips": (position, orientation),
        "Spine": (position, orientation),
        ...
    }
    """
    src_fps = smplx_data["mocap_frame_rate"].item()
    frame_skip = int(src_fps / tgt_fps)
    num_frames = smplx_data["pose_body"].shape[0]
    global_orient = smplx_output.global_orient.squeeze()
    full_body_pose = smplx_output.full_pose.reshape(num_frames, -1, 3)
    joints = smplx_output.joints.detach().numpy().squeeze()
    joint_names = JOINT_NAMES[: len(body_model.parents)]
    parents = body_model.parents
    
    if tgt_fps < src_fps:
        # perform fps alignment with proper interpolation
        new_num_frames = num_frames // frame_skip
        
        # Create time points for interpolation
        original_time = np.arange(num_frames)
        target_time = np.linspace(0, num_frames-1, new_num_frames)
        
        # Interpolate global orientation using SLERP
        global_orient_interp = []
        for i in range(len(target_time)):
            t = target_time[i]
            idx1 = int(np.floor(t))
            idx2 = min(idx1 + 1, num_frames - 1)
            alpha = t - idx1
            
            rot1 = R.from_rotvec(global_orient[idx1])
            rot2 = R.from_rotvec(global_orient[idx2])
            interp_rot = slerp(rot1, rot2, alpha)
            global_orient_interp.append(interp_rot.as_rotvec())
        global_orient = np.stack(global_orient_interp, axis=0)
        
        # Interpolate full body pose using SLERP
        full_body_pose_interp = []
        for i in range(full_body_pose.shape[1]):  # For each joint
            joint_rots = []
            for j in range(len(target_time)):
                t = target_time[j]
                idx1 = int(np.floor(t))
                idx2 = min(idx1 + 1, num_frames - 1)
                alpha = t - idx1
                
                rot1 = R.from_rotvec(full_body_pose[idx1, i])
                rot2 = R.from_rotvec(full_body_pose[idx2, i])
                interp_rot = slerp(rot1, rot2, alpha)
                joint_rots.append(interp_rot.as_rotvec())
            full_body_pose_interp.append(np.stack(joint_rots, axis=0))
        full_body_pose = np.stack(full_body_pose_interp, axis=1)
        
        # Interpolate joint positions using linear interpolation
        joints_interp = []
        for i in range(joints.shape[1]):  # For each joint
            for j in range(3):  # For each coordinate
                interp_func = interp1d(original_time, joints[:, i, j], kind='linear')
                joints_interp.append(interp_func(target_time))
        joints = np.stack(joints_interp, axis=1).reshape(new_num_frames, -1, 3)
        
        aligned_fps = len(global_orient) / num_frames * src_fps
    else:
        aligned_fps = tgt_fps
        
    smplx_data_frames = []
    for curr_frame in range(len(global_orient)):
        result = {}
        single_global_orient = global_orient[curr_frame]
        single_full_body_pose = full_body_pose[curr_frame]
        single_joints = joints[curr_frame]
        joint_orientations = []
        for i, joint_name in enumerate(joint_names):
            if i == 0:
                rot = R.from_rotvec(single_global_orient)
            else:
                rot = joint_orientations[parents[i]] * R.from_rotvec(
                    single_full_body_pose[i].squeeze()
                )
            joint_orientations.append(rot)
            result[joint_name] = (single_joints[i], rot.as_quat(scalar_first=True))


        smplx_data_frames.append(result)

    return smplx_data_frames, aligned_fps



def get_gvhmr_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=30):
    """
    Must return a dictionary with the following structure:
    {
        "Hips": (position, orientation),
        "Spine": (position, orientation),
        ...
    }
    """
    src_fps = smplx_data["mocap_frame_rate"].item()
    frame_skip = int(src_fps / tgt_fps)
    num_frames = smplx_data["pose_body"].shape[0]
    global_orient = smplx_output.global_orient.squeeze()
    full_body_pose = smplx_output.full_pose.reshape(num_frames, -1, 3)
    joints = smplx_output.joints.detach().numpy().squeeze()
    joint_names = JOINT_NAMES[: len(body_model.parents)]
    parents = body_model.parents
    
    if tgt_fps < src_fps:
        # perform fps alignment with proper interpolation
        new_num_frames = num_frames // frame_skip
        
        # Create time points for interpolation
        original_time = np.arange(num_frames)
        target_time = np.linspace(0, num_frames-1, new_num_frames)
        
        # Interpolate global orientation using SLERP
        global_orient_interp = []
        for i in range(len(target_time)):
            t = target_time[i]
            idx1 = int(np.floor(t))
            idx2 = min(idx1 + 1, num_frames - 1)
            alpha = t - idx1
            
            rot1 = R.from_rotvec(global_orient[idx1])
            rot2 = R.from_rotvec(global_orient[idx2])
            interp_rot = slerp(rot1, rot2, alpha)
            global_orient_interp.append(interp_rot.as_rotvec())
        global_orient = np.stack(global_orient_interp, axis=0)
        
        # Interpolate full body pose using SLERP
        full_body_pose_interp = []
        for i in range(full_body_pose.shape[1]):  # For each joint
            joint_rots = []
            for j in range(len(target_time)):
                t = target_time[j]
                idx1 = int(np.floor(t))
                idx2 = min(idx1 + 1, num_frames - 1)
                alpha = t - idx1
                
                rot1 = R.from_rotvec(full_body_pose[idx1, i])
                rot2 = R.from_rotvec(full_body_pose[idx2, i])
                interp_rot = slerp(rot1, rot2, alpha)
                joint_rots.append(interp_rot.as_rotvec())
            full_body_pose_interp.append(np.stack(joint_rots, axis=0))
        full_body_pose = np.stack(full_body_pose_interp, axis=1)
        
        # Interpolate joint positions using linear interpolation
        joints_interp = []
        for i in range(joints.shape[1]):  # For each joint
            for j in range(3):  # For each coordinate
                interp_func = interp1d(original_time, joints[:, i, j], kind='linear')
                joints_interp.append(interp_func(target_time))
        joints = np.stack(joints_interp, axis=1).reshape(new_num_frames, -1, 3)
        
        aligned_fps = len(global_orient) / num_frames * src_fps
    else:
        aligned_fps = tgt_fps
        
    smplx_data_frames = []
    for curr_frame in range(len(global_orient)):
        result = {}
        single_global_orient = global_orient[curr_frame]
        single_full_body_pose = full_body_pose[curr_frame]
        single_joints = joints[curr_frame]
        joint_orientations = []
        for i, joint_name in enumerate(joint_names):
            if i == 0:
                rot = R.from_rotvec(single_global_orient)
            else:
                rot = joint_orientations[parents[i]] * R.from_rotvec(
                    single_full_body_pose[i].squeeze()
                )
            joint_orientations.append(rot)
            result[joint_name] = (single_joints[i], rot.as_quat(scalar_first=True))


        smplx_data_frames.append(result)
        
    # add correct rotations
    rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    rotation_quat = R.from_matrix(rotation_matrix).as_quat(scalar_first=True)
    for result in smplx_data_frames:
        for joint_name in result.keys():
            orientation = quat_mul(rotation_quat, result[joint_name][1])
            position = result[joint_name][0] @ rotation_matrix.T
            result[joint_name] = (position, orientation)
            

    return smplx_data_frames, aligned_fps
