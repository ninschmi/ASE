"""
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.

Actor Scaling
------------
- Loads a handful of MJCF and URDF assets and scales them using the runtime scaling API
"""

import math
import numpy as np
from isaacgym import gymapi, gymutil
import yaml
import os
import xml.etree.ElementTree as et

class AssetDesc:
    def __init__(self, file_name, flip_visual_attachments=False):
        self.file_name = file_name
        self.flip_visual_attachments = flip_visual_attachments


asset_descriptors = [
    #AssetDesc("amp_humanoid_sword_shield.xml", False),
    AssetDesc("arm_leg_parametrization/amp_humanoid_sword_shield_arm_0-5_leg_0-5.xml", False),
    AssetDesc("arm_leg_parametrization/amp_humanoid_sword_shield_arm_0-5_leg_0-75.xml", False),
    AssetDesc("arm_leg_parametrization/amp_humanoid_sword_shield_arm_0-5_leg_1-0.xml", False),
    AssetDesc("arm_leg_parametrization/amp_humanoid_sword_shield_arm_0-5_leg_1-25.xml", False),
    AssetDesc("arm_leg_parametrization/amp_humanoid_sword_shield_arm_0-5_leg_1-5.xml", False),
    AssetDesc("arm_leg_parametrization/amp_humanoid_sword_shield_arm_0-5_leg_1-75.xml", False),
    AssetDesc("arm_leg_parametrization/amp_humanoid_sword_shield_arm_0-5_leg_2-0.xml", False),
    AssetDesc("arm_leg_parametrization/amp_humanoid_sword_shield_arm_0-75_leg_0-5.xml", False),
    AssetDesc("arm_leg_parametrization/amp_humanoid_sword_shield_arm_0-75_leg_0-75.xml", False),
    AssetDesc("arm_leg_parametrization/amp_humanoid_sword_shield_arm_0-75_leg_1-0.xml", False),
    AssetDesc("arm_leg_parametrization/amp_humanoid_sword_shield_arm_0-75_leg_1-25.xml", False),
    AssetDesc("arm_leg_parametrization/amp_humanoid_sword_shield_arm_0-75_leg_1-5.xml", False),
    AssetDesc("arm_leg_parametrization/amp_humanoid_sword_shield_arm_0-75_leg_1-75.xml", False),
    AssetDesc("arm_leg_parametrization/amp_humanoid_sword_shield_arm_0-75_leg_2-0.xml", False),
    AssetDesc("arm_leg_parametrization/amp_humanoid_sword_shield_arm_1-0_leg_0-5.xml", False),
    AssetDesc("arm_leg_parametrization/amp_humanoid_sword_shield_arm_1-0_leg_0-75.xml", False),
    AssetDesc("arm_leg_parametrization/amp_humanoid_sword_shield_arm_1-0_leg_1-0.xml", False),
    AssetDesc("arm_leg_parametrization/amp_humanoid_sword_shield_arm_1-0_leg_1-25.xml", False),
    AssetDesc("arm_leg_parametrization/amp_humanoid_sword_shield_arm_1-0_leg_1-5.xml", False),
    AssetDesc("arm_leg_parametrization/amp_humanoid_sword_shield_arm_1-0_leg_1-75.xml", False),
    AssetDesc("arm_leg_parametrization/amp_humanoid_sword_shield_arm_1-0_leg_2-0.xml", False),
    AssetDesc("arm_leg_parametrization/amp_humanoid_sword_shield_arm_1-25_leg_0-5.xml", False),
    AssetDesc("arm_leg_parametrization/amp_humanoid_sword_shield_arm_1-25_leg_0-75.xml", False),
    AssetDesc("arm_leg_parametrization/amp_humanoid_sword_shield_arm_1-25_leg_1-0.xml", False),
    AssetDesc("arm_leg_parametrization/amp_humanoid_sword_shield_arm_1-25_leg_1-25.xml", False),
    AssetDesc("arm_leg_parametrization/amp_humanoid_sword_shield_arm_1-25_leg_1-5.xml", False),
    AssetDesc("arm_leg_parametrization/amp_humanoid_sword_shield_arm_1-25_leg_1-75.xml", False),
    AssetDesc("arm_leg_parametrization/amp_humanoid_sword_shield_arm_1-25_leg_2-0.xml", False),
    AssetDesc("arm_leg_parametrization/amp_humanoid_sword_shield_arm_1-5_leg_0-5.xml", False),
    AssetDesc("arm_leg_parametrization/amp_humanoid_sword_shield_arm_1-5_leg_0-75.xml", False),
    AssetDesc("arm_leg_parametrization/amp_humanoid_sword_shield_arm_1-5_leg_1-0.xml", False),
    AssetDesc("arm_leg_parametrization/amp_humanoid_sword_shield_arm_1-5_leg_1-25.xml", False),
    AssetDesc("arm_leg_parametrization/amp_humanoid_sword_shield_arm_1-5_leg_1-5.xml", False),
    AssetDesc("arm_leg_parametrization/amp_humanoid_sword_shield_arm_1-5_leg_1-75.xml", False),
    AssetDesc("arm_leg_parametrization/amp_humanoid_sword_shield_arm_1-5_leg_2-0.xml", False),
]

with open('data/cfg/humanoid_ase_sword_shield_getup.yaml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

args = gymutil.parse_arguments(description="Show Actor")

# initialize gym
gym = gymapi.acquire_gym()

# configure sim
sim_params = gymapi.SimParams()
sim_params.dt = dt = 1.0 / 60.0
#sim_params.num_client_threads = args.slices
if args.physics_engine == gymapi.SIM_FLEX:
    pass
elif args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.num_threads = 4
    sim_params.physx.use_gpu = args.use_gpu
    #sim_params.physx.num_subscenes = args.subscenes
    #sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024
    sim_params.gravity = gymapi.Vec3(0.0,0.0,-9.8)

sim_params.use_gpu_pipeline = False
#sim_params.use_gpu_pipeline = args.use_gpu_pipeline
#sim_params.physx.use_gpu = args.use_gpu

if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

# if sim options are provided in cfg, parse them and update/override above:
if "sim" in cfg:
    gymutil.parse_sim_config(cfg["sim"], sim_params)

# Override num_threads if passed on the command line
if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
    sim_params.physx.num_threads = args.num_threads

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
gym.add_ground(sim, plane_params)

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# load asset
#asset_root = "data/assets/mjcf/arm_leg_parametrization/"
#asset_file = "amp_humanoid_sword_shield_arm_0-5_leg_2-0.xml"

asset_root = "data/assets/mjcf/"
#asset_file = "amp_humanoid_sword_shield.xml"

assets = []
for asset_desc in asset_descriptors:
    # load asset
    #asset_root = "data/assets/mjcf/"
    asset_file = asset_desc.file_name

    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.flip_visual_attachments = asset_desc.flip_visual_attachments
    asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
    asset_options.use_mesh_materials = True

    print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
    assets.append(gym.load_asset(sim, asset_root, asset_file, asset_options))

# set up the env grid
spacing = 1
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)
#lower = gymapi.Vec3(-spacing, -spacing, 0.0)
#upper = gymapi.Vec3(spacing, spacing, spacing)

# position the camera
cam_pos = gymapi.Vec3(17.2, 2.0, 10)
cam_target = gymapi.Vec3(5, -2.5, 13)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# cache useful handles
envs = []
actor_handles = []

num_envs = 35

num_per_row = 7


for i, asset in enumerate(assets):

    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # determine position
    pose_string = et.parse(asset_root+asset_descriptors[i].file_name).getroot().find('.//body[@name="pelvis"]').get('pos')
    pose_list = pose_string.split(" ")
    pose_list = [float(x) for x in pose_list]
    
    # add actor
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(pose_list[0], pose_list[1], (pose_list[2]/1*0.89))
    pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

    actor_handle = gym.create_actor(env, asset, pose, "humanoid", i, 1)
    actor_handles.append(actor_handle)

    gym.set_actor_scale(env, actor_handle, 1)

while not gym.query_viewer_has_closed(viewer):
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

print("Done")

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
