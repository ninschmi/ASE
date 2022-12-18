# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from enum import Enum
import numpy as np
import torch
import os

from isaacgym import gymapi
from isaacgym import gymtorch

from env.tasks.humanoid import Humanoid, dof_to_obs
from utils import gym_util
from utils.motion_lib import MotionLib
from isaacgym.torch_utils import *

from utils import torch_utils

class HumanoidAMP(Humanoid):
    class StateInit(Enum):
        Default = 0
        Start = 1
        Random = 2
        Hybrid = 3

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        state_init = cfg["env"]["stateInit"]
        self._state_init = HumanoidAMP.StateInit[state_init]
        self._hybrid_init_prob = cfg["env"]["hybridInitProb"]
        self._num_amp_obs_steps = cfg["env"]["numAMPObsSteps"]
        assert(self._num_amp_obs_steps >= 2)

        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)

        motion_file = cfg['env']['motion_file']
        self._load_motion(motion_file)

        self._amp_obs_buf = torch.zeros((self.num_envs, self._num_amp_obs_steps, self._num_amp_obs_per_step), device=self.device, dtype=torch.float)
        self._curr_amp_obs_buf = self._amp_obs_buf[:, 0]
        self._hist_amp_obs_buf = self._amp_obs_buf[:, 1:]
        
        self._amp_obs_demo_buf = None

        return

    def post_physics_step(self):
        super().post_physics_step()
        
        self._update_hist_amp_obs()
        self._compute_amp_observations()

        amp_obs_flat = self._amp_obs_buf.view(-1, self.get_num_amp_obs())
        self.extras["amp_obs"] = amp_obs_flat

        return

    def get_num_amp_obs(self):
        return self._num_amp_obs_steps * self._num_amp_obs_per_step

    def fetch_amp_obs_demo(self, num_samples):

        if (self._amp_obs_demo_buf is None):
            self._build_amp_obs_demo_buf(num_samples)
        else:
            assert(self._amp_obs_demo_buf.shape[0] == num_samples)
        
        motion_ids = self._motion_lib.sample_motions(num_samples)
        motion_times0 = self._motion_lib.sample_time(motion_ids)
        amp_obs_demo = self.build_amp_obs_demo(motion_ids, motion_times0)
        self._amp_obs_demo_buf[:] = amp_obs_demo.view(self._amp_obs_demo_buf.shape)
        amp_obs_demo_flat = self._amp_obs_demo_buf.view(-1, self.get_num_amp_obs())

        return amp_obs_demo_flat

    def build_amp_obs_demo(self, motion_ids, motion_times0):
        dt = self.dt

        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, self._num_amp_obs_steps])
        motion_times = motion_times0.unsqueeze(-1)
        time_steps = -dt * torch.arange(0, self._num_amp_obs_steps, device=self.device)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos, local_rot \
               = self._motion_lib.get_motion_state(motion_ids, motion_times)
        amp_obs_demo = build_amp_observations(root_pos, root_rot, root_vel, root_ang_vel,
                                              dof_pos, dof_vel, key_pos,
                                              self._local_root_obs, self._root_height_obs,
                                              self._dof_obs_size, self._dof_offsets)
        return amp_obs_demo

    def _build_amp_obs_demo_buf(self, num_samples):
        self._amp_obs_demo_buf = torch.zeros((num_samples, self._num_amp_obs_steps, self._num_amp_obs_per_step), device=self.device, dtype=torch.float32)
        return
        
    def _setup_character_props(self, key_bodies):
        super()._setup_character_props(key_bodies)

        asset_file = self.cfg["env"]["asset"]["assetFileName"]
        asset_path = "ase/data/assets/" + asset_file
        num_key_bodies = len(key_bodies)

        if (asset_file == "mjcf/amp_humanoid.xml"):
            self._num_amp_obs_per_step = 13 + self._dof_obs_size + 28 + 3 * num_key_bodies # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
        elif (asset_file == "mjcf/amp_humanoid_sword_shield.xml"):
            self._num_amp_obs_per_step = 13 + self._dof_obs_size + 31 + 3 * num_key_bodies # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
        elif (os.path.isfile(asset_path)):
            self._num_amp_obs_per_step = 13 + self._dof_obs_size + 31 + 3 * num_key_bodies # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
        else:
            print("Unsupported character config file: {s}".format(asset_file))
            assert(False)

        return

    def _load_motion(self, motion_file):
        assert(self._dof_offsets[-1] == self.num_dof)
        self._motion_lib = MotionLib(motion_file=motion_file,
                                     dof_body_ids=self._dof_body_ids,
                                     dof_offsets=self._dof_offsets,
                                     key_body_ids=self._key_body_ids.cpu().numpy(), 
                                     device=self.device)
        return
    
    def _reset_envs(self, env_ids):
        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []

        super()._reset_envs(env_ids)
        self._init_amp_obs(env_ids)

        return

    def _reset_actors(self, env_ids):
        if (self._state_init == HumanoidAMP.StateInit.Default):
            self._reset_default(env_ids)
        elif (self._state_init == HumanoidAMP.StateInit.Start
              or self._state_init == HumanoidAMP.StateInit.Random):
            self._reset_ref_state_init(env_ids)
        elif (self._state_init == HumanoidAMP.StateInit.Hybrid):
            self._reset_hybrid_state_init(env_ids)
        else:
            assert(False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))
        return
    
    def _reset_default(self, env_ids):
        self._humanoid_root_states[env_ids] = self._initial_humanoid_root_states[env_ids]
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]
        self._reset_default_env_ids = env_ids
        return

    def _reset_ref_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        motion_ids = self._motion_lib.sample_motions(num_envs)
        
        if (self._state_init == HumanoidAMP.StateInit.Random
            or self._state_init == HumanoidAMP.StateInit.Hybrid):
            motion_times = self._motion_lib.sample_time(motion_ids)
        elif (self._state_init == HumanoidAMP.StateInit.Start):
            motion_times = torch.zeros(num_envs, device=self.device)
        else:
            assert(False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))

        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos, local_rot \
               = self._motion_lib.get_motion_state(motion_ids, motion_times)

        if self.randomize:

            # FORWARD KINEMATICS APPROACH

            # check that there is no penetration
            # compute forward kinematics to get rigid body pos in global coordinates
            # find global coordinates of rigid body positionst

            # make sure there is an environment with a base character for reference
            #if not self.base_char_idx in to_torch(self.env_char_mapping, device=self.device, dtype=torch.long)[env_ids]:
            #    x = self.envs_per_file * self.base_char_idx
            #    env_ids = torch.cat((env_ids, to_torch([self.envs_per_file * self.base_char_idx], dtype=torch.int)), dim=0)

            # retrieve local translation from mujoco file for forward kinematics
            local_translations = to_torch(self.local_rigid_body_pos,device=self.device, dtype=torch.float32)[to_torch(self.env_char_mapping, device=self.device, dtype=torch.long)[env_ids]]
                
            num_bodies = local_rot.shape[1]

            # compute transforms for each rigid body
            transform_q = torch.empty((num_envs, num_bodies, 4), device=self.device)   # num envs x num rigid bodies x 4 (quaternions)
            transform_t = torch.empty((num_envs, num_bodies, 3), device=self.device)   # num envs x num rigid bodies x 3 (translation)

            # pelvis aka root
            transform_q[:,0,:] = root_rot
            transform_t[:,0,:] = root_pos

            # torso (torso -> pelvis)
            transform_q[:,1,:], transform_t[:,1,:] = tf_combine(transform_q[:,0,:], transform_t[:,0,:], local_rot[:,1,:], local_translations[:,1,:])
            # head (head -> torso -> pelvis)
            transform_q[:,2,:], transform_t[:,2,:] = tf_combine(transform_q[:,1,:], transform_t[:,1,:], local_rot[:,2,:], local_translations[:,2,:])
            # right upper arm (r_u_arm -> torso -> pelvis)
            transform_q[:,3,:], transform_t[:,3,:] = tf_combine(transform_q[:,1,:], transform_t[:,1,:], local_rot[:,3,:], local_translations[:,3,:])
            # right lower arm  (r_l_arm -> r_u_arm -> torso -> pelvis)
            transform_q[:,4,:], transform_t[:,4,:] = tf_combine(transform_q[:,3,:], transform_t[:,3,:], local_rot[:,4,:], local_translations[:,4,:])
            # right hand (r_hand -> r_l_arm -> r_u_arm -> torso -> pelvis)
            transform_q[:,5,:], transform_t[:,5,:] = tf_combine(transform_q[:,4,:], transform_t[:,4,:], local_rot[:,5,:], local_translations[:,5,:])
            # sword (sword -> r_hand -> r_l_arm -> r_u_arm -> torso -> pelvis)
            transform_q[:,6,:], transform_t[:,6,:] = tf_combine(transform_q[:,5,:], transform_t[:,5,:], local_rot[:,6,:], local_translations[:,6,:])
            # left upper arm (l_u_arm -> torso -> pelvis)
            transform_q[:,7,:], transform_t[:,7,:] = tf_combine(transform_q[:,1,:], transform_t[:,1,:], local_rot[:,7,:], local_translations[:,7,:])
            # left lower arm (l_l_arm -> l_u_arm -> torso -> pelvis)
            transform_q[:,8,:], transform_t[:,8,:] = tf_combine(transform_q[:,7,:], transform_t[:,7,:], local_rot[:,8,:], local_translations[:,8,:])
            # shield (shield -> l_l_arm -> l_u_arm -> torso -> pelvis)
            transform_q[:,9,:], transform_t[:,9,:] = tf_combine(transform_q[:,8,:], transform_t[:,8,:], local_rot[:,9,:], local_translations[:,9,:])
            # left hand (l_hand -> l_l_arm -> l_u_arm -> torso -> pelvis)
            transform_q[:,10,:], transform_t[:,10,:] = tf_combine(transform_q[:,8,:], transform_t[:,8,:], local_rot[:,10,:], local_translations[:,10,:])
            # right thigh (r_thigh -> pelvis)
            transform_q[:,11,:], transform_t[:,11,:] = tf_combine(transform_q[:,0,:], transform_t[:,0,:], local_rot[:,11,:], local_translations[:,11,:])
            # right shin (r_shin -> r_thigh -> pelvis)
            transform_q[:,12,:], transform_t[:,12,:] = tf_combine(transform_q[:,11,:], transform_t[:,11,:], local_rot[:,12,:], local_translations[:,12,:])
            # right foot (r_foot -> r_shin -> r_thigh -> pelvis)
            transform_q[:,13,:], transform_t[:,13,:] = tf_combine(transform_q[:,12,:], transform_t[:,12,:], local_rot[:,13,:], local_translations[:,13,:])
            # left thigh (l_thigh -> pelvis)
            transform_q[:,14,:], transform_t[:,14,:] = tf_combine(transform_q[:,0,:], transform_t[:,0,:], local_rot[:,14,:], local_translations[:,14,:])
            # left shin (l_shin -> l_thigh -> pelvis)
            transform_q[:,15,:], transform_t[:,15,:] = tf_combine(transform_q[:,14,:], transform_t[:,14,:], local_rot[:,15,:], local_translations[:,15,:])
            # left foot (l_foot -> l_shin -> l_thigh -> pelvis)
            transform_q[:,16,:], transform_t[:,16,:] = tf_combine(transform_q[:,15,:], transform_t[:,15,:], local_rot[:,16,:], local_translations[:,16,:])

            # return global coordinates for each rigid body (origin of body frame)
            global_pos = torch.empty((num_envs, num_bodies, 3), device=self.device)   # num envs x num rigid bodies x 3 (position)

            global_pos = transform_t
            #global_pos[:,body,:] = tf_apply(transform_q[:,body,:], transform_t[:,body,:], torch.zeros((3), device=self.device))

            # compute not only global coordinates of body frame origin but some further points to avoid penetrations
            # compute global coordinates of corners of left and right foot
            # base foot size="0.0885 0.045 0.0275" 
            foot_corners = torch.tensor([[0.0885/2, 0.045/2, 0.0275/2], [0.0885/2, 0.045/2, -0.0275/2], [0.0885/2, -0.045/2, 0.0275/2], [0.0885/2, -0.045/2, -0.0275/2], [-0.0885/2, 0.045/2, 0.0275/2], [-0.0885/2, 0.045/2, -0.0275/2], [-0.0885/2, -0.045/2, 0.0275/2], [-0.0885/2, -0.045/2, -0.0275/2]], device=self.device, requires_grad=False)
            right_foot_q, right_foot_t = tf_combine(transform_q[:,13,:], transform_t[:,13,:], to_torch([0,0,0,1], device=self.device).expand(num_envs,4), to_torch(self.local_right_foot_pos,device=self.device, dtype=torch.float32)[to_torch(self.env_char_mapping, device=self.device, dtype=torch.long)[env_ids]])
            left_foot_q, left_foot_t = tf_combine(transform_q[:,16,:], transform_t[:,16,:], to_torch([0,0,0,1], device=self.device).expand(num_envs,4), to_torch(self.local_left_foot_pos,device=self.device, dtype=torch.float32)[to_torch(self.env_char_mapping, device=self.device, dtype=torch.long)[env_ids]])
            for corner in foot_corners:
                # tf_apply(q, t, v): quat_apply(q, v) + t
                point_right = tf_apply(right_foot_q, right_foot_t, corner.expand(num_envs,3))
                point_left = tf_apply(left_foot_q, left_foot_t, corner.expand(num_envs,3))
                global_pos = torch.cat((global_pos, point_right.unsqueeze(1)), dim=1)
                global_pos = torch.cat((global_pos, point_left.unsqueeze(1)), dim=1)

            # compute distance to ground for base character as reference

            if torch.sum(torch.gt(torch.min(global_pos[:,:,2], dim=1).values,torch.ones_like(global_pos[:,0,0])*0.1))>0:
                print("here")

            # find minimal z coordinate and increase root pos by that amount if negative
            if torch.sum(torch.gt(torch.zeros_like(global_pos[:,0,0]),torch.min(global_pos[:,:,2], dim=1).values))>0:
                    root_pos[:,2] -= torch.clamp_max(torch.min(global_pos[:,:,2], dim=1).values, max=0)

            # SIMULATION APPROACH

            #self._humanoid_root_states[env_ids, 0:3] = root_pos
            #self._humanoid_root_states[env_ids, 3:7] = root_rot
            #self._humanoid_root_states[env_ids, 7:10] = root_vel
            #self._humanoid_root_states[env_ids, 10:13] = root_ang_vel
            #
            #self._dof_pos[env_ids] = dof_pos
            #self._dof_vel[env_ids] = dof_vel
#   
            #env_ids_int32 = self._humanoid_actor_ids[env_ids]
            #self.gym.set_actor_root_state_tensor_indexed(self.sim,
            #                                             gymtorch.unwrap_tensor(self._root_states),
            #                                             gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
            #self.gym.set_dof_state_tensor_indexed(self.sim,
            #                                      gymtorch.unwrap_tensor(self._dof_state),
            #                                      gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
#   
            #self.render()
            #self.gym.simulate(self.sim)
            #self.render()
            #
#   
            #test_00 = torch.zeros_like(self._rigid_body_pos[env_ids])
            #test_000 = torch.zeros_like(self._rigid_body_pos[env_ids,0,0])
            #test_01 = torch.min(self._rigid_body_pos[env_ids,:,2], dim=1).values
            #test_1 = torch.gt(torch.zeros_like(self._rigid_body_pos[env_ids,0,0]),torch.min(self._rigid_body_pos[env_ids,:,2], dim=1).values)
            #test_2 = torch.sum(torch.gt(torch.zeros_like(self._rigid_body_pos[env_ids,0,0]),torch.min(self._rigid_body_pos[env_ids,:,2], dim=1).values))
            #if torch.sum(torch.gt(torch.zeros_like(self._rigid_body_pos[env_ids,0,0]),torch.min(self._rigid_body_pos[env_ids,:,2], dim=1).values))>0:
            #        test_4 = torch.clamp_max(torch.min(self._rigid_body_pos[env_ids,:,2], dim=1).values, max=0)
            #        root_pos[:,2] -= torch.clamp_max(torch.min(self._rigid_body_pos[env_ids,:,2], dim=1).values, max=0)
#   
            #        testtt = to_torch(self._scale_leg, device=self.device)[to_torch(self.env_char_mapping, device=self.device, dtype=torch.long)[env_ids]]
            #        corr = 0.0275 * to_torch(self._scale_leg, device=self.device)[to_torch(self.env_char_mapping, device=self.device, dtype=torch.long)[env_ids]]
            #        root_pos[:,2] = torch.where(torch.min(self._rigid_body_pos[env_ids,:,2], dim=1).indices == 13, root_pos[:,2] + 2*0.0885 * to_torch(self._scale_leg, device=self.device)[to_torch(self.env_char_mapping, device=self.device, dtype=torch.long)[env_ids]], root_pos[:,2])
            #        root_pos[:,2] = torch.where(torch.min(self._rigid_body_pos[env_ids,:,2], dim=1).indices == 16, root_pos[:,2] + 2*0.0885 * to_torch(self._scale_leg, device=self.device)[to_torch(self.env_char_mapping, device=self.device, dtype=torch.long)[env_ids]], root_pos[:,2])
            #        
            #        #elif [5,10] in torch.min(self._rigid_body_pos[env_ids,:,2], dim=1).indices:
#   
            #self._humanoid_root_states[env_ids, 0:3] = root_pos
            #self._humanoid_root_states[env_ids, 3:7] = root_rot
            #self._humanoid_root_states[env_ids, 7:10] = root_vel
            #self._humanoid_root_states[env_ids, 10:13] = root_ang_vel
            #
            #self._dof_pos[env_ids] = dof_pos
            #self._dof_vel[env_ids] = dof_vel
#   
            #env_ids_int32 = self._humanoid_actor_ids[env_ids]
            #self.gym.set_actor_root_state_tensor_indexed(self.sim,
            #                                             gymtorch.unwrap_tensor(self._root_states),
            #                                             gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
            #self.gym.set_dof_state_tensor_indexed(self.sim,
            #                                      gymtorch.unwrap_tensor(self._dof_state),
            #                                      gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
#   
            #self.render()
            #self.gym.simulate(self.sim)
            #self.render()

            # RANDOM APPROACHES

            #dof_frame = self.gym.get_actor_dof_frames(self.envs[13], self.humanoid_handles[13])

            #poses = self.gym.get_actor_rigid_body_states(self.envs[13], self.humanoid_handles[13], gymapi.STATE_POS)['pose']
            # Get pose for all of the handles
            #pose = gymapi.Transform.from_buffer(poses[0])

            #gymapi.Transform.from_buffer(self._rigid_body_pos[13,0])

            # check that there is no penetration
            # compute forward kinematics to get rigid body pos
            #self.joint_transforms #35 x np.array(16x ((3) pos, (4) rot)
            #root_pos


            #for env_id in env_ids:
                    #env = self.envs[env_id]
                    #handle = self.gym.find_actor_handle(env, actor)
                    #actor_handle = self.humanoid_handles[env_id]
                    #transforms = self.gym.get_actor_joint_transforms(env, actor_handle)


            #top_drawer_grasp = gymapi.Transform(top_drawer_point, top_drawer_handle_pose.r)
            # min z value should be greater than 0
            # ow increase root pos

            #self.joint_transforms #35 x np.array(16x ((3) pos, (4) rot)

            #for i in self._rigid_body_pos.shape[1]-1:
            #    local_rot = self.gym.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), dof_pos[0])

            # generate transforms
            #transforms = []
            #transforms.append(gymapi.Transform(root_pos,root_rot))
            #for i in self._rigid_body_pos.shape[1]-1:
                # retrieve initial global transform for rigid body i + 1
            #    transform = self.joint_transforms_2[i+1]
            #    axis = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 0.5 * math.pi)

            #    transform = gymapi.Transform(self.joint_transforms_2[i].p,self.joint_transforms_2[i].r)
            #    old_glob_trans = self.joint_transforms_2[i+1].p
            #    loc_trans = transform.inverse().transform_point(old_glob_trans)
            #    new_glob_trans = transforms[i].transform_point(loc_trans)

            #self.joint_transforms_2[0].r = root_rot

            #transform = gymapi.Transform(self.joint_transforms_2[0].p,self.joint_transforms_2[0].r)

        self._set_env_state(env_ids=env_ids, 
                            root_pos=root_pos, 
                            root_rot=root_rot, 
                            dof_pos=dof_pos, 
                            root_vel=root_vel, 
                            root_ang_vel=root_ang_vel, 
                            dof_vel=dof_vel)

        self._reset_ref_env_ids = env_ids
        self._reset_ref_motion_ids = motion_ids
        self._reset_ref_motion_times = motion_times
        return

    def _reset_hybrid_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        ref_probs = to_torch(np.array([self._hybrid_init_prob] * num_envs), device=self.device)
        ref_init_mask = torch.bernoulli(ref_probs) == 1.0

        ref_reset_ids = env_ids[ref_init_mask]
        if (len(ref_reset_ids) > 0):
            self._reset_ref_state_init(ref_reset_ids)

        default_reset_ids = env_ids[torch.logical_not(ref_init_mask)]
        if (len(default_reset_ids) > 0):
            self._reset_default(default_reset_ids)

        return

    def _init_amp_obs(self, env_ids):
        self._compute_amp_observations(env_ids)

        if (len(self._reset_default_env_ids) > 0):
            self._init_amp_obs_default(self._reset_default_env_ids)

        if (len(self._reset_ref_env_ids) > 0):
            self._init_amp_obs_ref(self._reset_ref_env_ids, self._reset_ref_motion_ids,
                                   self._reset_ref_motion_times)
        
        return

    def _init_amp_obs_default(self, env_ids):
        curr_amp_obs = self._curr_amp_obs_buf[env_ids].unsqueeze(-2)
        self._hist_amp_obs_buf[env_ids] = curr_amp_obs
        return

    def _init_amp_obs_ref(self, env_ids, motion_ids, motion_times):
        dt = self.dt
        motion_ids = torch.tile(motion_ids, [1, self._num_amp_obs_steps - 1])
        motion_times = motion_times.unsqueeze(-1)
        time_steps = -dt * (torch.arange(0, self._num_amp_obs_steps - 1, device=self.device) + 1)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos, local_rot \
               = self._motion_lib.get_motion_state(motion_ids, motion_times)
        amp_obs_demo = build_amp_observations(root_pos, root_rot, root_vel, root_ang_vel, 
                                              dof_pos, dof_vel, key_pos, 
                                              self._local_root_obs, self._root_height_obs, 
                                              self._dof_obs_size, self._dof_offsets)
        self._hist_amp_obs_buf[env_ids] = amp_obs_demo.view(self._hist_amp_obs_buf[env_ids].shape)
        return
    
    def _set_env_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):
        #make sure character is placed correctly by scaling z component of root_pos by leg's scale factor
        #scale_leg_for_envs = to_torch(self._scale_leg, device=self.device)[to_torch(self.env_char_mapping, device=self.device, dtype=torch.long)[env_ids]]
        #root_pos[:,2] = torch.mul(root_pos[:,2],scale_leg_for_envs)
        self._humanoid_root_states[env_ids, 0:3] = root_pos
        self._humanoid_root_states[env_ids, 3:7] = root_rot
        self._humanoid_root_states[env_ids, 7:10] = root_vel
        self._humanoid_root_states[env_ids, 10:13] = root_ang_vel
        
        self._dof_pos[env_ids] = dof_pos
        self._dof_vel[env_ids] = dof_vel
        return

    def _update_hist_amp_obs(self, env_ids=None):
        if (env_ids is None):
            self._hist_amp_obs_buf[:] = self._amp_obs_buf[:, 0:(self._num_amp_obs_steps - 1)]
        else:
            self._hist_amp_obs_buf[env_ids] = self._amp_obs_buf[env_ids, 0:(self._num_amp_obs_steps - 1)]
        return
    
    def _compute_amp_observations(self, env_ids=None):
        key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
        if (env_ids is None):
            self._curr_amp_obs_buf[:] = build_amp_observations(self._rigid_body_pos[:, 0, :],
                                                               self._rigid_body_rot[:, 0, :],
                                                               self._rigid_body_vel[:, 0, :],
                                                               self._rigid_body_ang_vel[:, 0, :],
                                                               self._dof_pos, self._dof_vel, key_body_pos,
                                                               self._local_root_obs, self._root_height_obs, 
                                                               self._dof_obs_size, self._dof_offsets)
        else:
            self._curr_amp_obs_buf[env_ids] = build_amp_observations(self._rigid_body_pos[env_ids][:, 0, :],
                                                                   self._rigid_body_rot[env_ids][:, 0, :],
                                                                   self._rigid_body_vel[env_ids][:, 0, :],
                                                                   self._rigid_body_ang_vel[env_ids][:, 0, :],
                                                                   self._dof_pos[env_ids], self._dof_vel[env_ids], key_body_pos[env_ids],
                                                                   self._local_root_obs, self._root_height_obs, 
                                                                   self._dof_obs_size, self._dof_offsets)
        return


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def build_amp_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos, 
                           local_root_obs, root_height_obs, dof_obs_size, dof_offsets):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, int, List[int]) -> Tensor
    root_h = root_pos[:, 2:3]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    if (local_root_obs):
        root_rot_obs = quat_mul(heading_rot, root_rot)
    else:
        root_rot_obs = root_rot
    root_rot_obs = torch_utils.quat_to_tan_norm(root_rot_obs)
    
    if (not root_height_obs):
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h
    
    local_root_vel = quat_rotate(heading_rot, root_vel)
    local_root_ang_vel = quat_rotate(heading_rot, root_ang_vel)

    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand
    
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(local_key_body_pos.shape[0] * local_key_body_pos.shape[1], local_key_body_pos.shape[2])
    flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                               heading_rot_expand.shape[2])
    local_end_pos = quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(local_key_body_pos.shape[0], local_key_body_pos.shape[1] * local_key_body_pos.shape[2])
    
    dof_obs = dof_to_obs(dof_pos, dof_obs_size, dof_offsets)
    obs = torch.cat((root_h_obs, root_rot_obs, local_root_vel, local_root_ang_vel, dof_obs, dof_vel, flat_local_key_pos), dim=-1)
    return obs