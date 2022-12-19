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

import torch

from isaacgym import gymapi
from isaacgym import gymtorch

from env.tasks.humanoid_amp import HumanoidAMP
from isaacgym.torch_utils import *

from utils import torch_utils
from utils import gym_util

import random


class HumanoidAMPGetup(HumanoidAMP):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        
        self._recovery_episode_prob = cfg["env"]["recoveryEpisodeProb"]
        self._recovery_steps = cfg["env"]["recoverySteps"]
        self._fall_init_prob = cfg["env"]["fallInitProb"]

        self._reset_fall_env_ids = []

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        
        self._recovery_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)

        self._generate_fall_states()

        return

    
    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)

        self._update_recovery_count()
        return

    def _generate_fall_states(self):
        max_steps = 150
        # first try to generate random poses from base character and transform them when reset (but hard to find scale factor)
        ##if self.randomize:
        ##    # find envs with base character to generate fall states from this and adjust for scale when character put into state
        ##    base_envs = np.where(np.array(self.env_char_mapping)[np.arange(0,self.num_envs)] == self.base_char_idx)
        ##    env_ids = base_envs
        ##else:
        ##    # generate fall states with all envs
        ##    env_ids = to_torch(np.arange(self.num_envs), device=self.device, dtype=torch.long)
        env_ids = to_torch(np.arange(self.num_envs), device=self.device, dtype=torch.long)
        root_states = self._initial_humanoid_root_states[env_ids].clone()
        root_states[..., 3:7] = torch.randn_like(root_states[..., 3:7])
        root_states[..., 3:7] = torch.nn.functional.normalize(root_states[..., 3:7], dim=-1)
        self._humanoid_root_states[env_ids] = root_states
        
        env_ids_int32 = self._humanoid_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))


        rand_actions = np.random.uniform(-0.5, 0.5, size=[self.num_envs, self.get_action_size()])
        rand_actions = to_torch(rand_actions, device=self.device)
        self.pre_physics_step(rand_actions)

        # step physics and render each frame
        for i in range(max_steps):
            self.render()
            self.gym.simulate(self.sim)
            
        self._refresh_sim_tensors()

        self._fall_root_states = self._humanoid_root_states[env_ids].clone()
        self._fall_dof_pos = self._dof_pos[env_ids].clone()
        self._fall_root_states[:, 7:13] = 0

        #if self.randomize:
            # generate multiple fall root states (#env_per_char) for each character
            #self._fall_root_states = self._fall_root_states.view(self.num_chars, self.envs_per_file, 13)
            #self._fall_dof_pos = self._fall_dof_pos.view(self.num_chars, self.envs_per_file, 31)

        self._fall_dof_vel = torch.zeros_like(self._fall_dof_pos, device=self.device, dtype=torch.float)

        return

    def _reset_actors(self, env_ids):
        num_envs = env_ids.shape[0]
        recovery_probs = to_torch(np.array([self._recovery_episode_prob] * num_envs), device=self.device)
        recovery_mask = torch.bernoulli(recovery_probs) == 1.0
        terminated_mask = (self._terminate_buf[env_ids] == 1)
        recovery_mask = torch.logical_and(recovery_mask, terminated_mask)

        recovery_ids = env_ids[recovery_mask]
        if (len(recovery_ids) > 0):
            self._reset_recovery_episode(recovery_ids)
            

        nonrecovery_ids = env_ids[torch.logical_not(recovery_mask)]
        fall_probs = to_torch(np.array([self._fall_init_prob] * nonrecovery_ids.shape[0]), device=self.device)
        fall_mask = torch.bernoulli(fall_probs) == 1.0
        fall_ids = nonrecovery_ids[fall_mask]
        if (len(fall_ids) > 0):
            self._reset_fall_episode(fall_ids)
            

        nonfall_ids = nonrecovery_ids[torch.logical_not(fall_mask)]
        if (len(nonfall_ids) > 0):
            super()._reset_actors(nonfall_ids)
            self._recovery_counter[nonfall_ids] = 0

        return

    def _reset_recovery_episode(self, env_ids):
        self._recovery_counter[env_ids] = self._recovery_steps
        return
    
    def _reset_fall_episode(self, env_ids):
        if self.randomize:
            indices = np.random.randint(env_ids.shape[0]*[0],self.num_envs_per_char[self.env_char_mapping[env_ids.tolist()]].tolist())
            fall_state_ids = [(self.char_env_mapping[self.env_char_mapping[env_ids[i].item()]][indices[i]])  for i in range(env_ids.shape[0])]         
            #fall_state_ids = torch.randint_like(env_ids, low=0, high=self._fall_root_states.shape[1])
            #fall_root_states = self._fall_root_states[to_torch(self.env_char_mapping, device=self.device, dtype=torch.long)[env_ids],fall_state_ids,:]
            #dof_pos = self._fall_dof_pos[to_torch(self.env_char_mapping, device=self.device, dtype=torch.long)[env_ids],fall_state_ids,:]
            #dof_vel = self._fall_dof_vel[to_torch(self.env_char_mapping, device=self.device, dtype=torch.long)[env_ids],fall_state_ids,:]
        else:
            fall_state_ids = torch.randint_like(env_ids, low=0, high=self._fall_root_states.shape[0])
        fall_root_states = self._fall_root_states[fall_state_ids]
        dof_pos = self._fall_dof_pos[fall_state_ids]
        dof_vel = self._fall_dof_vel[fall_state_ids]
        #if self.randomize:
        #    #make sure character is placed correctly by scaling z component of _fall_root_state by leg's scale factor
        #    test_0 = to_torch(self._scale_leg, device=self.device)
        #    test_1 = to_torch(self.env_char_mapping, device=self.device, dtype=torch.long)[env_ids]
        #    scale_leg_for_envs = to_torch(self._scale_leg, device=self.device)[to_torch(self.env_char_mapping, device=self.device, dtype=torch.long)[env_ids]]
        #    test_3 = fall_root_states[:,2]
        #    test_4 = torch.mul(fall_root_states[:,2],scale_leg_for_envs)
        #    fall_root_states[:,2] = torch.mul(fall_root_states[:,2],scale_leg_for_envs)

        self._humanoid_root_states[env_ids] = fall_root_states
        self._dof_pos[env_ids] = dof_pos
        self._dof_vel[env_ids] = dof_vel
        #self._humanoid_root_states[env_ids] = self._fall_root_states[fall_state_ids]
        #self._dof_pos[env_ids] = self._fall_dof_pos[fall_state_ids]
        #self._dof_vel[env_ids] = self._fall_dof_vel[fall_state_ids]
        self._recovery_counter[env_ids] = self._recovery_steps
        self._reset_fall_env_ids = env_ids

        ##if self.randomize:
        ##    #avoid jumps
        ##    #right_hand, sword, left_hand
        ##    #bodies_sizes = [0.04, 0.11, 0.04]
        ##    root_rot_expanded = self._humanoid_root_states[env_ids, 3:7].unsqueeze(-2)
        ##    root_rot_expanded = root_rot_expanded.repeat((1, self._rigid_body_pos.shape[1], 1))
        ##    new_rigid_body_pos_l = quat_rotate(root_rot_expanded.view(root_rot_expanded.shape[0]*root_rot_expanded.shape[1], root_rot_expanded.shape[2]),self._rigid_body_pos[env_ids,:,:].view(self._rigid_body_pos[env_ids].shape[0]*self._rigid_body_pos[env_ids].shape[1], self._rigid_body_pos[env_ids].shape[2]))
        ##    new_rigid_body_pos_l = new_rigid_body_pos_l.view(self._rigid_body_pos[env_ids].shape[0],self._rigid_body_pos.shape[1], self._rigid_body_pos.shape[2] )
        ##    difference = self._humanoid_root_states[env_ids, 2] - new_rigid_body_pos_l[:,0,2]
        ##    difference_expanded = difference.unsqueeze(-1)
        ##    new_rigid_body_pos = new_rigid_body_pos_l[:,:,2] + difference_expanded
        ##    test_0 = torch.min(new_rigid_body_pos, dim=1).values
        ##    index = torch.min(new_rigid_body_pos, dim=1).indices
        ##    test_01 = torch.zeros_like(test_0)
        ##    test_1 = torch.gt(torch.zeros_like(test_0),torch.min(new_rigid_body_pos, dim=1).values)
        ##    test_2 = torch.sum(torch.gt(torch.zeros_like(test_0),torch.min(new_rigid_body_pos, dim=1).values))
        ##    if torch.sum(torch.gt(torch.zeros_like(test_0),torch.min(new_rigid_body_pos, dim=1).values))>0:
        ##        test_4 = torch.clamp_max(torch.min(new_rigid_body_pos, dim=1).values, max=0)
        ##        self._humanoid_root_states[env_ids, 2] -= torch.clamp_max(torch.min(new_rigid_body_pos, dim=1).values, max=0)
##
        ##    env_ids_int32 = self._humanoid_actor_ids[env_ids]
        ##    self.gym.set_actor_root_state_tensor_indexed(self.sim,
        ##                                                 gymtorch.unwrap_tensor(self._root_states),
        ##                                                 gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        ##    self.render()
        ##    self.gym.simulate(self.sim)
        ##    self.render()
        ##    self.gym.simulate(self.sim)
##

        ##    new_diff = new_rigid_body_pos_l[:,:,2] + self._humanoid_root_states[env_ids, 2].unsqueeze(-1)

            #env_ids_int32 = self._humanoid_actor_ids[env_ids]
            #self.gym.set_actor_root_state_tensor_indexed(self.sim,
            #                                             gymtorch.unwrap_tensor(self._root_states),
            #                                             gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
#   
            #self.gym.refresh_actor_root_state_tensor(self.sim)
            #self.gym.refresh_rigid_body_state_tensor(self.sim)
#   
            #test_0 = torch.min(self._rigid_body_pos[env_ids,:,2], dim=1).values
            #test_1 = torch.gt(torch.zeros_like(test_0),torch.min(self._rigid_body_pos[env_ids,:,2], dim=1).values)
            #test_2 = torch.sum(torch.gt(torch.zeros_like(test_0),torch.min(self._rigid_body_pos[env_ids,:,2], dim=1).values))>0
            #if torch.sum(torch.gt(torch.zeros_like(test_0),torch.min(self._rigid_body_pos[env_ids,:,2], dim=1).values))>0:
            #    correction = torch.clamp_min(torch.min(self._rigid_body_pos[env_ids,:,2], dim=1).values, min=0)


            #new_root_pos = quat_rotate(self._humanoid_root_states[env_ids, 3:7], self._rigid_body_pos[env_ids,0,:])
            #if torch.sum(torch.gt(new_root_pos[:,2], self._humanoid_root_states[env_ids,2]))>0:
            #    correction = new_root_pos[:,2] - self._humanoid_root_states[env_ids,2]
            #    self._humanoid_root_states[env_ids,2] += correction

            #test_0=self._rigid_body_pos[env_ids,0,2]
            #test_1=self._humanoid_root_states[env_ids,2]
            #test_2=torch.gt(self._rigid_body_pos[env_ids,0,2],self._humanoid_root_states[env_ids,2])
            #test_3=torch.sum(torch.gt(self._rigid_body_pos[env_ids,0,2],self._humanoid_root_states[env_ids,2]))
            #if torch.sum(torch.gt(self._rigid_body_pos[env_ids,0,2],self._humanoid_root_states[env_ids,2]))>0:
            #    correction = self._rigid_body_pos[env_ids,0,2] - self._humanoid_root_states[env_ids,2]
            #    scale = to_torch(self._scale_leg, device=self.device)[to_torch(self.env_char_mapping, device=self.device, dtype=torch.long)[env_ids]]
            #    self._humanoid_root_states[env_ids,2] += correction + 0.89 * to_torch(self._scale_leg, device=self.device)[to_torch(self.env_char_mapping, device=self.device, dtype=torch.long)[env_ids]]

        ##    x_0 = self._rigid_body_pos[env_ids,0,2]
        ##    x_1 = self._humanoid_root_states[env_ids,2]

        return
    
    def _reset_envs(self, env_ids):
        self._reset_fall_env_ids = []
        super()._reset_envs(env_ids)
        return

    def _init_amp_obs(self, env_ids):
        super()._init_amp_obs(env_ids)

        if (len(self._reset_fall_env_ids) > 0):
            self._init_amp_obs_default(self._reset_fall_env_ids)

        return

    def _update_recovery_count(self):
        self._recovery_counter -= 1
        self._recovery_counter = torch.clamp_min(self._recovery_counter, 0)
        return

    def _compute_reset(self):
        super()._compute_reset()

        is_recovery = self._recovery_counter > 0
        self.reset_buf[is_recovery] = 0
        self._terminate_buf[is_recovery] = 0
        return