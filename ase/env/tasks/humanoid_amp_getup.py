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

        # trained to GetUp during recovery_steps steps after early termination and reset
        # evaluate for recovery_steps + 10 steps (time to get up and successfully stand for 10 steps)
        #self._recovery_steps += 10

        self._reset_fall_env_ids = []

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        
        self._recovery_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)

        if self.eval:
            self.not_terminated = torch.zeros_like(self._terminate_buf)
            self.success_envs = torch.zeros([self.num_envs], device=self.device, dtype=torch.int64)
            self.failure_envs = torch.zeros([self.num_envs], device=self.device, dtype=torch.int64)
            self.games_played_counter = torch.zeros_like(self.progress_buf)

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

        self._humanoid_root_states[env_ids] = fall_root_states
        self._dof_pos[env_ids] = dof_pos
        self._dof_vel[env_ids] = dof_vel
        #self._humanoid_root_states[env_ids] = self._fall_root_states[fall_state_ids]
        #self._dof_pos[env_ids] = self._fall_dof_pos[fall_state_ids]
        #self._dof_vel[env_ids] = self._fall_dof_vel[fall_state_ids]
        self._recovery_counter[env_ids] = self._recovery_steps
        self._reset_fall_env_ids = env_ids

        return
    
    def _reset_envs(self, env_ids):
        self._reset_fall_env_ids = []
        # reset self.not_terminated
        if self.eval:
            self.not_terminated[env_ids] = torch.zeros_like(self._terminate_buf[env_ids])
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

        if self.eval:
            self.success_envs = torch.zeros_like(self.progress_buf)
            # when for more than 10 consecutive time steps no early termination, consider GET UP task successfull
            self.not_terminated += torch.logical_not(self._terminate_buf)
            self.not_terminated = torch.mul(self.not_terminated, torch.logical_not(self._terminate_buf)) # assures that consecutiveness
            self.success_envs = torch.where(self.not_terminated >= 10, torch.ones_like(self.progress_buf), self.success_envs) 
            success_envs_ids = self.success_envs.nonzero(as_tuple=False).flatten()
            # reset task
            self.reset_buf[success_envs_ids] = 1 
            if (not self.headless):
                for i in success_envs_ids:
                    self.gym.set_rigid_body_color(self.envs[i], self.humanoid_handles[i], 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.2, 0.1, 0.9))
                    super().render()

            # when during recovery steps no success is achieved condsider GET UP task as failure
            self.failure_envs = torch.logical_not(is_recovery)
            self.failure_envs = torch.where(self.success_envs > 0, torch.zeros_like(self.failure_envs), self.failure_envs)
            failure_envs_ids = self.failure_envs.nonzero(as_tuple=False).flatten()

            # reset task
            self.reset_buf[failure_envs_ids] = 1  

            assert self.success_envs.isnan().sum()==0, f"success envs is nan: {self.success_envs.isnan().sum()}"
            assert self.failure_envs.isnan().sum()==0, f"failure envs is nan: {self.failure_envs.isnan().sum()}"
        
            #when an episode end; consider it as game played
            self.games_played_counter += 1
            game_played = self.games_played_counter >= (self.max_episode_length - 1)
            self.games_played_count = game_played.sum().item()
            if self.games_played_count > 0:
                #when game played ends, env should be reset
                self.reset_buf[game_played] = 1
                self.games_played_counter[game_played] = 0

        
        self._terminate_buf[is_recovery] = 0
        return