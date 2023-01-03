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
import yaml

from rl_games.algos_torch import players
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.common.player import BasePlayer

import numpy as np

class CommonPlayer(players.PpoPlayerContinuous):
    def __init__(self, config):
        BasePlayer.__init__(self, config)
        self.network = config['network']
        
        self._setup_action_space()
        self.mask = [False]

        self.normalize_input = self.config['normalize_input']
        
        net_config = self._build_net_config()
        self._build_net(net_config)   
        return

    def run(self, eval=False):
        n_games = self.games_num
        render = self.render_env
        n_game_life = self.n_game_life
        is_determenistic = self.is_determenistic
        sum_rewards = 0
        sum_steps = 0
        sum_game_res = 0
        #n_games = n_games * n_game_life
        n_games = 4096
        games_played = 0
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None

        op_agent = getattr(self.env, "create_agent", None)
        if op_agent:
            agent_inited = True

        if has_masks_func:
            has_masks = self.env.has_action_mask()

        need_init_rnn = self.is_rnn
        for _ in range(n_games):
            if games_played >= n_games:
                break
                
            if eval:
                #evaluation metrics
                successes = torch.zeros_like(self.env.task.progress_buf, dtype=torch.float32)
                failures = torch.zeros_like(self.env.task.progress_buf, dtype=torch.float32)
                steps_to_succeed = torch.zeros_like(self.env.task.progress_buf, dtype=torch.float32)
                success_counts = 0
                failure_counts = 0
                completion_time = 0

            obs_dict = self.env_reset()
            batch_size = 1
            batch_size = self.get_batch_size(obs_dict['obs'], batch_size)

            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False

            cr = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            steps = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

            print_game_res = False

            done_indices = []

            for n in range(self.max_steps):
                obs_dict = self.env_reset(done_indices)

                if has_masks:
                    masks = self.env.get_action_mask()
                    action = self.get_masked_action(obs_dict, masks, is_determenistic)
                else:
                    action = self.get_action(obs_dict, is_determenistic)
                obs_dict, r, done, info =  self.env_step(self.env, action)
                cr += r
                steps += 1
  
                #compute total successes and failures over all environements
                if eval:
                    successes += self.env.task.success_envs
                    assert successes.isnan().sum() == 0, f"successes count is nan: {successes.isnan().sum()}"
                    failures += self.env.task.failure_envs
                    assert failures.isnan().sum() == 0, f"Failures count is nan: {failures.isnan().sum()}"
                    #include steps 
                    steps_to_succeed += torch.mul(steps, self.env.task.success_envs)
                    assert steps_to_succeed.isnan().sum() == 0, f"steps_to_succeed is nan: {steps_to_succeed.isnan().sum()}"
                self._post_step(info)

                if render:
                    self.env.render(mode = 'human')
                    time.sleep(self.render_sleep)

                all_done_indices = done.nonzero(as_tuple=False)
                done_indices = all_done_indices[::self.num_agents]
                done_count = len(done_indices)
                if eval:
                    games_played += self.env.task.games_played_count
                else:
                    games_played += done_count

                if done_count > 0:
                    if self.is_rnn:
                        for s in self.states:
                            s[:,all_done_indices,:] = s[:,all_done_indices,:] * 0.0

                    cur_rewards = cr[done_indices].sum().item()
                    cur_steps = steps[done_indices].sum().item()

                    cr = cr * (1.0 - done.float())
                    steps = steps * (1.0 - done.float())
                    sum_rewards += cur_rewards
                    sum_steps += cur_steps

                    game_res = 0.0
                    if isinstance(info, dict):
                        if 'battle_won' in info:
                            print_game_res = True
                            game_res = info.get('battle_won', 0.5)
                        if 'scores' in info:
                            print_game_res = True
                            game_res = info.get('scores', 0.5)
                    if self.print_stats:
                        if print_game_res:
                            print('reward:', cur_rewards/done_count, 'steps:', cur_steps/done_count, 'w:', game_res)
                        else:
                            print('reward:', cur_rewards/done_count, 'steps:', cur_steps/done_count)

                    #max_t = 120

                    # in each step calculate metrics score
                    # compute completion time metric
                    #task_selected = ['HumanoidAMPGetup', 'HumanoidReach', 'HumanoidLocation']
                    #task = 'HumanoidAMPGetup'
                    #if task in task_selected:
                    #print("metrics to be implemented")
                    #if self.time < max_t:
                    #    if not is_done:
                    #        self.time += 1
                    #    else:
                    #        print("completion time metric: ")
                    #        print(self.time)
                    #else:
                    #    print("completion time metric: ")
                    #    print(self.time)
                    sum_game_res += game_res
                    if batch_size//self.num_agents == 1 or games_played >= n_games:
                        break
                
                done_indices = done_indices[:, 0]

        print(sum_rewards)
        if print_game_res:
            print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:', sum_steps / games_played * n_game_life, 'winrate:', sum_game_res / games_played * n_game_life)
        else:
            print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:', sum_steps / games_played * n_game_life)

        if eval:
            #compute metrics
            success_counts = successes.sum().item()
            failure_counts = failures.sum().item()
            #steps mean (divide steps per env by successes per env and take mean)
            #steps_per_env = torch.where(successes.bool(), torch.div(hlc_steps_to_succeed, successes).float(), torch.zeros_like(successes))
            #steps_mean = torch.mean(steps_per_env)
            #completion time (sum over all steps than divide by success_count)
            completion_time = steps_to_succeed.sum().item() / success_counts

            #assert steps_per_env.isnan().sum() == 0, f"steps_per_env is nan: {steps_per_env.isnan().sum()}"
            #assert steps_mean.isnan().sum() == 0, f"steps_mean is nan: {steps_mean.isnan().sum()}"

            evaluation = {'total_successes':success_counts,
                            'total_failures':failure_counts,
                            'total_tasks':success_counts +failure_counts,
                            'average_completion_time':completion_time,
                            'success_rate':success_counts/(success_counts +failure_counts),
                            'failure_rate':1-success_counts/(success_counts +failure_counts),
                            'av_reward':sum_rewards / games_played * n_game_life,
                            'av steps':sum_steps / games_played * n_game_life}

            task_name = self.env.task.cfg["args"].__getattribute__('task')
            if task_name == "HumanoidReach":
                tar_change_steps = 112*2
            elif task_name == "HumanoidLocation":
                tar_change_steps = 299
            else:
                tar_change_steps = 26*2
            
            filename = self.env.task.cfg["env"]["asset"]["assetFileName"]
            filename = filename.replace("mjcf/", "")
            file = filename.replace(".xml", "_" + task_name + "_eval.yaml")

            data = {'character':filename,
                    'task':task_name,
                    'numEnvs':self.env.task.cfg["env"]['numEnvs'],
                    'episodeLength':self.env.task.cfg["env"]['episodeLength'],
                    'enableEarlyTermination':self.env.task.cfg["env"]['enableEarlyTermination'],
                    'numGames':n_games,
                    'GamesPlayed':games_played,
                    'initState':self.env.task.cfg["env"]['stateInit'],
                    #'tar_change_step':self.env.task.cfg["env"]['tarChangeStepsMax'],
                    #'tar_change_step':300,
                    'tar_change_step':tar_change_steps,
                    'evaluation':evaluation}
            with open(file, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)


        return

    def obs_to_torch(self, obs):
        obs = super().obs_to_torch(obs)
        obs_dict = {
            'obs': obs
        }
        return obs_dict

    def get_action(self, obs_dict, is_determenistic = False):
        output = super().get_action(obs_dict['obs'], is_determenistic)
        return output

    def env_step(self, env, actions):
        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()
        obs, rewards, dones, infos = env.step(actions)

        if hasattr(obs, 'dtype') and obs.dtype == np.float64:
            obs = np.float32(obs)
        if self.value_size > 1:
            rewards = rewards[0]
        if self.is_tensor_obses:
            return obs, rewards.to(self.device), dones.to(self.device), infos
        else:
            if np.isscalar(dones):
                rewards = np.expand_dims(np.asarray(rewards), 0)
                dones = np.expand_dims(np.asarray(dones), 0)
            return self.obs_to_torch(obs), torch.from_numpy(rewards), torch.from_numpy(dones), infos

    def _build_net(self, config):
        self.model = self.network.build(config)
        self.model.to(self.device)
        self.model.eval()
        self.is_rnn = self.model.is_rnn()
        if self.normalize_input:
            obs_shape = torch_ext.shape_whc_to_cwh(self.obs_shape)
            self.running_mean_std = RunningMeanStd(obs_shape).to(self.device)
            self.running_mean_std.eval() 
        return

    def env_reset(self, env_ids=None):
        obs = self.env.reset(env_ids)
        return self.obs_to_torch(obs)

    def _post_step(self, info):
        return

    def _build_net_config(self):
        obs_shape = torch_ext.shape_whc_to_cwh(self.obs_shape)
        config = {
            'actions_num' : self.actions_num,
            'input_shape' : obs_shape,
            'num_seqs' : self.num_agents
        } 
        return config

    def _setup_action_space(self):
        self.actions_num = self.action_space.shape[0] 
        self.actions_low = torch.from_numpy(self.action_space.low.copy()).float().to(self.device)
        self.actions_high = torch.from_numpy(self.action_space.high.copy()).float().to(self.device)
        return