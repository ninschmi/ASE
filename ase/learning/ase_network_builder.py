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

from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import layers
from rl_games.algos_torch import network_builder

import torch
import torch.nn as nn
import numpy as np
import enum

from learning import amp_network_builder

ENC_LOGIT_INIT_SCALE = 0.1

class LatentType(enum.Enum):
    uniform = 0
    sphere = 1

class ASEBuilder(amp_network_builder.AMPBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    class Network(amp_network_builder.AMPBuilder.Network):
        def __init__(self, params, **kwargs):
            actions_num = kwargs.get('actions_num')
            input_shape = kwargs.get('input_shape')
            self.value_size = kwargs.get('value_size', 1)
            self.num_seqs = num_seqs = kwargs.get('num_seqs', 1)
            amp_input_shape = kwargs.get('amp_input_shape')
            self._ase_latent_shape = kwargs.get('ase_latent_shape')

            network_builder.NetworkBuilder.BaseNetwork.__init__(self)
            
            self.load(params)

            actor_out_size, critic_out_size = self._build_actor_critic_net(input_shape, self._ase_latent_shape)
            if self.mlp_correct:
                self.shape_parameter = kwargs.get('scale_factors')
                actor_corr_out_size, critic_corr_out_out_size = self._build_correction_net(actions_num, self.value_size)

            self.value = torch.nn.Linear(critic_out_size, self.value_size)
            self.value_act = self.activations_factory.create(self.value_activation)
            if self.mlp_correct:
                self.value_corr = torch.nn.Linear(critic_corr_out_out_size, self.value_size)
            
            if self.is_discrete:
                self.logits = torch.nn.Linear(actor_out_size, actions_num)
                if self.mlp_correct:
                    self.logits_corr = torch.nn.Linear(actor_corr_out_size, actions_num)
            '''
                for multidiscrete actions num is a tuple
            '''
            if self.is_multi_discrete:
                self.logits = torch.nn.ModuleList([torch.nn.Linear(actor_out_size, num) for num in actions_num])
                if self.mlp_correct:
                    self.logits_corr = torch.nn.ModuleList([torch.nn.Linear(actor_corr_out_size, num) for num in actions_num])
            if self.is_continuous:
                self.mu = torch.nn.Linear(actor_out_size, actions_num)
                self.mu_act = self.activations_factory.create(self.space_config['mu_activation']) 
                mu_init = self.init_factory.create(**self.space_config['mu_init'])
                if self.mlp_correct:
                    self.mu_corr = torch.nn.Linear(actor_corr_out_size, actions_num)
                self.sigma_act = self.activations_factory.create(self.space_config['sigma_activation']) 

                sigma_init = self.init_factory.create(**self.space_config['sigma_init'])

                if (not self.space_config['learn_sigma']):
                    self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=False, dtype=torch.float32), requires_grad=False)
                elif self.space_config['fixed_sigma']:
                    self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)
                else:
                    self.sigma = torch.nn.Linear(actor_out_size, actions_num)

            mlp_init = self.init_factory.create(**self.initializer)
            if self.has_cnn:
                cnn_init = self.init_factory.create(**self.cnn['initializer'])

            for m in self.modules():         
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                    cnn_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)    

            self.actor_mlp.init_params()
            self.critic_mlp.init_params()

            if self.mlp_correct:
                self.residual_weight = 0.5
                self.actor_corr_mlp.init_params()
                self.critic_corr_mlp.init_params()

                self.actor_mlp.requires_grad_(False)
                self.critic_mlp.requires_grad_(False)
                self.value.requires_grad_(False)
                if self.is_discrete:
                    self.logits.requires_grad_(False)
                if self.is_multi_discrete:
                    [layer.requires_grad_(False) for layer in self.logits]
                if self.is_continuous:
                    self.mu.requires_grad_(False)

            if self.is_continuous:
                mu_init(self.mu.weight)
                if self.mlp_correct:
                    mu_init(self.mu_corr.weight)
                if self.space_config['fixed_sigma']:
                    sigma_init(self.sigma)
                else:
                    sigma_init(self.sigma.weight)

            self._build_disc(amp_input_shape)
            self._build_enc(amp_input_shape)

            return
        
        def load(self, params):
            super().load(params)

            self._enc_units = params['enc']['units']
            self._enc_activation = params['enc']['activation']
            self._enc_initializer = params['enc']['initializer']
            self._enc_separate = params['enc']['separate']

            self.mlp_correct = params.get('ac_corr', False)
            self.toggle = True
            return

        def forward(self, obs_dict):
            obs = obs_dict['obs']
            ase_latents = obs_dict['ase_latents']
            states = obs_dict.get('rnn_states', None)
            use_hidden_latents = obs_dict.get('use_hidden_latents', False)
            batch_shape_parameters = obs_dict.get('batch_shape_parameters', None)

            actor_outputs = self.eval_actor(obs, ase_latents, use_hidden_latents, batch_shape_parameters)
            value = self.eval_critic(obs, ase_latents, use_hidden_latents, batch_shape_parameters)

            output = actor_outputs + (value, states)

            return output

        def eval_critic(self, obs, ase_latents, use_hidden_latents=False, batch_shape_parameters=None):
            c_out = self.critic_cnn(obs)
            c_out = c_out.contiguous().view(c_out.size(0), -1)
            
            c_out = self.critic_mlp(c_out, ase_latents, use_hidden_latents)
            value = self.value_act(self.value(c_out))
            if self.mlp_correct:
                c_out = self.critic_corr_mlp(value, batch_shape_parameters)
                value_corrected = self.value_act(self.value_corr(c_out))
                value = self.value_act(self.value_corr(c_out))
                value = (1-self.residual_weight) * value + self.residual_weight * value_corrected
            return value

        def eval_actor(self, obs, ase_latents, use_hidden_latents=False, batch_shape_parameters=None):
            a_out = self.actor_cnn(obs)
            a_out = a_out.contiguous().view(a_out.size(0), -1)
            a_out = self.actor_mlp(a_out, ase_latents, use_hidden_latents)
                     
            if self.is_discrete:
                logits = self.logits(a_out)
                if self.mlp_correct:
                    a_out = self.actor_corr_mlp(logits, batch_shape_parameters)
                    logits_corrected = self.logits_corr(a_out)
                    logits = (1-self.residual_weight)* logits + self.residual_weight * logits_corrected
                return logits

            if self.is_multi_discrete:
                logits = [logit(a_out) for logit in self.logits]
                if self.mlp_correct:
                    a_out = self.actor_corr_mlp(logits, batch_shape_parameters)
                    logits_corrected = [logit(a_out) for logit in self.logits_corr]
                    logits = (1-self.residual_weight)* logits + self.residual_weight * logits_corrected
                return logits

            if self.is_continuous:
                mu = self.mu_act(self.mu(a_out))
                if self.mlp_correct:
                    a_out = self.actor_corr_mlp(mu, batch_shape_parameters)
                    #mu = self.mu_act(self.mu_corr(a_out))
                    mu_corrected = self.mu_act(self.mu_corr(a_out))
                    mu = (1-self.residual_weight)* mu + self.residual_weight* mu_corrected
                if self.space_config['fixed_sigma']:
                    sigma = mu * 0.0 + self.sigma_act(self.sigma)
                else:
                    sigma = self.sigma_act(self.sigma(a_out))

                return mu, sigma
            return

        def get_enc_weights(self):
            weights = []
            for m in self._enc_mlp.modules():
                if isinstance(m, nn.Linear):
                    weights.append(torch.flatten(m.weight))

            weights.append(torch.flatten(self._enc.weight))
            return weights

        def _build_actor_critic_net(self, input_shape, ase_latent_shape):
            style_units = [512, 256]
            style_dim = ase_latent_shape[-1]

            self.actor_cnn = nn.Sequential()
            self.critic_cnn = nn.Sequential()
            
            act_fn = self.activations_factory.create(self.activation)
            initializer = self.init_factory.create(**self.initializer)

            self.actor_mlp = AMPStyleCatNet1(obs_size=input_shape[-1],
                                             ase_latent_size=ase_latent_shape[-1],
                                             units=self.units,
                                             activation=act_fn,
                                             style_units=style_units,
                                             style_dim=style_dim,
                                             initializer=initializer)

            if self.separate:
                self.critic_mlp = AMPMLPNet(obs_size=input_shape[-1],
                                            ase_latent_size=ase_latent_shape[-1],
                                            units=self.units,
                                            activation=act_fn,
                                            initializer=initializer)

            actor_out_size = self.actor_mlp.get_out_size()
            critic_out_size = self.critic_mlp.get_out_size()

            return actor_out_size, critic_out_size

        def _build_correction_net(self, actor_dim, critic_dim):      
            act_fn = self.activations_factory.create(self.activation)
            initializer = self.init_factory.create(**self.initializer)

            self.actor_corr_mlp = CorrectionNet(actor_dim, act_fn, initializer, self.shape_parameter)
            if self.separate:
                self.critic_corr_mlp = CorrectionNet(critic_dim, act_fn, initializer, self.shape_parameter)
            
            actor_corr_out = self.actor_corr_mlp.get_out_size()
            critic_corr_out = self.critic_corr_mlp.get_out_size()

            return actor_corr_out, critic_corr_out

        def _build_enc(self, input_shape):
            if (self._enc_separate):
                self._enc_mlp = nn.Sequential()
                mlp_args = {
                    'input_size' : input_shape[0], 
                    'units' : self._enc_units, 
                    'activation' : self._enc_activation, 
                    'dense_func' : torch.nn.Linear
                }
                self._enc_mlp = self._build_mlp(**mlp_args)

                mlp_init = self.init_factory.create(**self._enc_initializer)
                for m in self._enc_mlp.modules():
                    if isinstance(m, nn.Linear):
                        mlp_init(m.weight)
                        if getattr(m, "bias", None) is not None:
                            torch.nn.init.zeros_(m.bias)
            else:
                self._enc_mlp = self._disc_mlp

            mlp_out_layer = list(self._enc_mlp.modules())[-2]
            mlp_out_size = mlp_out_layer.out_features
            self._enc = torch.nn.Linear(mlp_out_size, self._ase_latent_shape[-1])
            
            torch.nn.init.uniform_(self._enc.weight, -ENC_LOGIT_INIT_SCALE, ENC_LOGIT_INIT_SCALE)
            torch.nn.init.zeros_(self._enc.bias) 
            
            return

        def eval_enc(self, amp_obs):
            enc_mlp_out = self._enc_mlp(amp_obs)
            enc_output = self._enc(enc_mlp_out)
            enc_output = torch.nn.functional.normalize(enc_output, dim=-1)

            return enc_output

        def sample_latents(self, n):
            device = next(self._enc.parameters()).device
            z = torch.normal(torch.zeros([n, self._ase_latent_shape[-1]], device=device))
            z = torch.nn.functional.normalize(z, dim=-1)
            return z

        def train(self, mode=True):
            super().train(mode)
            if mode and self.mlp_correct:
                self.toggle = not self.toggle
                self.actor_corr_mlp.requires_grad_(self.toggle)
                self.critic_corr_mlp.requires_grad_(not self.toggle)

    def build(self, name, **kwargs):
        net = ASEBuilder.Network(self.params, **kwargs)
        return net


class AMPMLPNet(torch.nn.Module):
    def __init__(self, obs_size, ase_latent_size, units, activation, initializer):
        super().__init__()

        input_size = obs_size + ase_latent_size
        print('build amp mlp net:', input_size)
        
        self._units = units
        self._initializer = initializer
        self._mlp = []

        in_size = input_size
        for i in range(len(units)):
            unit = units[i]
            curr_dense = torch.nn.Linear(in_size, unit)
            self._mlp.append(curr_dense)
            self._mlp.append(activation)
            in_size = unit

        self._mlp = nn.Sequential(*self._mlp)
        self.init_params()
        return

    def forward(self, obs, latent, skip_style):
        inputs = [obs, latent]
        input = torch.cat(inputs, dim=-1)
        output = self._mlp(input)
        return output

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                self._initializer(m.weight)
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)
        return

    def get_out_size(self):
        out_size = self._units[-1]
        return out_size

class AMPStyleCatNet1(torch.nn.Module):
    def __init__(self, obs_size, ase_latent_size, units, activation,
                 style_units, style_dim, initializer):
        super().__init__()

        print('build amp style cat net:', obs_size, ase_latent_size)
            
        self._activation = activation
        self._initializer = initializer
        self._dense_layers = []
        self._units = units
        self._style_dim = style_dim
        self._style_activation = torch.tanh

        self._style_mlp = self._build_style_mlp(style_units, ase_latent_size)
        self._style_dense = torch.nn.Linear(style_units[-1], style_dim)

        in_size = obs_size + style_dim
        for i in range(len(units)):
            unit = units[i]
            out_size = unit
            curr_dense = torch.nn.Linear(in_size, out_size)
            self._dense_layers.append(curr_dense)
            
            in_size = out_size

        self._dense_layers = nn.ModuleList(self._dense_layers)

        self.init_params()

        return

    def forward(self, obs, latent, skip_style):
        if (skip_style):
            style = latent
        else:
            style = self.eval_style(latent)

        h = torch.cat([obs, style], dim=-1)

        for i in range(len(self._dense_layers)):
            curr_dense = self._dense_layers[i]
            h = curr_dense(h)
            h = self._activation(h)

        return h

    def eval_style(self, latent):
        style_h = self._style_mlp(latent)
        style = self._style_dense(style_h)
        style = self._style_activation(style)
        return style

    def init_params(self):
        scale_init_range = 1.0

        for m in self.modules():
            if isinstance(m, nn.Linear):
                self._initializer(m.weight)
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)

        nn.init.uniform_(self._style_dense.weight, -scale_init_range, scale_init_range)
        return

    def get_out_size(self):
        out_size = self._units[-1]
        return out_size

    def _build_style_mlp(self, style_units, input_size):
        in_size = input_size
        layers = []
        for unit in style_units:
            layers.append(torch.nn.Linear(in_size, unit))
            layers.append(self._activation)
            in_size = unit

        enc_mlp = nn.Sequential(*layers)
        return enc_mlp

class CorrectionNet(torch.nn.Module):
    def __init__(self, input_dim, activation, initializer, shape_parameter):

        super().__init__()
        in_size = input_dim + 2
        #in_size = input_dim
        print('build correction net:', input_dim)
            
        self._activation = activation
        self._initializer = initializer
        self._mlp = []
        self.corr_units = [128, 128]
        self.shape_parameter = shape_parameter

        
        for unit in self.corr_units:
            curr_dense = torch.nn.Linear(in_size, unit)
            self._mlp.append(curr_dense)
            self._mlp.append(activation)
            in_size = unit

        self._mlp = nn.Sequential(*self._mlp)               

        self.init_params()

        return

    def forward(self, input, batch_env_ids):
        if batch_env_ids is not None:      #shape_parameter.shape[0] != batch_size:
            #shape_parameter = self.shape_parameter[batch_env_ids]
            shape_parameter = batch_env_ids
        else:
            shape_parameter = self.shape_parameter
        input_mlp = torch.cat([input, shape_parameter], dim=-1)
        output = self._mlp(input_mlp)
        return output


    def init_params(self):
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                self._initializer(m.weight)
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)
        return
    
    def get_out_size(self):
        out_size = self.corr_units[-1]
        return out_size
    
