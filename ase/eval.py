import os
import yaml
import subprocess

# 0: evaluate HumanoidReach
# 1: evaluate HumanoidLocation
# 2: evaluate HumanoidGetUp

eval = 0

path = "ase/data/assets/mjcf/arm_leg_parametrization/"
files = os.listdir(path)

if eval == 0:
    cfg_fn = "ase/data/cfg/humanoid_sword_shield_reach_eval_param.yaml"
    # PRE-TRAIN
    #command = "python ase/run.py --test --task HumanoidReach --cfg_env ase/data/cfg/humanoid_sword_shield_reach_eval_param.yaml --cfg_train ase/data/cfg/train/rlg/hrl_humanoid.yaml --motion_file ase/data/motions/reallusion_sword_shield/RL_Avatar_Idle_Ready_Motion.npy --llc_checkpoint ase/data/models/ase_llc_reallusion_sword_shield.pth --checkpoint ase/data/models/ase_hlc_reach_reallusion_sword_shield.pth --headless"
    # SELF-TRAINED
    #command = "python ase/run.py --test --task HumanoidReach --cfg_env ase/data/cfg/humanoid_sword_shield_reach_eval_param.yaml --cfg_train ase/data/cfg/train/rlg/hrl_humanoid.yaml --motion_file ase/data/motions/reallusion_sword_shield/RL_Avatar_Idle_Ready_Motion.npy --llc_checkpoint ase/data/models/ase_llc_reallusion_sword_shield.pth --checkpoint output/Humanoid_15-15-25-42/nn/Humanoid.pth --headless"
    # UNIFORM
    command = "python ase/run.py --test --task HumanoidReach --cfg_env ase/data/cfg/humanoid_sword_shield_reach_eval_param.yaml --cfg_train ase/data/cfg/train/rlg/hrl_humanoid.yaml --motion_file ase/data/motions/reallusion_sword_shield/RL_Avatar_Idle_Ready_Motion.npy --llc_checkpoint ase/data/models/Humanoid_TF_Learning_DR_00229050.pth --checkpoint ase/data/models/Humanoid_TF_Learning_DR_Reach_00018550.pth --headless"
elif eval ==1 :
     cfg_fn = "ase/data/cfg/humanoid_sword_shield_location_eval_param.yaml"
     # PRE_TRAIN
     #command = "python ase/run.py --test --task HumanoidLocation --cfg_env ase/data/cfg/humanoid_sword_shield_location_eval_param.yaml --cfg_train ase/data/cfg/train/rlg/hrl_humanoid.yaml --motion_file ase/data/motions/reallusion_sword_shield/RL_Avatar_Idle_Ready_Motion.npy --llc_checkpoint ase/data/models/ase_llc_reallusion_sword_shield.pth --checkpoint ase/data/models/ase_hlc_location_reallusion_sword_shield.pth --headless"
     # UNIFORM
     command = "python ase/run.py --test --task HumanoidLocation --cfg_env ase/data/cfg/humanoid_sword_shield_location_eval_param.yaml --cfg_train ase/data/cfg/train/rlg/hrl_humanoid.yaml --motion_file ase/data/motions/reallusion_sword_shield/RL_Avatar_Idle_Ready_Motion.npy --llc_checkpoint ase/data/models/Humanoid_TF_Learning_DR_00229050.pth --checkpoint ase/data/models/Humanoid_TF_Learning_DR_Location_00017900.pth --headless"
else:
     cfg_fn = "ase/data/cfg/humanoid_ase_sword_shield_getup_eval_param.yaml"
     # PRE-TRAIN
     #command = "python ase/run.py --test --task HumanoidAMPGetup --cfg_env ase/data/cfg/humanoid_ase_sword_shield_getup_eval_param.yaml --cfg_train ase/data/cfg/train/rlg/ase_humanoid.yaml --motion_file ase/data/motions/reallusion_sword_shield/dataset_reallusion_sword_shield.yaml --checkpoint ase/data/models/ase_llc_reallusion_sword_shield.pth --headless"
     # UNIFORM
     #command = "python ase/run.py --test --task HumanoidAMPGetup --cfg_env ase/data/cfg/humanoid_ase_sword_shield_getup_eval_param.yaml --cfg_train ase/data/cfg/train/rlg/ase_humanoid.yaml --motion_file ase/data/motions/reallusion_sword_shield/dataset_reallusion_sword_shield.yaml --checkpoint ase/data/models/Humanoid_TF_Learning_DR_00229050.pth --headless"
     # ADAPTIVE SAMPLING
     command = "python ase/run.py --test --task HumanoidAMPGetup --cfg_env ase/data/cfg/humanoid_ase_sword_shield_getup_eval_param.yaml --cfg_train ase/data/cfg/train/rlg/ase_humanoid.yaml --motion_file ase/data/motions/reallusion_sword_shield/dataset_reallusion_sword_shield.yaml --checkpoint ase/data/models/Humanoid_TF_Learning_DR_ADAPTIVE_00229050.pth --headless"

for file in files:
    if(os.path.isfile(path + file)):
        #modify mujoco file
        with open(cfg_fn, 'r') as f_in:
            cfg = yaml.load(f_in, Loader=yaml.FullLoader)
            cfg["env"]["asset"]["assetFileName"] = "mjcf/arm_leg_parametrization/" + file
        with open (cfg_fn, 'w') as f_out:
            yaml.dump(cfg, f_out, sort_keys=False)
        #run run.py with arguments
        #os.system("python ase/run.py --test --task HumanoidReach --cfg_env ase/data/cfg/humanoid_sword_shield_reach_eval_param.yaml --cfg_train ase/data/cfg/train/rlg/hrl_humanoid.yaml --motion_file ase/data/motions/reallusion_sword_shield/RL_Avatar_Idle_Ready_Motion.npy --llc_checkpoint ase/data/models/ase_llc_reallusion_sword_shield.pth --checkpoint ase/data/models/ase_hlc_reach_reallusion_sword_shield.pth --headless")
        os.system(command)
        #p = subprocess.Popen("python ase/run.py --test --task HumanoidReach --cfg_env ase/data/cfg/humanoid_sword_shield_reach_eval_param.yaml --cfg_train ase/data/cfg/train/rlg/hrl_humanoid.yaml --motion_file ase/data/motions/reallusion_sword_shield/RL_Avatar_Idle_Ready_Motion.npy --llc_checkpoint ase/data/models/ase_llc_reallusion_sword_shield.pth --checkpoint ase/data/models/ase_hlc_reach_reallusion_sword_shield.pth --headless", stdout=subprocess.PIPE, shell=True)
        #p.wait()
    else:
        print("Unknown character config file: {s}".format(file))
        assert(False)