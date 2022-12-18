import os
import yaml
import subprocess


path = "ase/data/assets/mjcf/arm_leg_parametrization/"
files = os.listdir(path)

for file in files:
    if(os.path.isfile(path + file)):
        #modify mujoco file
        with open("ase/data/cfg/humanoid_sword_shield_reach_eval_param.yaml", 'r') as f_in:
            cfg = yaml.load(f_in, Loader=yaml.FullLoader)
            cfg["env"]["asset"]["assetFileName"] = "mjcf/arm_leg_parametrization/" + file
        with open ("ase/data/cfg/humanoid_sword_shield_reach_eval_param.yaml", 'w') as f_out:
            yaml.dump(cfg, f_out, sort_keys=False)
        #run run.py with arguments
        os.system("python ase/run.py --test --task HumanoidReach --cfg_env ase/data/cfg/humanoid_sword_shield_reach_eval_param.yaml --cfg_train ase/data/cfg/train/rlg/hrl_humanoid.yaml --motion_file ase/data/motions/reallusion_sword_shield/RL_Avatar_Idle_Ready_Motion.npy --llc_checkpoint ase/data/models/ase_llc_reallusion_sword_shield.pth --checkpoint ase/data/models/ase_hlc_reach_reallusion_sword_shield.pth --headless")
        #p = subprocess.Popen("python ase/run.py --test --task HumanoidReach --cfg_env ase/data/cfg/humanoid_sword_shield_reach_eval_param.yaml --cfg_train ase/data/cfg/train/rlg/hrl_humanoid.yaml --motion_file ase/data/motions/reallusion_sword_shield/RL_Avatar_Idle_Ready_Motion.npy --llc_checkpoint ase/data/models/ase_llc_reallusion_sword_shield.pth --checkpoint ase/data/models/ase_hlc_reach_reallusion_sword_shield.pth --headless", stdout=subprocess.PIPE, shell=True)
        #p.wait()
    else:
        print("Unknown character config file: {s}".format(file))
        assert(False)