import yaml
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import pandas as pd

path = "/Users/ninaschmid/SA/ASE/evaluation_uniform-trained/getup_evaluation_uniform-trained/"

files = os.listdir(path)

data = np.empty((0,4))

for file in files:
    if os.path.isfile(path+file) and file.endswith('.yaml'):
        with open(path + file, 'r') as f:
            data_eval = yaml.load(f, Loader=yaml.FullLoader)
            character = data_eval["character"]
            character = character.replace("arm_leg_parametrization/amp_humanoid_sword_shield_arm_","").replace(".xml", "")
            point = character.replace("-", ".").split("_leg_")
            x = float(point[0])
            y = float(point[1])
            t = data_eval["evaluation"]["average_completion_time"]
            rate = data_eval["evaluation"]["success_rate"]
            data = np.append(data, np.array([x,y,t, rate]).reshape((-1,4)), axis=0)
  
# data points
df = pd.DataFrame(data,columns=['arm_scale','leg_scale','t', 'rate'])
df_2D = df.pivot(index='leg_scale', columns='arm_scale', values=['rate', 't'])
data_rate = df_2D.rate.sort_values(by='leg_scale', ascending=False)
data_time = df_2D.t.sort_values(by='leg_scale', ascending=False)
#data_time = df_time.pivot(index='leg_scale', columns='arm_scale', values='t')


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.0,7.5))
sb.heatmap(data_rate, ax=ax1, annot=True, fmt=".2f", square=True, cbar=True, linewidths=0.1)
sb.heatmap(data_time, ax=ax2, annot=True, fmt=".2f", square=True, cbar=True, linewidths=0.1)
ax1.xaxis.tick_bottom()
ax2.xaxis.tick_bottom()
ax1.yaxis.tick_left()
ax2.yaxis.tick_left()
ax1.set(xlabel="Scaling Factor Arms", ylabel="Scaling Factor Legs")
ax2.set(xlabel="Scaling Factor Arms", ylabel="Scaling Factor Legs")
ax1.set_title("Success Rate", fontweight='bold')
ax2.set_title("Recovery Time ", fontweight='bold')

plt.show()