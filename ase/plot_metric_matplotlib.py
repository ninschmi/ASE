import yaml
import os
import matplotlib.pyplot as plt
import numpy as np

path = "/Users/ninaschmid/Downloads/getup_evaluation_uniform-trained/"

files = os.listdir(path)

complete_time = np.empty((0,4))

for file in files:
    if os.path.isfile(path+file) and file.endswith('.yaml'):
        with open(path + file, 'r') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            character = data["character"]
            character = character.replace("arm_leg_parametrization/amp_humanoid_sword_shield_arm_","").replace(".xml", "")
            point = character.replace("-", ".").split("_leg_")
            x = float(point[0])
            y = float(point[1])
            t = data["evaluation"]["average_completion_time"]
            rate = data["evaluation"]["success_rate"]
            complete_time = np.append(complete_time, np.array([x,y,t, rate]).reshape((-1,4)), axis=0)
  
# data points
x_pts = complete_time[:,0]
y_pts = complete_time[:,1]
t_pts = complete_time[:,2]
r_pts = complete_time[:,3]
  
t_img = t_pts.reshape(5, 7)
r_img = r_pts.reshape(5, 7)

fig, (ax1, ax2) = plt.subplots(1,2)
fig.suptitle('Evaluation for Arm Leg Parametrization')
ax1.set_title("Recovery Time")

sc = ax1.scatter(x_pts,y_pts, c=t_pts, cmap=plt.plasma())
ax1.set_ylabel('Scaling Factor Legs')
ax1.set_xlabel('Scaling Factor Arms')
ax1.xaxis.set_ticks(np.arange(0.5,1.75,0.25))
ax1.yaxis.set_ticks(np.arange(0.5,2.25,0.25))
cbar = fig.colorbar(sc)
cbar.set_label("Recovery Time")

ax2.set_title("Success Rate")
sc = ax2.scatter(x_pts,y_pts, c=r_pts, cmap=plt.plasma())
ax2.set_ylabel('Scaling Factor Legs')
ax2.set_xlabel('Scaling Factor Arms')
ax2.xaxis.set_ticks(np.arange(0.5,1.75,0.25))
ax2.yaxis.set_ticks(np.arange(0.5,2.25,0.25))
cbar = fig.colorbar(sc)
cbar.set_label("Success Rate")
plt.show()