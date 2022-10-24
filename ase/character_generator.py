import os
import numpy as np
import xml.etree.ElementTree as et

# read in base character xml file
filename = 'data/assets/mjcf/amp_humanoid_sword_shield.xml'
tree = et.parse(filename)
root= tree.getroot()

# generate modified characters according to different shape parameterizations

# shape parameterizations

# leg length vs. arm length shape parameterization
# factors: legs: up to 2.0 times longer and 2.0 times shorter; arms: up to 1.5 times longer and 2.0 times shorter 
factors_leg = np.array([0.5, 0.75, 1, 1.25, 1.5, 1.75, 2])
factors_arm = np.array([0.5, 0.75, 1, 1.25, 1.5])
bodies_arms = np.array(['right_upper_arm', 'right_lower_arm', 'right_hand', 'sword', 'left_upper_arm', 'left_lower_arm', 'shield'])
bodies_legs = np.array(['right_thigh', 'right_shin', 'right_foot'])
tags = np.array(['geom', 'site'])
attributes = [[['fromto', 'size'], 'density'],['pos', 'size']] #change to dir?

for factor_arm in factors_arm:
    for factor_leg in factors_leg:
        print('building model with params: ' + str(factor_arm) + "," + str(factor_leg))
        # rename mujoco model
        value = 'humanoid_' + str(factor_arm) + '_' + str(factor_leg)
        root.set('model', value)
        # modify arms
        for i_arm in bodies_arms:
            body_arm = root.find('.//body[@name="%s"]' %i_arm)
            body_arm_str = body_arm.get('name')
            print(body_arm_str)
            # if parent has benn changed, adjust position of body frame
            parent = root.find('.//body[@name="%s"]...'  %i_arm).get('name')
            if (parent in bodies_arms) or (parent in bodies_legs):
                ##TODO change position
                print('here change pos due to parent')
            if body_arm_str == 'sword' or body_arm_str == 'shield':
                print('here continue')
                continue
            for j,tag_arm in np.ndenumerate(tags):
                modify_tag = body_arm.find(tag_arm)
                print(modify_tag.get('name'))
                print(tag_arm)
                for attr in modify_tag.attrib:
                #for attr in attributes[j[0]]:
                    if modify_tag.get('type')=='sphere':
                        if attr=='size' and tag_arm=='geom':
                            ##TODO change geom size attrib
                            print('here geom size')
                            # Element.set(‘attrname’, ‘value’) – Modifying element attributes. 
                    else:
                        if attr=='fromto':
                            ##TODO change geom formto attrib
                            print('here geom fromto')
                        elif attr=='pos':
                            ##TODO change site pos attrib
                            print('here site pos')
                    if attr=='density':
                        ##TODO change geom density attrib
                        print('here geom density')
                    elif attr=='size' and tag_arm=='site':
                        ##TODO change site size attrib
                        print('here site size')
        # modiy legs
        for i_leg in bodies_legs:
            body_leg = root.find('.//body[@name="%s"]' %i_leg)
            body_leg_str = body_leg.get('name')
            print(body_leg_str)
            # if parent has benn changed, adjust position of body frame
            parent = root.find('.//body[@name="%s"]...'  %i_leg).get('name')
            if (parent in bodies_arms) or (parent in bodies_legs):
                ##TODO change position
                print('here change pos due to parent')
            for k,tag_leg in np.ndenumerate(tags):
                tag_leg_last = tag_leg + '[last()]'
                modify_tag = body_leg.find(tag_leg_last)
                print(modify_tag.get('name'))
                print(tag_leg)
                #for attr in attributes[j[0]]:
                for attr in modify_tag.attrib:
                    if modify_tag.get('type')=='box':
                        if attr=='size':
                            ##TODO change geom and site size attrib
                            print('here geom and site size leg')
                        elif attr=='pos':
                            ##TODO change geom and site pos attrib
                            print('here geom and site pos leg')
                    else:
                        if attr=='fromto':
                            ##TODO change geom formto attrib
                            print('here geom fromto leg')
                        elif attr=='pos':
                            ##TODO change site pos attrib
                            print('here site pos leg')
                        elif attr=='size' and tag_leg=='site':
                            ##TODO change site size attrib
                            print('here site size leg')
                    if attr=='density':
                        ##TODO change geom density attrib
                        print('here geom density leg')
        # generate new mujoco file for each arm and leg modification
        path = 'data/assets/mjcf/arm_leg_parametrization/'
        if not os.path.exists(path):
            print(os.path.exists(path))
            os.makedirs(path)
            print(os.path.exists(path))
        output_file = 'arm_' + str(factor_arm) + '_leg_' + str(factor_leg) + '.xml'
        output = path + output_file
        # save all files in new directory
        with open(output, "wb") as f:
            tree.write(f)



# height vs. body thickness shape parameterization
# factors: height: from 0.75 to 1.5 times; thickness from 0.5 to 1.5 times

# mass vs volume shape parameterization
# ??

# leg length (left) vs. leg length (right) shape parameterization
# factors: both legs lengths up to 10% independently

# upper body thickness vs. lower body thickness shape parametrization
# factos: upper and lower bodies thicker from 0.5 to 1.5 times