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

for factor_arm in factors_arm:
    for factor_leg in factors_leg:
        print('building model with params: ' + str(factor_arm) + "," + str(factor_leg))
        # rename mujoco model
        value = 'humanoid_' + str(factor_arm) + '_' + str(factor_leg)
        root.set('model', value)
        trans_frames = {}
        # modify arms
        for i_arm in bodies_arms:
            body_arm = root.find('.//body[@name="%s"]' %i_arm)
            body_arm_str = body_arm.get('name')
            print(body_arm_str)
            # if parent has benn changed, adjust position of body frame
            parent = root.find('.//body[@name="%s"]...'  %i_arm).get('name')
            if (parent in bodies_arms) or (parent in bodies_legs):
                # change position according to geom
                #print('here change pos due to parent')
                frame_pos_str = body_arm.get('pos')
                frame_pos_list = frame_pos_str.split(" ")
                trans_frame = trans_frames[parent]
                print(trans_frames)
                frame_pos_list[2] = float(frame_pos_list[2])-trans_frame
                frame_pos_list = [str(x) for x in frame_pos_list]
                frame_pos_str = ' '.join(frame_pos_list)
                print(type(frame_pos_str))
                body_arm.set('pos', frame_pos_str)
            if body_arm_str == 'sword' or body_arm_str == 'shield':
                continue
            for j,tag_arm in np.ndenumerate(tags):
                modify_tag = body_arm.find(tag_arm)
                #print(modify_tag.get('name'))
                #print(tag_arm)
                for attr in modify_tag.attrib:
                    if modify_tag.get('type')=='sphere':
                        if attr=='size' and tag_arm=='geom':
                            # change geom size attrib; scale size (radius if sphere) by factor
                            #print('here geom size')
                            size = float(modify_tag.get('size'))
                            vol_old = np.pi*pow(size,2)    #inital volume used later
                            delta_length = size*factor_arm-size
                            size *= factor_arm
                            modify_tag.set('size', str(size))
                            trans_frames[body_arm_str] = 2*delta_length   #translation of body part
                            vol_new = np.pi*pow(size,2)    #new volume used later
                    else:
                        if attr=='fromto':
                            # change geom formto attrib; scale fromto by scale factor, and redefine end (start - scaled length)
                            #print('here geom fromto')
                            value = modify_tag.get('fromto')
                            value_list = value.split(" ")
                            start = float(value_list[2])
                            end = float(value_list[5])
                            dist = abs(start-end)
                            rad = float(modify_tag.get('size'))
                            vol_old = (dist-2*rad)*2*np.pi*rad + 4/3.*np.pi*pow(rad,3)
                            dist *=factor_arm
                            end_new = start - dist
                            value_list[5] = end_new
                            value_list = [str(x) for x in value_list]
                            new_value = ' '.join(value_list)
                            modify_tag.set('fromto', new_value)
                            trans_frames[body_arm_str] = dist/2.   #translation of body part
                            vol_new = (dist-2*rad)*2*np.pi*rad + 4/3.*np.pi*pow(rad,3)
                        elif attr=='pos':
                            # change site pos attrib
                            #print('here site pos')
                            size_site = modify_tag.get('size')
                            size_site_list = size_site.split(" ")
                            if len(size_site_list) == 1:
                                size_site_f = float(size_site_list[0])
                                old_length = size_site_f*2
                                size_site_f *= factor_arm   # if sphere scale size (radius) by factor
                                new_length = size_site_f*2
                                size_site = size_site_f
                            else:
                                half_height = float(size_site_list[1])*factor_arm # only scale half-length of capsule (second value), leave radius (first value; := size val in geom) as for geom unchanged
                                old_length = 2*float(size_site_list[1])
                                new_length = 2*half_height
                            # shift pos of size prop to delta length while keeping position relative to geom constant
                            pos_site = modify_tag.get('pos')
                            pos_site_list = pos_site.split(" ")
                            ratio = abs(float(pos_site_list[2])-start)/old_length
                            pos_site_new = start - ratio*new_length
                            ratio_new = abs(pos_site_new-start)/new_length
                            print("ration before: " + str(ratio))
                            print("ration after: " + str(ratio_new))
                            pos_site_list[2] = pos_site_new
                            pos_site_list = [str(x) for x in pos_site_list]
                            pos_site = ' '.join(pos_site_list)
                            modify_tag.set('pos', pos_site)
                    if attr=='density':
                        # change geom density attrib; change mass prop to volume increase, recalculate neccessary density
                        #print('here geom density')
                        density = float(modify_tag.get('density'))
                        mass = density*vol_old
                        mass += (vol_new-vol_old)/vol_old*mass
                        density = mass/vol_new
                        modify_tag.set('density', str(density))
                    elif attr=='size' and tag_arm=='site':
                        # change site size attrib
                        #print('here site size')
                        if len(size_site_list) != 1:
                            size_site_list[1] = half_height
                            size_site_list = [str(x) for x in size_site_list]
                            size_site = ' '.join(size_site_list)
                        modify_tag.set('size', size_site)
        # modify legs
        for i_leg in bodies_legs:
            body_leg = root.find('.//body[@name="%s"]' %i_leg)
            body_leg_str = body_leg.get('name')
            print(body_leg_str)
            # if parent has benn changed, adjust position of body frame
            parent = root.find('.//body[@name="%s"]...'  %i_leg).get('name')
            if (parent in bodies_arms) or (parent in bodies_legs):
                # change position according to geom
                #print('here change pos due to parent')
                frame_pos_str = body_leg.get('pos')
                frame_pos_list = frame_pos_str.split(" ")
                trans_frame = trans_frames[parent]
                frame_pos_list[2] = float(frame_pos_list[2])-trans_frame
                frame_pos_list = [str(x) for x in frame_pos_list]
                frame_pos_str = ' '.join(frame_pos_list)
                body_leg.set('pos', frame_pos_str)
            for k,tag_leg in np.ndenumerate(tags):
                tag_leg_last = tag_leg + '[last()]'
                modify_tag = body_leg.find(tag_leg_last)
                #print(modify_tag.get('name'))
                #print(tag_leg)
                for attr in modify_tag.attrib:
                    if modify_tag.get('type')=='box':
                        if attr=='size':
                            # change geom and site size attrib; scale box in all dir with factor
                            #print('here geom and site size leg')
                            size = modify_tag.get('size')
                            size_list = size.split(" ")
                            size_list = [float(x) for x in size_list]
                            vol_old = np.prod(size_list)*8    #inital volume used later
                            size_list = [x*factor_leg for x in size_list]
                            vol_new = np.prod(size_list)*8    #new volume used later
                            size_list = [str(x) for x in size_list]
                            modify_tag.set('size', size)
                            trans_frames[body_leg_str] = None   #translation of body part
                        #elif attr=='pos':
                            ##TODO change geom and site pos attrib
                            #print('here geom and site pos leg')
                    else:
                        if attr=='fromto':
                            # change geom formto attrib; scale fromto by scale factor, and redefine end (start - scaled length)
                            #print('here geom fromto leg')
                            value = modify_tag.get('fromto')
                            value_list = value.split(" ")
                            start = float(value_list[2])
                            end = float(value_list[5])
                            dist = abs(start-end)
                            rad = float(modify_tag.get('size'))
                            vol_old = (dist-2*rad)*2*np.pi*rad + 4/3.*np.pi*pow(rad,3)
                            dist *=factor_leg
                            end_new = start - dist
                            value_list[5] = end_new
                            value_list = [str(x) for x in value_list]
                            new_value = ' '.join(value_list)
                            modify_tag.set('fromto', new_value)
                            trans_frames[body_leg_str] = dist/2.   #translation of body part
                            vol_new = (dist-2*rad)*2*np.pi*rad + 4/3.*np.pi*pow(rad,3)
                        elif attr=='pos':
                            # change site pos attrib
                            #print('here site pos leg')
                            size_site = modify_tag.get('size')
                            size_site_list = size_site.split(" ")
                            half_height = float(size_site_list[1])*factor_leg # only scale half-length of capsule (second value), leave radius (first value; := size val in geom) as for geom unchanged
                            old_length = 2*float(size_site_list[1])
                            new_length = 2*half_height
                            # shift pos of size prop to delta length while keeping position relative to geom constant
                            pos_site = modify_tag.get('pos')
                            pos_site_list = pos_site.split(" ")
                            ratio = abs(float(pos_site_list[2])-start)/old_length
                            pos_site_new = start - ratio*new_length
                            ratio_new = abs(pos_site_new-start)/new_length
                            print("ration before leg: " + str(ratio))
                            print("ration after leg: " + str(ratio_new))
                            pos_site_list[2] = pos_site_new
                            pos_site_list = [str(x) for x in pos_site_list]
                            pos_site = ' '.join(pos_site_list)
                            modify_tag.set('pos', pos_site)
                        elif attr=='size' and tag_leg=='site':
                            # change site size attrib
                            #print('here site size leg')
                            size_site_list[1] = half_height
                            size_site_list = [str(x) for x in size_site_list]
                            size_site = ' '.join(size_site_list)
                            modify_tag.set('size', size_site)
                    if attr=='density':
                        # change geom density attrib; change mass prop to volume increase, recalculate neccessary density
                        #print('here geom density leg')
                        density = float(modify_tag.get('density'))
                        mass = density*vol_old
                        mass += (vol_new-vol_old)/vol_old*mass
                        density = mass/vol_new
                        modify_tag.set('density', str(density))
        # generate new mujoco file for each arm and leg modification
        path = 'data/assets/mjcf/arm_leg_parametrization/'
        if not os.path.exists(path):
            print(os.path.exists(path))
            os.makedirs(path)
            print(os.path.exists(path))
        output_file = 'amp_humanoid_sword_shield_arm_' + str(factor_arm).replace(".", "-") + '_leg_' + str(factor_leg).replace(".","-") + '.xml'
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