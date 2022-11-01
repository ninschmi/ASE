import os
import numpy as np
import xml.etree.ElementTree as et

# base character xml file
filename = 'data/assets/mjcf/amp_humanoid_sword_shield.xml'

# generate modified characters according to different shape parameterizations

# shape parameterizations

# leg length vs. arm length shape parameterization
# factors: legs: up to 2.0 times longer and 2.0 times shorter; arms: up to 1.5 times longer and 2.0 times shorter 
factors_leg = np.array([0.5, 0.75, 1, 1.25, 1.5, 1.75, 2])
factors_arm = np.array([0.5, 0.75, 1, 1.25, 1.5])
bodies_arms = np.array(['right_upper_arm', 'right_lower_arm', 'right_hand', 'sword', 'left_upper_arm', 'left_lower_arm', 'left_hand', 'shield'])
bodies_legs = np.array(['right_thigh', 'right_shin', 'right_foot', 'left_thigh', 'left_shin', 'left_foot'])
tags = np.array(['geom'])

for factor_arm in factors_arm:
    for factor_leg in factors_leg:
        # read in base character for each scaling factor pair
        tree = et.parse(filename)
        root= tree.getroot()
        
        print('building model with params: ' + str(factor_arm) + "," + str(factor_leg))
        # rename mujoco model
        value = 'humanoid_' + str(factor_arm) + '_' + str(factor_leg)
        root.set('model', value)
        trans_frames = {}
        end_points = {}
        translation = {}

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
                frame_pos_list = [float(x) for x in frame_pos_list]
                #trans_frame = trans_frames[parent]
                # adjust gap btw parent geom and child frame according to scaling
                gap = [0,0,0]
                #only relevant for sword (only child of a hand body)
                print(type(end_points[parent]))
                if type(end_points[parent]) is not list:
                    #frame_pos_list[0] += end_points[parent]
                    print(frame_pos_list)
                else:
                    end_old = end_points[parent][0]
                    end_new = end_points[parent][1]
                    gain_child = end_points[parent][2]
                    print('child gain: ')
                    print(gain_child)
                    for i in range(3):
                        gap[i] = (frame_pos_list[i]-end_old[i])*factor_arm
                        #frame_pos_list[i] = end_new[i] + gap[i] + gain_child[i]
                        frame_pos_list[i] = end_new[i] + gap[i]
                #frame_pos_list[2] = float(frame_pos_list[2])-trans_frame
                frame_pos_list = [str(x) for x in frame_pos_list]
                frame_pos_str = ' '.join(frame_pos_list)
                body_arm.set('pos', frame_pos_str)
            if body_arm_str == 'sword' or body_arm_str == 'shield':
                continue

            for tag_arm in tags:
                modify_tag = body_arm.find(tag_arm) # find('geom')
                #print(modify_tag.get('name'))
                #print(tag_arm)

                # change geom size attrib; scale size (radius of sphere or capsule) by factor
                #print('here geom size')
                size = float(modify_tag.get('size'))
                if modify_tag.get('type')=='sphere':
                    vol_old = 4/3.*np.pi*pow(size,3)    #inital volume used later
                    size_old = size
                    size *= factor_arm 
                    delta_length = size - size_old #used for translating child frame
                    vol_new = 4/3.*np.pi*pow(size,3)    #new volume used later     
                    modify_tag.set('size', str(size))       
                    #trans_frames[body_arm_str] = delta_length   #translation of body part used for child frames         
                    #end_points[parent][2] = [0, 0, delta_length]
                    end_points[body_arm_str] = delta_length
                    
                else:
                    size_old = size
                    size *= factor_arm
                    modify_tag.set('size', str(size))
                    delta_size = size - size_old
                    #trans_frames[body_arm_str] = delta_length along x and y axis of capsule
                    # move body frame if child is attached to torso or pelvis
                    if parent == 'torso':
                        # change position of body frame according to radial change (scaling of size)
                        translation[body_arm_str] = [0, delta_size, 0]

                    # change geom formto attrib; scale fromto by scale factor, and redefine end (start - scaled length)
                    # change geom fromto attrib; scale gap along z btw geom and body and redefine start
                    #print('here geom fromto')
                    value = modify_tag.get('fromto')
                    value_list = value.split(" ")
                    value_list = [float(x) for x in value_list]
                    start = float(value_list[2])
                    end = float(value_list[5])
                    end_pt = value_list[3:]
                    dist = abs(start-end)
                    vol_old = (dist-2*size_old)*np.pi*pow(size_old,2) + 4/3.*np.pi*pow(size_old,3)
                    dist *=factor_arm
                    vol_new = (dist-2*size)*np.pi*pow(size,2) + 4/3.*np.pi*pow(size,3)
                    start_new = start*factor_arm
                    end_new = start_new - dist
                    value_list[2] = start_new
                    value_list[5] = end_new
                    end_new_pt = value_list[3:]
                    value_list = [str(x) for x in value_list]
                    new_value = ' '.join(value_list)
                    modify_tag.set('fromto', new_value)
                    trans_frames[body_arm_str] =  end - end_new #translation of body part used for child frames
                    end_points[body_arm_str] = [end_pt, end_new_pt, [0,0,0]]
                
                # change geom density attrib; change mass prop to volume increase, recalculate neccessary density
                #print('here geom density')
                density = float(modify_tag.get('density'))
                mass = density*vol_old
                mass += (vol_new-vol_old)/vol_old*mass
                density = mass/vol_new
                modify_tag.set('density', str(density))

            if parent == 'torso':
                # change position of body frame according to radial change (scaling of size)
                bframe_str = body_arm.get('pos')
                bframe_list = bframe_str.split(" ")
                if 'right' in body_arm_str:
                    dir = 1
                else:
                    dir = -1
                bframe_list = [str(float(x)+translation[body_arm_str][i]*dir) for i,x in enumerate(bframe_list)]
                bframe_str = ' '.join(bframe_list)
                body_arm.set('pos', bframe_str)

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
                frame_pos_list = [float(x) for x in frame_pos_list]
                #trans_frame = trans_frames[parent]
                # adjust gap btw parent geom and child frame according to scaling
                gap = [0,0,0]
                print(type(end_points[parent]))
                if type(end_points[parent]) is not list:
                    #frame_pos_list[0] += end_points[parent]
                    print(frame_pos_list)
                else:
                    end_old = end_points[parent][0]
                    end_new = end_points[parent][1]
                    gain_child = end_points[parent][2]
                    print('child gain: ')
                    print(gain_child)
                    for i in range(3):
                        gap[i] = (frame_pos_list[i]-end_old[i])*factor_leg
                        #frame_pos_list[i] = end_new[i] + gap[i] + gain_child[i]
                        frame_pos_list[i] = end_new[i] + gap[i]
                #frame_pos_list[2] = float(frame_pos_list[2])-trans_frame
                frame_pos_list = [str(x) for x in frame_pos_list]
                frame_pos_str = ' '.join(frame_pos_list)
                body_leg.set('pos', frame_pos_str)

            for tag_leg in tags:
                tag_leg_last = tag_leg + '[last()]'
                modify_tag = body_leg.find(tag_leg_last)    #find('geom[last()]')
                #print(modify_tag.get('name'))
                #print(tag_leg)

                # change geom size attrib; scale size (box in all dir or radius of capsule) by factor
                #print('here geom size leg')
                if modify_tag.get('type')=='box':
                    size = modify_tag.get('size')
                    size_list = size.split(" ")
                    size_list = [float(x) for x in size_list]
                    vol_old = np.prod(size_list)*8    #inital volume used later
                    size_list = [x*factor_leg for x in size_list]
                    vol_new = np.prod(size_list)*8    #new volume used later
                    size_list = [str(x) for x in size_list]
                    size = ' '.join(size_list)
                    modify_tag.set('size', size)

                    pos = modify_tag.get('pos')
                    pos_list = pos.split(" ")
                    pos_list = [str(float(x)*factor_leg) for x in pos_list]
                    pos = ' '.join(pos_list)
                    modify_tag.set('pos', pos)

                    #delta_length in x, y, z #used for translating child frame
                    #trans_frames[body_leg_str] = None    #translation of body part used for child frames (delta x,y,z)
                    end_points[body_leg_str] = [[0,0,0], [0,0,0]]
                else:
                    size = float(modify_tag.get('size'))
                    size_old = size
                    size *= factor_leg
                    delta_size = size - size_old
                    modify_tag.set('size', str(size))
                    #trans_frames[body_arm_str] = delta_length along x and y axis of capsule
                    # move body frame if child is attached to torso or pelvis
                    if parent == 'pelvis':
                        # change position of body frame according to radial change (scaling of size)
                        translation[body_leg_str] = [0, delta_size, 0]

                    # change geom formto attrib; scale fromto by scale factor, and redefine end (start - scaled length)
                    #print('here geom fromto leg')
                    value = modify_tag.get('fromto')
                    value_list = value.split(" ")
                    value_list = [float(x) for x in value_list]
                    start = float(value_list[2])
                    end = float(value_list[5])
                    end_pt = value_list[3:]
                    dist = abs(start-end)
                    vol_old = (dist-2*size_old)*np.pi*pow(size_old,2) + 4/3.*np.pi*pow(size_old,3)
                    dist *=factor_leg
                    vol_new = (dist-2*size)*np.pi*pow(size,2) + 4/3.*np.pi*pow(size,3)
                    start_new = start*factor_leg
                    end_new = start_new - dist
                    value_list[2] = start_new
                    value_list[5] = end_new
                    end_new_pt = value_list[3:]
                    value_list = [str(x) for x in value_list]
                    new_value = ' '.join(value_list)
                    modify_tag.set('fromto', new_value)
                    trans_frames[body_leg_str] = end - end_new  #translation of body part used for child frames
                    end_points[body_leg_str] = [end_pt, end_new_pt, [0,0,0]]

                # change geom density attrib; change mass prop to volume increase, recalculate neccessary density
                #print('here geom density leg')
                density = float(modify_tag.get('density'))
                mass = density*vol_old
                mass += (vol_new-vol_old)/vol_old*mass
                density = mass/vol_new
                modify_tag.set('density', str(density))  

            if parent == 'pelvis':
                # change position of body frame according to radial change (scaling of size)
                bframe_str = body_leg.get('pos')
                bframe_list = bframe_str.split(" ")
                if 'right' in body_leg_str:
                    dir = -1
                else:
                    dir = 1
                bframe_list = [str(float(x)+translation[body_leg_str][i]*dir) for i,x in enumerate(bframe_list)]
                bframe_str = ' '.join(bframe_list)
                body_leg.set('pos', bframe_str)
                            
        #adjust pelvis wrt to change of legs
        z_pelvis_str = root.find('.//body[@name="pelvis"]').get('pos')
        z_pelvis_list = z_pelvis_str.split(" ")
        z_pelvis_list[2] = str(float(z_pelvis_list[2]) * factor_leg)
        z_pelvis_str = ' '.join(z_pelvis_list)
        root.find('.//body[@name="pelvis"]').set('pos', z_pelvis_str)

        # generate new mujoco file for each arm and leg modification
        path = 'data/assets/mjcf/arm_leg_parametrization/'
        if not os.path.exists(path):
            os.makedirs(path)
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