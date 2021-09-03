# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 12:38:16 2021

@author: Maria Royo
"""

''' General plots settings '''
small_plots_font_size = 16
plots_font_size = 18
titles_font_size = 20

''' Data file paths and lengths ''' 
EMT46_2_path = "./DATA/EMT46_2DLC_resnet_50_EMT16_EMT46Apr7shuffle1_200000_Adapted.csv"
EMT46_4_path = "./DATA/EMT46_4DLC_resnet_50_EMT16_EMT46Apr7shuffle1_200000_Adapted.csv"
EMT16_path = "./DATA/EMT16DLC_resnet_50_EMT16_EMT46Apr7shuffle1_200000_Adapted.csv"
length_EMT46_2 = 21125
length_EMT46_4 = 21175
length_EMT16 = 21075
EMT45_path = "./DATA/EMT45DLC_resnet_50_EMT45Feb23shuffle2_300000_Adapted.csv"
subject45_path = "./DATA/subject45DLC_resnet_50_EMT45_KineticMar10shuffle1_560000_Adapted.csv"
length_EMT45 = 21075
length_subject45 = 12579
EMT19_path = "./DATA/EMT19DLC_resnet_50_EMT19Jul7shuffle1_200000_Adapted.csv"
length_EMT19 = 21250
EMT28_path = "./DATA/EMT28DLC_resnet_50_EMT28Jul7shuffle1_120000_Adapted.csv"
length_EMT28 = 21100
EMT48_path = "./DATA/EMT48DLC_resnet_50_EMT48Jul16shuffle1_120000_Adapted.csv"
length_EMT48 = 21050
EMT17_path = "./DATA/EMT17DLC_resnet_50_EMT17Jul16shuffle1_60000_Adapted.csv"
length_EMT17 = 21200
EMT18_path = "./DATA/EMT18DLC_resnet_50_EMT18Jul16shuffle1_90000_Adapted.csv"
length_EMT18 = 21075
EMT47_path = "./DATA/EMT47DLC_resnet_50_EMT47Jul16shuffle1_120000_Adapted.csv"
length_EMT47 = 21100

''' DATA LABELS '''

pos_idx = {
    'nose_x':0, 'nose_y':1,
    'L_nipple_x':2,'L_nipple_y':3,'R_nipple_x':4,'R_nipple_y':5,
    'L_elbow_x':6,'L_elbow_y':7,'R_elbow_x':8,'R_elbow_y':9,
    'L_wrist_x':10,'L_wrist_y':11,'R_wrist_x':12,'R_wrist_y':13,
    'L_hip_x':14,'L_hip_y':15,'R_hip_x':16,'R_hip_y':17,
    'L_knee_x':18,'L_knee_y':19,'R_knee_x':20,'R_knee_y':21,
    'L_ankle_x':22,'L_ankle_y':23,'R_ankle_x':24,'R_ankle_y':25
}

angle_idx = {
    'L_elbow':0, 'R_elbow':1,
    'L_knee':2, 'R_knee':3, 
    'L_arm':4, 'R_arm':5,
    'L_leg':6, 'R_leg':7, 
    'L_upper_trunk':8, 'R_upper_trunk':9,
    'L_lower_trunk':10,'R_lower_trunk':11,
    'L_nose_shoulders':12,'R_nose_shoulders':13                                     
}
position_labels_ = ['nose_x', 'nose_y',
    'L_nipple_x','L_nipple_y','R_nipple_x','R_nipple_y',
    'L_elbow_x','L_elbow_y','R_elbow_x','R_elbow_y',
    'L_wrist_x','L_wrist_y','R_wrist_x','R_wrist_y',
    'L_hip_x','L_hip_y','R_hip_x','R_hip_y',
    'L_knee_x','L_knee_y','R_knee_x','R_knee_y',
    'L_ankle_x','L_ankle_y','R_ankle_x','R_ankle_y']

angle_labels_ = ['Left elbow','Right elbow','Left knee','Right knee',
            'Left arm','Right arm','Left leg','Right leg',
            'Left upper trunk','Right upper trunk','Left lower trunk','Right lower trunk',
            'Left shoulder with nose','Right shoulder with nose']

angle_labels_2 = ['L elbow','R elbow','L knee','R knee',
            'L arm','R arm','L leg','R leg',
            'L up-trunk','R up-trunk','L low-trunk','R low-trunk',
            'L head','R head']

angles_angular_velocity_labels = ['Left elbow','Right elbow','Left knee','Right knee',
            'Left arm','Right arm','Left leg','Right leg',
            'Left upper trunk','Right upper trunk','Left lower trunk','Right lower trunk',
            'Left shoulder with nose','Right shoulder with nose',
            'Left elbow vel.','Right elbow vel.','Left knee vel.','Right knee vel.',
            'Left arm vel.','Right arm vel.','Left leg vel.','Right leg vel.',
            'Left upper trunk vel.','Right upper trunk vel.','Left lower trunk vel.','Right lower trunk vel.',
            'Left shoulder with nose vel.','Right shoulder with nose vel.']

body_parts_labels_ = ['Nose','Left shoulder','Right shoulder',
              'Left elbow','Right elbow','Left wrist','Right wrist',
              'Left hip','Right hip','Left knee','Right knee','Left ankle','Right ankle']