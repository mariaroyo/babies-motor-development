# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 12:38:16 2021

@author: Maria Royo
"""

import numpy as np
import csv
import math
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import welch
import datetime
import matplotlib.dates as mdates
from settings_config import plots_font_size,titles_font_size,pos_idx,\
    angle_idx,body_parts_labels_


def angle(p1,p2,p3):
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)
    ba = a - b
    bc = c - b
    atanA = math.atan2(ba[1],ba[0])
    atanB = math.atan2(bc[1],bc[0])
    angle = atanB-atanA # Radians
    if angle > np.pi:
        angle = - (2*np.pi - angle)
    if angle < -np.pi:
        angle = 2*np.pi + angle
    return abs(angle)

def smooth(a,WSZ):
    # a: NumPy 1-D array containing the data to be smoothed
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    out0 = np.convolve(a,np.ones(WSZ,dtype=int),'valid')/WSZ    
    r = np.arange(1,WSZ-1,2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((  start , out0, stop  ))

def delete_nans(data):
    n, m = data.shape
    delete_cols = []
    for time_step in range(m):
        if True in np.isnan(data[:,time_step]):
            delete_cols.append(time_step)
    data_ = np.delete(data, delete_cols, 1) # Delete data at timesteps where there are nans.
    print('There are {} frames with high likelihood'.format(data_.shape[1]))
    return data_

def remove_outliers(data):
    for i,time_series in enumerate(data):
        data_points_before = np.count_nonzero(~np.isnan(time_series))
        length = len(time_series)
        min_limit1 = np.nanpercentile(time_series,1)
        max_limit1 = np.nanpercentile(time_series,99)
        min_limit2 = np.nanpercentile(time_series,5)
        max_limit2 = np.nanpercentile(time_series,95)
        for j,sample in enumerate(time_series):
            if j > 0:
                if abs(time_series[j-1]-sample)>100:
                    data[i,j] = np.NaN
            if j < length-1:
                if abs(time_series[j+1]-sample)>100:
                    data[i,j] = np.NaN
            if sample > max_limit1+100 or sample<min_limit1-100\
                or sample > max_limit2+300 or sample<min_limit2-300:
                data[i,j] = np.NaN
        data_points_after = np.count_nonzero(~np.isnan(time_series))
        #print('Deleted ouliers: {}'.format(data_points_before-data_points_after))

def get_likelihood_positions_angles_nans(data_path,video_len,likelihood_threshold=0.995,labels_format=None):
    '''
    Returns the likelihood of each body feature, the position of the body feature and the angle created between body features.
    The position and angle correspond to NaNs if the likelihood is below a threshold
    '''
    likelihood=np.zeros((int(len(pos_idx)/2),video_len))
    X_pos_=np.zeros((len(pos_idx),video_len))
    X_angles_ = np.zeros((len(angle_idx),video_len))

    i=-1

    shoulder = 'shoulder'
    hand = 'wrist'
    foot = 'ankle'
    if labels_format == '45':
        hand = 'hand'
        foot = 'foot'
    if labels_format == '46/16':
        shoulder = 'nipple' 
    with open(data_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            i+=1
            likelihood[:,i]=[row['nose_l'],
                             row['L_{}_l'.format(shoulder)],row['R_{}_l'.format(shoulder)],
                             row['L_elbow_l'],row['R_elbow_l'],
                             row['L_{}_l'.format(hand)],row['R_{}_l'.format(hand)],
                             row['L_hip_l'],row['R_hip_l'],
                             row['L_knee_l'],row['R_knee_l'],
                             row['L_{}_l'.format(foot)],row['R_{}_l'.format(foot)]]
                             #row['L_eye_l'],row['R_eye_l']
                             #row['L_ear_l'],row['R_ear_l'],]


            X_pos_[:,i] = [row['nose_x'],row['nose_y'],
                           row['L_{}_x'.format(shoulder)],row['L_{}_y'.format(shoulder)],row['R_{}_x'.format(shoulder)],row['R_{}_y'.format(shoulder)],
                           row['L_elbow_x'],row['L_elbow_y'],row['R_elbow_x'],row['R_elbow_y'],
                           row['L_{}_x'.format(hand)],row['L_{}_y'.format(hand)],row['R_{}_x'.format(hand)],row['R_{}_y'.format(hand)],
                           row['L_hip_x'],row['L_hip_y'],row['R_hip_x'],row['R_hip_y'],
                           row['L_knee_x'],row['L_knee_y'],row['R_knee_x'],row['R_knee_y'],
                           row['L_{}_x'.format(foot)],row['L_{}_y'.format(foot)],row['R_{}_x'.format(foot)],row['R_{}_y'.format(foot)]]
                           #row['L_eye_x'],row['L_eye_y'],row['R_eye_x'],row['R_eye_y'],
                           #row['L_ear_x'],row['L_ear_y'],row['R_ear_x'],row['R_ear_y']]

    low_likelihood = likelihood<likelihood_threshold
    n_,m_=low_likelihood.shape
    for i in range(n_):
        for j in range(m_):
            if low_likelihood[i,j]:
                X_pos_[2*i  ,j] = np.NaN
                X_pos_[2*i+1,j] = np.NaN
    remove_outliers(X_pos_)
    n_,m_=X_pos_.shape
    for j in range(m_):
        X_angles_[angle_idx['L_elbow'],j] = angle([X_pos_[pos_idx['L_wrist_x'],j],X_pos_[pos_idx['L_wrist_y'],j]],
                                                  [X_pos_[pos_idx['L_elbow_x'],j],X_pos_[pos_idx['L_elbow_y'],j]],
                                                  [X_pos_[pos_idx['L_nipple_x'],j],X_pos_[pos_idx['L_nipple_y'],j]]); 
        X_angles_[angle_idx['R_elbow'],j] = angle([X_pos_[pos_idx['R_wrist_x'],j],X_pos_[pos_idx['R_wrist_y'],j]],
                                                  [X_pos_[pos_idx['R_elbow_x'],j],X_pos_[pos_idx['R_elbow_y'],j]],
                                                  [X_pos_[pos_idx['R_nipple_x'],j],X_pos_[pos_idx['R_nipple_y'],j]]); 
        X_angles_[angle_idx['L_knee'],j] = angle([X_pos_[pos_idx['L_ankle_x'],j],X_pos_[pos_idx['L_ankle_y'],j]],
                                                  [X_pos_[pos_idx['L_knee_x'],j],X_pos_[pos_idx['L_knee_y'],j]],
                                                  [X_pos_[pos_idx['L_hip_x'],j],X_pos_[pos_idx['L_hip_y'],j]]); 
        X_angles_[angle_idx['R_knee'],j] = angle([X_pos_[pos_idx['R_ankle_x'],j],X_pos_[pos_idx['R_ankle_y'],j]],
                                                  [X_pos_[pos_idx['R_knee_x'],j],X_pos_[pos_idx['R_knee_y'],j]],
                                                  [X_pos_[pos_idx['R_hip_x'],j],X_pos_[pos_idx['R_hip_y'],j]]); 
        X_angles_[angle_idx['L_arm'],j] = angle([X_pos_[pos_idx['L_elbow_x'],j],X_pos_[pos_idx['L_elbow_y'],j]],
                                                  [X_pos_[pos_idx['L_nipple_x'],j],X_pos_[pos_idx['L_nipple_y'],j]],
                                                  [X_pos_[pos_idx['L_hip_x'],j],X_pos_[pos_idx['L_hip_y'],j]]);  
        X_angles_[angle_idx['R_arm'],j] = angle([X_pos_[pos_idx['R_elbow_x'],j],X_pos_[pos_idx['R_elbow_y'],j]],
                                                  [X_pos_[pos_idx['R_nipple_x'],j],X_pos_[pos_idx['R_nipple_y'],j]],
                                                  [X_pos_[pos_idx['R_hip_x'],j],X_pos_[pos_idx['R_hip_y'],j]]);
        X_angles_[angle_idx['L_leg'],j] = angle([X_pos_[pos_idx['L_knee_x'],j],X_pos_[pos_idx['L_knee_y'],j]],
                                                  [X_pos_[pos_idx['L_hip_x'],j],X_pos_[pos_idx['L_hip_y'],j]],
                                                  [X_pos_[pos_idx['L_nipple_x'],j],X_pos_[pos_idx['L_nipple_y'],j]]); 
        X_angles_[angle_idx['R_leg'],j] = angle([X_pos_[pos_idx['R_knee_x'],j],X_pos_[pos_idx['R_knee_y'],j]],
                                                  [X_pos_[pos_idx['R_hip_x'],j],X_pos_[pos_idx['R_hip_y'],j]],
                                                  [X_pos_[pos_idx['R_nipple_x'],j],X_pos_[pos_idx['R_nipple_y'],j]]);
        X_angles_[angle_idx['L_upper_trunk'],j] = angle([X_pos_[pos_idx['L_hip_x'],j],X_pos_[pos_idx['L_hip_y'],j]],
                                                  [X_pos_[pos_idx['L_nipple_x'],j],X_pos_[pos_idx['L_nipple_y'],j]],
                                                  [X_pos_[pos_idx['R_nipple_x'],j],X_pos_[pos_idx['R_nipple_y'],j]]);
        X_angles_[angle_idx['R_upper_trunk'],j] = angle([X_pos_[pos_idx['R_hip_x'],j],X_pos_[pos_idx['R_hip_y'],j]],
                                                  [X_pos_[pos_idx['R_nipple_x'],j],X_pos_[pos_idx['R_nipple_y'],j]],
                                                  [X_pos_[pos_idx['L_nipple_x'],j],X_pos_[pos_idx['L_nipple_y'],j]]);
        X_angles_[angle_idx['L_lower_trunk'],j] = angle([X_pos_[pos_idx['L_nipple_x'],j],X_pos_[pos_idx['L_nipple_y'],j]],
                                                  [X_pos_[pos_idx['L_hip_x'],j],X_pos_[pos_idx['L_hip_y'],j]],
                                                  [X_pos_[pos_idx['R_hip_x'],j],X_pos_[pos_idx['R_hip_y'],j]]); 
        X_angles_[angle_idx['R_lower_trunk'],j] = angle([X_pos_[pos_idx['R_nipple_x'],j],X_pos_[pos_idx['R_nipple_y'],j]],
                                                  [X_pos_[pos_idx['R_hip_x'],j],X_pos_[pos_idx['R_hip_y'],j]],
                                                  [X_pos_[pos_idx['L_hip_x'],j],X_pos_[pos_idx['L_hip_y'],j]]);
        X_angles_[angle_idx['L_nose_shoulders'],j] = angle([X_pos_[pos_idx['nose_x'],j],X_pos_[pos_idx['nose_y'],j]],
                                                  [X_pos_[pos_idx['L_nipple_x'],j],X_pos_[pos_idx['L_nipple_y'],j]],
                                                  [X_pos_[pos_idx['R_nipple_x'],j],X_pos_[pos_idx['R_nipple_y'],j]]); 
        X_angles_[angle_idx['R_nose_shoulders'],j] = angle([X_pos_[pos_idx['nose_x'],j],X_pos_[pos_idx['nose_y'],j]],
                                                  [X_pos_[pos_idx['R_nipple_x'],j],X_pos_[pos_idx['R_nipple_y'],j]],
                                                  [X_pos_[pos_idx['L_nipple_x'],j],X_pos_[pos_idx['L_nipple_y'],j]]);
    return likelihood, X_pos_, X_angles_


def labels_likelihood_plot(likelihood):
    '''Plots labels reliability'''
    fig = plt.figure(figsize=(15,150))
    fig.suptitle('Likelihood distribution', size=titles_font_size, y=0.89)

    for i in range(len(likelihood)):

        plt.subplot(int(len(pos_idx)/2),2,i+1)
        plt.hist(likelihood[i], bins=70, color = 'skyblue', ec = 'blue', alpha=0.7)
        plt.xlabel('Probability of Correct Labels',size=plots_font_size)
        plt.ylabel('Likelihood Frequency',size=plots_font_size)
        plt.title(body_parts_labels_[i], size=titles_font_size)
    plt.show()

def plot_posture(L_shoulder,R_shoulder,L_hip,R_hip,L_elbow,R_elbow,L_hand,R_hand,L_knee,R_knee,L_foot,R_foot,nose):

    plt.figure(figsize=(6,6))
    plt.plot([L_hand[0],L_elbow[0],L_shoulder[0],R_shoulder[0],R_hip[0],L_hip[0],L_shoulder[0],nose[0],R_shoulder[0],R_elbow[0],R_hand[0]],
             [L_hand[1],L_elbow[1],L_shoulder[1],R_shoulder[1],R_hip[1],L_hip[1],L_shoulder[1],nose[1],R_shoulder[1],R_elbow[1],R_hand[1]],'b')
    plt.plot([L_foot[0],L_knee[0],L_hip[0],R_hip[0],R_knee[0],R_foot[0]],
             [L_foot[1],L_knee[1],L_hip[1],R_hip[1],R_knee[1],R_foot[1]],'b')
    x_points = [nose[0],L_shoulder[0],R_shoulder[0],L_hip[0],R_hip[0],
                L_elbow[0],R_elbow[0],L_hand[0],R_hand[0],L_knee[0],R_knee[0],L_foot[0],R_foot[0]]
    y_points = [nose[1],L_shoulder[1],R_shoulder[1],L_hip[1],R_hip[1],
                L_elbow[1],R_elbow[1],L_hand[1],R_hand[1],L_knee[1],R_knee[1],L_foot[1],R_foot[1]]
    plt.scatter(x_points,y_points,c = range(len(x_points)), cmap = 'hsv')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().invert_yaxis()
    plt.show()

def plot_posture_from_xy(X_pos,i):
    L_hand = np.empty((2,1));L_hand[:]=np.nan; R_hand = np.empty((2,1));R_hand[:]=np.nan;
    L_elbow = np.empty((2,1));L_elbow[:]=np.nan; R_elbow = np.empty((2,1));R_elbow[:]=np.nan;
    L_shoulder = np.empty((2,1));L_shoulder[:]=np.nan; R_shoulder = np.empty((2,1));R_shoulder[:]=np.nan; 
    L_foot = np.empty((2,1));L_foot[:]=np.nan; R_foot = np.empty((2,1)); R_foot[:]=np.nan;
    L_knee = np.empty((2,1));L_knee[:]=np.nan; R_knee = np.empty((2,1));R_knee[:]=np.nan; 
    L_hip = np.empty((2,1));L_hip[:]=np.nan; R_hip = np.empty((2,1));R_hip[:]=np.nan; 
    nose = np.empty((2,1)); nose[:]=np.nan;
    
    L_hand[:,0] = [X_pos[pos_idx['L_wrist_x'],i],X_pos[pos_idx['L_wrist_y'],i]]
    R_hand[:,0] = [X_pos[pos_idx['R_wrist_x'],i],X_pos[pos_idx['R_wrist_y'],i]]
    L_elbow[:,0] = [X_pos[pos_idx['L_elbow_x'],i],X_pos[pos_idx['L_elbow_y'],i]]
    R_elbow[:,0] = [X_pos[pos_idx['R_elbow_x'],i],X_pos[pos_idx['R_elbow_y'],i]]
    L_shoulder[:,0] = [X_pos[pos_idx['L_nipple_x'],i],X_pos[pos_idx['L_nipple_y'],i]]
    R_shoulder[:,0] = [X_pos[pos_idx['R_nipple_x'],i],X_pos[pos_idx['R_nipple_y'],i]]
    L_foot[:,0] = [X_pos[pos_idx['L_ankle_x'],i],X_pos[pos_idx['L_ankle_y'],i]]
    R_foot[:,0] = [X_pos[pos_idx['R_ankle_x'],i],X_pos[pos_idx['R_ankle_y'],i]]
    L_knee[:,0] = [X_pos[pos_idx['L_knee_x'],i],X_pos[pos_idx['L_knee_y'],i]]
    R_knee[:,0] = [X_pos[pos_idx['R_knee_x'],i],X_pos[pos_idx['R_knee_y'],i]]
    L_hip[:,0] = [X_pos[pos_idx['L_hip_x'],i],X_pos[pos_idx['L_hip_y'],i]]
    R_hip[:,0] = [X_pos[pos_idx['R_hip_x'],i],X_pos[pos_idx['R_hip_y'],i]]
    nose[:,0] = [X_pos[pos_idx['nose_x'],i],X_pos[pos_idx['nose_y'],i]]
    
    plot_posture(L_shoulder,R_shoulder,L_hip,R_hip,L_elbow,R_elbow,L_hand,R_hand,L_knee,R_knee,L_foot,R_foot,nose)

def plot_video(positions):
    for i in range(positions.shape[1]):
        plot_posture_from_xy(positions,i)    

def check_dataset_posture(positions):
    n,m = positions.shape
    print(m)
    for i in range(m):
        k = 20
        if i % k == 0:
            plot_posture_from_xy(positions,i)
      
# Plotting time series:

def positions_plot(positions,fs=50):
    n_samples = positions.shape[1]
    time = [datetime.datetime(2021,1,1,0,0,0) + datetime.timedelta(seconds=i/fs) for i in range(n_samples)]
    for label, idx in pos_idx.items():
        plt.figure(figsize=(16,4))
        plt.xlabel('Time (mins:secs)', size = plots_font_size)
        plt.ylabel('Position (pixel)', size = plots_font_size)
        plt.plot(time,positions[idx,:])
        
        plt.gcf().autofmt_xdate()
        myFmt = mdates.DateFormatter('%M:%S')
        plt.gca().xaxis.set_major_formatter(myFmt)
        
        plt.title('Time series of {} position'.format(label), size=titles_font_size)
    plt.show()
    
def angles_plot(angles,fs=50):
    n_samples = angles.shape[1]
    time = [datetime.datetime(2021,1,1,0,0,0) + datetime.timedelta(seconds=i/fs) for i in range(n_samples)]
    for label, idx in angle_idx.items():
        plt.figure(figsize=(16,4))
        plt.xlabel('Time (mins:secs)', size = plots_font_size)
        plt.ylabel('Angle (rads)', size = plots_font_size)
        plt.plot(time,angles[idx,:])
        plt.gcf().autofmt_xdate()
        myFmt = mdates.DateFormatter('%M:%S')
        plt.gca().xaxis.set_major_formatter(myFmt)
        plt.title('Time series of angle at: {}'.format(label), size=titles_font_size)
    plt.show()
    
def plot_corr_matrix(data,title,labels):
    ''' Plot correlation matrix'''
    data_ = delete_nans(data)
    corr_matrix=np.corrcoef(data_) 
    
    fig, ax = plt.subplots(1,1,figsize=(8,8))
    im = ax.imshow(abs(corr_matrix),cmap='binary', vmin = 0, vmax = 1,origin='lower')
    cbar = fig.colorbar(im,ax=ax,anchor=(0, 0.1), shrink=0.8); 
    cbar.ax.set_ylabel('Pearson Correlation Coefficient', size = plots_font_size)
    plt.title(title,size=titles_font_size)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels);
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels);
    plt.xticks(rotation=90)
    plt.show()
    return abs(corr_matrix)
    
# Preprocessing data

def center(data):
    data_ = np.empty_like(data)
    for i,time_series in enumerate(data):
        data_[i] = data[i] - np.nanmean(data[i])
    return data_

def standardize(data):
    data_ = np.empty_like(data)
    for i,time_series in enumerate(data):
        data_[i] = data[i] - np.nanmean(data[i])
        data_[i] = data_[i]/np.nanstd(data[i])
    return data_

def normalize(data,lower_limit=5,upper_limit=95):
    data_ = np.empty_like(data)
    for i,time_series in enumerate(data):
        min_ = np.nanpercentile(data[i],lower_limit)
        max_ = np.nanpercentile(data[i],upper_limit)
        data_[i] = data[i] - min_
        data_[i] = data_[i]/(max_ - min_)
    data__ = center(data_)
    return data__

def normalize_angles_range(data):
    data_ = np.empty_like(data)
    elbow_knee_arm_leg_max = np.pi
    elbow_knee_arm_leg_min = 0
    low_percentile = 0.5
    high_percentile = 99.5
    trunk = np.array([data[angle_idx['L_upper_trunk'],:],
                     data[angle_idx['R_upper_trunk'],:],
                     data[angle_idx['L_lower_trunk'],:],
                     data[angle_idx['R_lower_trunk'],:]])
    min_trunk = np.nanpercentile(trunk,low_percentile)
    max_trunk = np.nanpercentile(trunk,high_percentile)
    #min_trunk = np.nanmin(trunk)#1 #np.nanpercentile(trunk,low_percentile)
    #max_trunk = np.nanmax(trunk)#2.2 #np.nanpercentile(trunk,high_percentile)
    head = np.array([data[angle_idx['L_nose_shoulders'],:],
                     data[angle_idx['R_nose_shoulders'],:]])
    #min_head =np.nanmin(head)#0.3 #np.nanpercentile(head,low_percentile)
    #max_head = np.nanmax(head)#2 #np.nanpercentile(head,high_percentile)
    
    min_head = np.nanpercentile(head,low_percentile)
    max_head = np.nanpercentile(head,high_percentile)
    
    print('max_trunk: {}'.format(max_trunk))
    print('min_trunk: {}'.format(min_trunk))
    print('max_head: {}'.format(max_head))
    print('min_head: {}'.format(min_head))
    
    data_[angle_idx['L_elbow'],:] = (data[angle_idx['L_elbow'],:] - elbow_knee_arm_leg_min)/(elbow_knee_arm_leg_max-elbow_knee_arm_leg_min)
    data_[angle_idx['R_elbow'],:] = (data[angle_idx['R_elbow'],:] - elbow_knee_arm_leg_min)/(elbow_knee_arm_leg_max-elbow_knee_arm_leg_min)
    data_[angle_idx['L_knee'],:] = (data[angle_idx['L_knee'],:] - elbow_knee_arm_leg_min)/(elbow_knee_arm_leg_max-elbow_knee_arm_leg_min)
    data_[angle_idx['R_knee'],:] = (data[angle_idx['R_knee'],:] - elbow_knee_arm_leg_min)/(elbow_knee_arm_leg_max-elbow_knee_arm_leg_min)
    data_[angle_idx['L_arm'],:] = (data[angle_idx['L_arm'],:] - elbow_knee_arm_leg_min)/(elbow_knee_arm_leg_max-elbow_knee_arm_leg_min)
    data_[angle_idx['R_arm'],:] = (data[angle_idx['R_arm'],:] - elbow_knee_arm_leg_min)/(elbow_knee_arm_leg_max-elbow_knee_arm_leg_min)
    data_[angle_idx['L_leg'],:] = (data[angle_idx['L_leg'],:] - elbow_knee_arm_leg_min)/(elbow_knee_arm_leg_max-elbow_knee_arm_leg_min)
    data_[angle_idx['R_leg'],:] = (data[angle_idx['R_leg'],:] - elbow_knee_arm_leg_min)/(elbow_knee_arm_leg_max-elbow_knee_arm_leg_min)
    
    data_[angle_idx['L_upper_trunk'],:] = (data[angle_idx['L_upper_trunk'],:] - min_trunk)/(max_trunk-min_trunk)
    data_[angle_idx['R_upper_trunk'],:] = (data[angle_idx['R_upper_trunk'],:] - min_trunk)/(max_trunk-min_trunk)
    data_[angle_idx['L_lower_trunk'],:] = (data[angle_idx['L_lower_trunk'],:] - min_trunk)/(max_trunk-min_trunk)
    data_[angle_idx['R_lower_trunk'],:] = (data[angle_idx['R_lower_trunk'],:] - min_trunk)/(max_trunk-min_trunk)
    data_[angle_idx['L_nose_shoulders'],:] = (data[angle_idx['L_nose_shoulders'],:] - min_head)/(max_head-min_head)
    data_[angle_idx['R_nose_shoulders'],:] = (data[angle_idx['R_nose_shoulders'],:] - min_head)/(max_head-min_head)

    return data_

def get_matrix_angles_angular_velocity(angles):
    # Angles: angles in the time series, including nans for values with low likelihood
    angles_normalized_ = normalize_angles_range(angles)
    angular_velocity_normalized = np.diff(angles_normalized_)
    #Delete last data point so that angles and angular velocity have same length
    angles_normalized = np.delete(angles_normalized_,0,1) 
    
    #Normalize angles and velocity by the maximum and minimum of each group
    max_angles = np.nanmax(angles_normalized)
    min_angles = np.nanmin(angles_normalized)
    angles_normalized__ = (angles_normalized.copy() - min_angles)/(max_angles-min_angles)
    max_angular_velocity = np.nanmax(angular_velocity_normalized)
    min_angular_velocity = np.nanmin(angular_velocity_normalized)
    angular_velocity_normalized__ = (angular_velocity_normalized.copy()-min_angular_velocity)/(max_angular_velocity-min_angular_velocity)
    
    data = np.concatenate((angles_normalized__,angular_velocity_normalized__),axis=0)
    data_ = delete_nans(data)
    return data_

def get_tangential_velocity(velocity):
    samples = velocity.shape[1]
    N = velocity.shape[0]
    vars = int(N/2)
    tangential_velocity = np.empty((vars,samples))
    for j in range(samples):
        for i in range(vars):
            tangential_velocity[i,j] = np.sqrt(velocity[2*i,j]**2 + velocity[2*i+1,j]**2)
    return tangential_velocity

def fourier_transform(data,labels,fs=50):
    for i,label in enumerate(labels):
        x = data[i,:]
        xi = np.arange(len(x))
        mask = np.isfinite(x)
        x_interpolated = np.interp(xi, xi[mask], x[mask])
        time=[]
        n_samples = data.shape[1]
        time = [datetime.datetime(2021,1,1,0,0,0) + datetime.timedelta(seconds=i/fs) for i in range(n_samples)]

        plt.figure(figsize=(16,4))
        plt.xlabel('Time (mins:secs)', size = plots_font_size)
        plt.ylabel('Position at: {} (pixel)'.format(label), size = plots_font_size)
        plt.plot(time,x_interpolated,'r',label='interpolated')
        plt.plot(time,x,'b',label='Original')
        plt.gcf().autofmt_xdate()
        myFmt = mdates.DateFormatter('%M:%S')
        plt.gca().xaxis.set_major_formatter(myFmt)
        plt.title('Time series of {} position'.format(label), size=titles_font_size)
        plt.legend(fontsize = 12, loc='upper right')
        plt.show()
        
        N = len(x_interpolated) # Number of sample points
        T = 1.0 / fs # Period

        #time_steps = np.arange(0,N,1)
        #time = T * time_steps

        yf = fft(x_interpolated)
        xf = fftfreq(N, T)[:N//2]

        plt.figure(figsize=(15,3)); 
        plt.suptitle('Fourier transform of {} timeseries'.format(label),size=titles_font_size)
        plt.subplot(1,3,1);
        
        plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]),'kx')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude spectrum')
        plt.grid()
        plt.xlim(-0.001,0.02)
        #plt.ylim([-1,10])

        plt.subplot(1,3,2);

        plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude spectrum')
        plt.grid()
        plt.xlim(-0.1,1.5)
        plt.ylim(-1,30)
        
        plt.subplot(1,3,3);
        
        plt.semilogy(xf, 2.0/N * np.abs(yf[0:N//2]))
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude spectrum')
        plt.grid()
        plt.ylim(1e-4, 1e4)
        
        plt.show()
        
        f, Pxx_den = welch(x_interpolated,fs)
        plt.semilogy(f, Pxx_den)
        plt.ylim([1e-3, 1e4])
        plt.title('Welch periodogram of {} timeseries'.format(label), size=titles_font_size)
        plt.xlabel('frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')
        plt.grid()
        plt.show()