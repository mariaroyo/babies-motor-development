# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 11:07:15 2021

@author: Maria Royo
"""
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
from scipy.stats import ranksums
from IPython.display import display
import scipy.stats
import matplotlib.patches as mpatches
from scipy.linalg import qr, svd, inv
from settings_config import small_plots_font_size,plots_font_size,titles_font_size,\
    angle_idx,position_labels_,angle_labels_,angle_labels_2,angles_angular_velocity_labels,body_parts_labels_
from preprocessing import center,get_matrix_angles_angular_velocity
from scipy.stats import distributions,rankdata
from matplotlib.lines import Line2D 


################### PCA ################

def perform_PCA(data):
    # data should be nxm. n=#features, m=#samples
    pca = PCA()
    pca.fit(data.T)
    PCs = pca.components_.T
    explained_variance_ratio = pca.explained_variance_ratio_
    cum = 0
    cum_explained_var_ratio = []
    for value in explained_variance_ratio:
        cum += value
        cum_explained_var_ratio.append(cum)
    return PCs,cum_explained_var_ratio

def plot_early_vs_late_cum_var(early,late,dim=3,save_fig=None):
    labels = ['Baby 1','Baby 2', 'Baby 3','Baby 4']
    early_cum_var = []
    late_cum_var = []
    for i in range(len(early)):
        _,early_cum_vars = perform_PCA(early[i])
        early_cum_var.append(early_cum_vars[dim-1]*100)
        _,late_cum_vars = perform_PCA(late[i])
        late_cum_var.append(late_cum_vars[dim-1]*100)
    
    print(early_cum_var)
    print(late_cum_var)
    
    fig,ax = plt.subplots(figsize=(4,4.4))
    #fig=plt.figure(figsize=(4,4))
    plt.scatter(early_cum_var,late_cum_var)
    
    ax.annotate(labels[0], (early_cum_var[0]-8, late_cum_var[0]-6),size = 11)
    ax.annotate(labels[1], (early_cum_var[1]+3, late_cum_var[1]-1),size = 11)
    ax.annotate(labels[2], (early_cum_var[2]+3, late_cum_var[2]-1),size = 11)
    ax.annotate(labels[3], (early_cum_var[3]-8, late_cum_var[3]+3),size = 11)
    '''
    ax.annotate(labels[0], (early_cum_var[0]-8, late_cum_var[0]-9),size = 11)
    ax.annotate(labels[1], (early_cum_var[1]-30, late_cum_var[1]-3),size = 11)
    ax.annotate(labels[2], (early_cum_var[2]-30, late_cum_var[2]-1),size = 11)
    ax.annotate(labels[3], (early_cum_var[3]-8, late_cum_var[3]+3),size = 11)
    '''
    plt.plot([0,100],[0,100],c='black',lw=1)
    plt.xlim(0,100)
    plt.ylim(0,100)
    plt.xlabel('Early var. (%)',size=plots_font_size)
    plt.ylabel('Late var. (%)',size=plots_font_size)
    plt.title('{}D manifold expl. variance'.format(dim),size=plots_font_size, y=1.08)
    fig.tight_layout() 
    ax.set_aspect('equal')
    plt.show()
    if save_fig != None:
        fig.savefig('./figures/{}.png'.format(save_fig))
        
def cum_expl_var_early_late(babies,save_fig=None):
    colors = ['blue','blue','blue','blue','red','red','red','red']
    labels = ['Early 1','Early 2','Early 3','Early 4','Late 1','Late 2','Late 3','Late 4']
    ls_=[':','--','-.','-',':','--','-.','-']
    fig,ax = plt.subplots(figsize=(6,4))
    #fig = plt.figure(figsize=(8,5))
    PC1_cum_var = []
    for i,baby_angles in enumerate(babies):
        _,cum_var = perform_PCA(baby_angles)
        plt.plot(np.array(range(len(cum_var)))+1,np.array(cum_var)*100,color=colors[i],
                 label = labels[i],alpha=0.8,lw=1,ls=ls_[i])
        PC1_cum_var.append(cum_var[0])
        #ax.annotate(labels_[i], (3+0.1*i, cum_var[0]*100),size = 6) 
        #ax.annotate(labels_[i], (0.4, pos[i]),size = 9) #ax.annotate(labels_[i], (0.6, cum_var[0]*100),size = 6) 

    plt.plot([1,len(cum_var)],[95,95],linewidth=1,color='k')
    #plt.plot([1,len(cum_var)],[90,90],linewidth=1,color='k')
    plt.xlim(0.5,9.5)
    plt.ylim(0, 105)
    plt.xlabel('Number of PCs',size=plots_font_size)
    plt.ylabel('Cum. expl. variance (%)',size=plots_font_size)
    #plt.title('Cumulative Explained Variance',size=titles_font_size)
    #plt.legend(fontsize = 12, loc='lower right')
    early1 = Line2D([0], [0], label='Early 1', color='b',ls=ls_[0])
    late1 = Line2D([0], [0], label='Late 1', color='r',ls=ls_[0])
    early2 = Line2D([0], [0], label='Early 2', color='b',ls=ls_[1])
    late2 = Line2D([0], [0], label='Late 2', color='r',ls=ls_[1])
    early3 = Line2D([0], [0], label='Early 3', color='b',ls=ls_[2])
    late3 = Line2D([0], [0], label='Late 3', color='r',ls=ls_[2])
    early4 = Line2D([0], [0], label='Early 4', color='b',ls=ls_[3])
    late4 = Line2D([0], [0], label='Late 4', color='r',ls=ls_[3])
    plt.legend(handles=[early1,late1,early2,late2,early3,late3,early4,late4],fontsize = 12,loc='lower right')
    fig.tight_layout() 
    plt.show()
    if save_fig != None:
        fig.savefig('./figures/{}.png'.format(save_fig))
        fig.savefig('./figures/{}.pdf'.format(save_fig))

def participation_ratio(data):
    n,m = data.shape
    C = np.dot(data, data.T) / (m-1)
    w, v = np.linalg.eig(C)
    PR = (np.sum(w))**2/np.sum(np.power(w,2))
    return PR
        

def plot_cum_expl_var_raw_stand_norm(raw,standardized,normalized,title): 
    #plt.figure()
    n=len(raw)
    plt.plot(raw,'kx',color='r',label = 'Raw')
    plt.plot(standardized,'kx',color='g',label = 'Standardized')
    plt.plot(normalized,'kx',color='b',label = 'Normalized')
    plt.plot([0,n-1],[0.8,0.8],linewidth=1,color='k')
    plt.plot([0,n-1],[0.95,0.95],linewidth=1,color='k')
    plt.ylim(0, 1.1)
    plt.xlabel('Number of PCs',size=16)
    plt.ylabel('Cumulative explained variance ',size=16)
    plt.title(title, size=titles_font_size)
    plt.legend(fontsize = 12, loc='lower right')
    #plt.show()

def plot_cum_expl_var_raw_norm(raw,normalized,title): 
    #plt.figure()
    n=len(raw)
    plt.plot(raw,'kx',color='r',label = 'Raw')
    plt.plot(normalized,'kx',color='b',label = 'Normalized')
    plt.plot([0,n-1],[0.8,0.8],linewidth=1,color='k')
    plt.plot([0,n-1],[0.95,0.95],linewidth=1,color='k')
    plt.ylim(0, 1.1)
    plt.xlabel('Number of PCs',size=16)
    plt.ylabel('Cumulative explained variance ',size=16)
    plt.title(title, size=titles_font_size)
    plt.legend(fontsize = 12, loc='lower right')
    #plt.show()

def plot_PCs(PCs,fig,ax,title,labels = angle_labels_, ylabel = True):
    #fig, ax = plt.subplots(1,1,figsize=(8,8))
    ax.imshow(abs(PCs)) #,cmap='hot'
    ax.set_title(title,size=plots_font_size);
    ax.set_xlabel('PCs', size = small_plots_font_size)
    if ylabel:
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels);
    #im.set_clim(0,1)

def plot_PCs_weights(PCs, n_PCs,labels = angle_labels_2):
    for i in range(n_PCs):
        PC = abs(pd.Series(PCs[:,i]))
        label_PC = [x for _,x in sorted(zip(PC,labels))]
        label_PC.reverse()
        plt.figure(figsize=(6, 2))
        fig = PC[np.argsort(PC)[::-1]].plot(kind='bar');
        fig.set_title('PC {} weights'.format(i+1),size=plots_font_size);
        fig.set_ylabel('Weight',size=14);
        fig.set_xticklabels(label_PC,size=12);

def compare_PCs_weights(PCs1,PCs2, n_PCs,video,labels = angle_labels_2):
    fig, axs = plt.subplots(n_PCs,2,figsize=(15,3*n_PCs+1))
    fig.suptitle('{} PCs weights: raw (left) vs. normalized (right)'.format(video), size=titles_font_size, y=1.05)
    for i in range(n_PCs):
        PC1 = abs(pd.Series(PCs1[:,i]))
        label_PC1 = [x for _,x in sorted(zip(PC1,labels))]
        label_PC1.reverse()
        
        #plt.subplot(n_PCs,2,2*i+1);
        PC1[np.argsort(PC1)[::-1]].plot(kind='bar',ax=axs[i,0]);
        axs[i,0].set_title('PC {}'.format(i+1),size=plots_font_size);
        axs[i,0].set_ylabel('Weight',size=14);
        axs[i,0].set_xticklabels(label_PC1,size=14);
        
        PC2 = abs(pd.Series(PCs2[:,i]))
        label_PC2 = [x for _,x in sorted(zip(PC2,labels))]
        label_PC2.reverse()
        #plt.subplot(n_PCs,2,2*i+2);
        PC2[np.argsort(PC2)[::-1]].plot(kind='bar',ax=axs[i,1],color = 'orange');
        axs[i,1].set_title('PC {}'.format(i+1),size=plots_font_size);
        axs[i,1].set_ylabel('Weight',size=14);
        axs[i,1].set_xticklabels(label_PC2,size=14);
    fig.tight_layout()
    plt.show();

def reconstruct_data_from_PCs(data,PCs,n_PCs,labels,y_label='Angle'):
    a=[]; b=[];PCs_reduced = [];areduced=[];breduced=[];
    a = np.dot(PCs.T,data) # Project data onto the principal components
    b = np.dot(a.T,PCs.T).T # Reverse project the projection back into data space

    PCs_reduced = PCs.copy()
    PCs_reduced[:][n_PCs:PCs.shape[1]]=0 # Reduced PCs space
    areduced = np.dot(PCs_reduced.T,data) # Project the data onto the principal components
    breduced = np.dot(areduced.T,PCs_reduced.T).T # Reverse project the projection back into data space

    for label, i in angle_idx.items():
        plt.figure(figsize=(16,4))
        plt.xlabel('Time samples', size = plots_font_size)
        plt.ylabel('{}'.format(y_label), size = plots_font_size)
        plt.plot(data[i,:],'r',linestyle='-',label = 'Original',linewidth=1)
        plt.plot(b[i,:],'b',linestyle='--',label = 'Reconstructed with all PCs',linewidth=1)
        plt.plot(breduced[i,:],'g',linestyle='--',label = 'Reconstructed with {} PCs'.format(n_PCs),linewidth=1)
        plt.legend(fontsize = 12, loc='upper right')
        plt.title('Time series of {0} at: {1}'.format(y_label,label), size=titles_font_size)
        plt.show()
    return breduced

def compare_PCA_results(data1,data2):
    PCs1,cum_explained_var_ratio1 = perform_PCA(data1)
    PCs2,cum_explained_var_ratio2 = perform_PCA(data2)
    plt.figure()
    plt.plot(cum_explained_var_ratio1,'kx',color='r',label = 'Data 1')
    plt.plot(cum_explained_var_ratio2,'kx',color='g',label = 'Data 2')
    n = len(cum_explained_var_ratio1)
    plt.plot([0,n-1],[0.8,0.8],linewidth=1,color='k')
    plt.plot([0,n-1],[0.95,0.95],linewidth=1,color='k')
    plt.ylim(0, 1.1)
    plt.xlabel('Number of PCs',size=16)
    plt.ylabel('Cumulative explained variance ',size=16)
    #plt.title(title, size=titles_font_size)
    plt.legend(fontsize = 12, loc='lower right')
    plt.show()
    
    fig, ax = plt.subplots(1,2,figsize=(16,9))
    plt.suptitle('PCs (PCA on angles)',y=0.91,size=titles_font_size)
    plot_PCs(PCs1,fig, ax[0],'Data1')
    plot_PCs(PCs2,fig, ax[1],'Data2')
    
    corr = []
    for i in range(14):
        #corr.append(np.corrcoef(scores1[:,i],scores2[:,i])[0,1])
        corr.append(np.corrcoef(PCs1[:,i],PCs2[:,i])[0,1])
    print(corr)
    
    plt.figure()
    plt.plot(list(range(1, 15)),abs(np.array(corr)),'kx',color='r')
    plt.xlabel('PC',size=16)
    plt.ylabel('Pearson Correlation Coefficient',size=16)
    plt.title('Correlation between PCs', size=titles_font_size)
    plt.ylim(-0.1,1.1)
    plt.show() 
    


def PCA_2_videos_raw_norm(angles_EMT45_, angles_subject45,
                                angles_EMT45_normalized_, angles_subject45_normalized_, 
                                videos,label = 'angles'):
    PCs_raw_EMT45,cum_var_raw_EMT45 = perform_PCA(angles_EMT45_)
    PCs_raw_EMTsubject45,cum_var_raw_subject45 = perform_PCA(angles_subject45)
    PCs_normalized_EMT45,cum_var_normalized_EMT45 = perform_PCA(angles_EMT45_normalized_)
    PCs_normalized_subject45,cum_var_normalized_subject45 = perform_PCA(angles_subject45_normalized_)
    
    fig = plt.figure(figsize=(10,4));  
    plt.suptitle('Cumulative Explained Variance (PCA on {})'.format(label),size=titles_font_size, y=1.1)
    plt.subplot(1,2,1);
    plot_cum_expl_var_raw_norm(cum_var_raw_EMT45,cum_var_normalized_EMT45,videos[0])
    plt.subplot(1,2,2);
    plot_cum_expl_var_raw_norm(cum_var_raw_subject45,cum_var_normalized_subject45,videos[1])
    plt.show();
    
    if label == 'angles' or label == 'angular velocity':
        labels = angle_labels_2
    elif label == 'positions' or label == 'x and y velocity':
        labels = position_labels_
    else:
        labels = body_parts_labels_
    
    fig, ax = plt.subplots(2,2,figsize=(12,13))
    plt.suptitle('PCs (PCA on {})'.format(label),y=0.91,size=titles_font_size)
    plot_PCs(PCs_raw_EMT45,fig, ax[0,0],'Raw {}'.format(videos[0]),labels)
    plot_PCs(PCs_normalized_EMT45,fig, ax[1,0],'Norm. {}'.format(videos[0]),labels)
    plot_PCs(PCs_raw_EMTsubject45,fig, ax[0,1],'Raw {}'.format(videos[1]),labels)
    plot_PCs(PCs_normalized_subject45,fig, ax[1,1],'Norm. {}'.format(videos[1]),labels)

def PCA_3_videos_angles_angular_velocity(angles_EMT16, angles_EMT46_2,angles_EMT46_4):
    angles_angular_velocity_EMT16 = center(get_matrix_angles_angular_velocity(angles_EMT16))
    angles_angular_velocity_EMT46_2 = center(get_matrix_angles_angular_velocity(angles_EMT46_2))
    angles_angular_velocity_EMT46_4 = center(get_matrix_angles_angular_velocity(angles_EMT46_4))
    PCs_EMT16,cum_var_EMT16 = perform_PCA(angles_angular_velocity_EMT16)
    PCs_EMT46_2,cum_var_EMT46_2 = perform_PCA(angles_angular_velocity_EMT46_2)
    PCs_EMT46_4,cum_var_EMT46_4 = perform_PCA(angles_angular_velocity_EMT46_4)
    
    plt.figure()
    n=len(cum_var_EMT16)
    plt.plot(cum_var_EMT16,'kx',color='r',label = 'Early mild')
    plt.plot(cum_var_EMT46_2,'kx',color='g',label = 'Late mild 1')
    plt.plot(cum_var_EMT46_4,'kx',color='b',label = 'Late mild 2')
    plt.plot([0,n-1],[0.8,0.8],linewidth=1,color='k')
    plt.plot([0,n-1],[0.95,0.95],linewidth=1,color='k')
    plt.ylim(0, 1.1)
    plt.xlabel('Number of PCs',size=16)
    plt.ylabel('Cum. explained variance ',size=16)
    plt.title('Cumulative Explained Variance (PCA on angles and angular velocity)',size=titles_font_size)
    plt.legend(fontsize = 12, loc='lower right')
    plt.show()
    
    fig, ax = plt.subplots(3,1,figsize=(16,25))
    plt.suptitle('PCs (PCA on angles and angular velocity)',y=0.91,size=titles_font_size)
    plot_PCs(PCs_EMT16,fig, ax[0],'Early mild',angles_angular_velocity_labels)
    plot_PCs(PCs_EMT46_2,fig, ax[1],'Late mild 1',angles_angular_velocity_labels)
    plot_PCs(PCs_EMT46_4,fig, ax[2],'Late mild 2',angles_angular_velocity_labels)

def PCA_2_videos_angles_angular_velocity(angles_EMT16, angles_EMT46_2):
    angles_angular_velocity_EMT16 = center(get_matrix_angles_angular_velocity(angles_EMT16))
    angles_angular_velocity_EMT46_2 = center(get_matrix_angles_angular_velocity(angles_EMT46_2))
    PCs_EMT16,cum_var_EMT16 = perform_PCA(angles_angular_velocity_EMT16)
    PCs_EMT46_2,cum_var_EMT46_2 = perform_PCA(angles_angular_velocity_EMT46_2)
    
    plt.figure()
    n=len(cum_var_EMT16)
    plt.plot(cum_var_EMT16,'kx',color='r',label = 'Typical 1')
    plt.plot(cum_var_EMT46_2,'kx',color='g',label = 'Typical 2')
    plt.plot([0,n-1],[0.8,0.8],linewidth=1,color='k')
    plt.plot([0,n-1],[0.95,0.95],linewidth=1,color='k')
    plt.ylim(0, 1.1)
    plt.xlabel('Number of PCs',size=16)
    plt.ylabel('Cum. explained variance ',size=16)
    plt.title('Cumulative Explained Variance (PCA on angles and angular velocity)',size=titles_font_size)
    plt.legend(fontsize = 12, loc='lower right')
    plt.show()
    
    fig, ax = plt.subplots(2,1,figsize=(16,18))
    plt.suptitle('PCs (PCA on angles and angular velocity)',y=0.91,size=titles_font_size)
    plot_PCs(PCs_EMT16,fig, ax[0],'Typical 1',angles_angular_velocity_labels)
    plot_PCs(PCs_EMT46_2,fig, ax[1],'Typical 2',angles_angular_velocity_labels)

############ PCA CROSSPROJECTION SIMILARITY ###############

def PCA_crossprojection_similarity(data1,data2):
   
    n1,m1 = data1.shape
    pca1 = PCA()
    pca1.fit(data1.T)
    PCs1 = pca1.components_.T
    
    pca2 = PCA()
    pca2.fit(data2.T)
    PCs2 = pca2.components_.T
    
    similarity = []
    for n_PCs in range(1,n1+1):
        # Reduced PCs space
        PCs1_reduced = PCs1.copy()
        PCs1_reduced.T[:][n_PCs:PCs1.shape[1]]=0 
        PCs2_reduced = PCs2.copy()
        PCs2_reduced.T[:][n_PCs:PCs2.shape[1]]=0 
    
        projected1_onto1 = np.dot(PCs1_reduced.T,data1) # Project the angles onto the principal components
        T1 = np.var(projected1_onto1,axis=1)
    
        projected1_onto2 = np.dot(PCs2_reduced.T,data1) # Project the angles onto the principal components
        T2 = np.var(projected1_onto2,axis=1)
                    
        similarity.append(np.sum(T2)/np.sum(T1))

    return similarity

def PCA_crossprojection_similarity_dims(data1,data2,dims):
   
    n1,m1 = data1.shape
    pca1 = PCA()
    pca1.fit(data1.T)
    PCs1 = pca1.components_.T
    
    pca2 = PCA()
    pca2.fit(data2.T)
    PCs2 = pca2.components_.T

    PCs1_reduced = np.delete(PCs1,list(range(dims,PCs1.shape[1])) ,1)
    PCs2_reduced = np.delete(PCs2,list(range(dims,PCs2.shape[1])) ,1)

    projected1_onto1 = np.dot(PCs1_reduced.T,data1) # Project the angles onto the principal components
    T1 = np.var(projected1_onto1,axis=1)

    projected1_onto2 = np.dot(PCs2_reduced.T,data1) # Project the angles onto the principal components
    T2 = np.var(projected1_onto2,axis=1)

    similarity = np.sum(T2)/np.sum(T1)

    return similarity

def PCA_random_crossprojection_similarity_dims(data1,dims):
   
    n1,m1 = data1.shape
    pca1 = PCA()
    pca1.fit(data1.T)
    PCs1 = pca1.components_.T

    PCs2 = PCs1.copy()
    np.random.shuffle(PCs2)

    PCs1_reduced = np.delete(PCs1,list(range(dims,PCs1.shape[1])) ,1)
    PCs2_reduced = np.delete(PCs2,list(range(dims,PCs2.shape[1])) ,1)

    projected1_onto1 = np.dot(PCs1_reduced.T,data1) # Project the angles onto the principal components
    T1 = np.var(projected1_onto1,axis=1)

    projected1_onto2 = np.dot(PCs2_reduced.T,data1) # Project the angles onto the principal components
    T2 = np.var(projected1_onto2,axis=1)

    similarity = np.sum(T2)/np.sum(T1)

    return similarity

def mean_PCA_crossprojection_similarity(data1,data2):
    similarity1 = PCA_crossprojection_similarity(data1,data2)
    print(similarity1)
    similarity2 = PCA_crossprojection_similarity(data2,data1)
    print(similarity2)
    similarity = []
    for i in range(len(similarity1)):
        similarity.append((similarity1[i]+similarity2[i])/2)
    #print(similarity)
    pcs = np.arange(1,(len(similarity)+1),1)
    plt.figure()
    plt.plot(pcs,similarity,'kx',color='r')
    plt.xlabel('Subspace Dimensionality',size=16)
    plt.ylabel('Projected Variance Similarity',size=16)
    #plt.title('PCA crossprojection similarity', size=titles_font_size)
    plt.ylim(-0.1,1.1)
    plt.show() 

def PCA_random_crossprojection_similarity(data1,plot=True):
   
    n1,m1 = data1.shape
    pca1 = PCA()
    pca1.fit(data1.T)
    PCs1 = pca1.components_.T

    PCs2 = PCs1.copy()
    np.random.shuffle(PCs2)

    similarity = []
    for n_PCs in range(1,n1+1):
        # Reduced PCs space
        PCs1_reduced = PCs1.copy()
        PCs1_reduced.T[:][n_PCs:PCs1.shape[1]]=0 
        PCs2_reduced = PCs2.copy()
        PCs2_reduced.T[:][n_PCs:PCs2.shape[1]]=0 
    
        projected1_onto1 = np.dot(PCs1_reduced.T,data1) # Project the angles onto the principal components
        T1 = np.var(projected1_onto1,axis=1)
    
        projected1_onto2 = np.dot(PCs2_reduced.T,data1) # Project the angles onto the principal components
        T2 = np.var(projected1_onto2,axis=1)
                    
        similarity.append(np.sum(T2)/np.sum(T1))
        
    if plot:
        pcs = np.arange(1,(len(similarity)+1),1)
        plt.figure()
        plt.plot(pcs,similarity,'kx',color='r')
        plt.xlabel('Subspace Dimensionality',size=16)
        plt.ylabel('Projected Variance Similarity',size=16)
        #plt.title('PCA crossprojection similarity', size=titles_font_size)
        plt.ylim(-0.1,1.1)
        plt.show() 

    return similarity

def self_crossprojection_similarity(data,video,chunk_length=200,dims=6):
    n,m = data.shape
    n_chunks = int(np.round(m/chunk_length))
    chunks = []
    for i in range(n_chunks-1):
        chunks.append(data[:,chunk_length*i:chunk_length*(i+1)])
    cross_similarity = []
    random_similarity = []
    for chunk1 in chunks:
        for chunk2 in chunks:
            if (chunk1 != chunk2).any():
                cross_similarity.append(PCA_crossprojection_similarity(chunk1,chunk2)[dims-1]*100)
                random_similarity.append(PCA_random_crossprojection_similarity(chunk1,plot=False)[dims-1]*100)
    bins=np.histogram(np.hstack((cross_similarity,random_similarity)), bins=80)[1]
    #plt.figure()
    plt.hist(random_similarity, bins=bins, color = 'skyblue', ec = 'blue', alpha=0.7,label='Random projection')
    plt.hist(cross_similarity, bins=bins, color = 'orange', ec = 'OrangeRed', alpha=0.7,label='Crossprojection')
    #plt.plot([], [], ' ', label="Dimensions: {}".format(dims))
    plt.xlabel('Var. expl. cross-segment (%)',size=plots_font_size)
    plt.ylabel('Segment pairs',size=plots_font_size)
    plt.title('{}'.format(video), size=titles_font_size)
    plt.legend(fontsize = 12)
    #plt.show()
    _, pvalue = ranksums(random_similarity,cross_similarity)
    print('Wilcoxon rank sum test p-value: {}.'.format(np.round(pvalue,3)))
    # If p-value < 0.05, this test rejects the null hypothesis that these measurements are drawn from the same distribution.
    
def self_crossprojection_videos_plot(videos,labels,dims,data_type):
    fig = plt.figure(figsize=(16,12))
    plt.suptitle('Self-crossprojection similarity on {} data for a {}-dimensional manifold'.format(data_type,dims),y=1.05,size=titles_font_size)
    for i in range(len(videos)):
        plt.subplot(math.ceil(len(videos)/2),2,i+1)
        self_crossprojection_similarity(videos[i],labels[i],dims=dims)
    fig.tight_layout()
    plt.show();

def crossprojection_videos(videos,labels,dim=6,data_type='Angles'):
    #videos = [angles_EMT16_normalized_,angles_EMT46_2_normalized_,angles_EMT46_4_normalized_,angles_EMT45_normalized_,angles_subject45_normalized_]
    n = len(videos)
    videos_random_similarity = np.empty(n)
    iters = 200
    n_vars = videos[0].shape[0]
    for i,video in enumerate(videos):
        random_similarity = np.zeros((iters,n_vars))
        for j in range(iters):
            random_similarity[j,:] = PCA_random_crossprojection_similarity(video,plot=False)
        videos_random_similarity[i] = np.percentile(random_similarity[:,dim-1],99)
    videos_similarity = np.empty((n,n,n_vars))
    for i,video1 in enumerate(videos):
        for j,video2 in enumerate(videos):
            similarity = PCA_crossprojection_similarity(video1,video2)
            videos_similarity[i,j,:] = similarity

    videos_similarity_ = videos_similarity[:,:,dim-1]
    new_column = [[i] for i in videos_random_similarity]
    videos_similarity_ = np.append(videos_similarity_,new_column,axis=1)
    new_row = np.append(videos_random_similarity.copy(),np.nan)
    videos_similarity_ = np.append(videos_similarity_,[new_row],axis=0)
    #labels = ['16','46_2','46_4','45','subj45','random']
    
    df = pd.DataFrame(np.round(videos_similarity_*100), columns=labels, index = labels)
    display(df.style.background_gradient(cmap='viridis',axis=None).format('{0:,.0f}%')\
    .set_caption('{} - Var. expl. cross-video manifold (%) ({} dimensions)'.format(data_type,dim)).set_table_styles([{
        'selector': 'caption',
        'props': [
            ('color', 'black'),
            ('font-size', '14px')]},
        dict(selector='.data', props=[('width', '5em')])])\
    .highlight_null(null_color='white'))

def crossprojection_videos_distributions(early,late,dim=6):
    videos = early + late
    similarity_same_group = []
    similarity_different_group = []
    for i in range(len(early)):
        for j in range(len(late)):
            similarity_different_group.append(PCA_crossprojection_similarity(early[i],late[j])[dim-1]*100)
    for group in [early,late]:
        for i in range(len(group)):
            for j in range(len(group)):
                if i!=j:
                    similarity_same_group.append(PCA_crossprojection_similarity(group[i],group[j])[dim-1]*100)

    videos_random_similarity = []
    iters = 200
    for i,video in enumerate(videos):
        #random_similarity = np.zeros((iters,n_vars))
        for j in range(iters):
            videos_random_similarity.append(PCA_random_crossprojection_similarity(video,plot=False)[dim-1]*100)
            #random_similarity[j,:] = PCA_random_crossprojection_similarity(video,plot=False)
        #videos_random_similarity[i] = np.percentile(random_similarity[:,dim-1],99)*100
    
    videos_random_similarity_ = np.random.choice(videos_random_similarity,100)
    bins=np.histogram(np.hstack((similarity_same_group,similarity_different_group)), bins=10)[1]
    plt.figure()
    plt.hist(videos_random_similarity_, bins=30, color = 'grey', ec = 'grey', alpha=0.7,label='Random')
    plt.hist(similarity_same_group, bins=bins, color = 'b', ec = 'b', alpha=0.7,label='Within group')
    plt.hist(similarity_different_group, bins=bins, color = 'r', ec = 'r', alpha=0.7,label='Across group')
    #plt.plot([], [], ' ', label="Dimensions: {}".format(dims))
    plt.xlabel('Var. expl. cross-video (%)',size=plots_font_size)
    plt.ylabel('Video pairs',size=plots_font_size)
    plt.title('Manifold crossprojection (%) ({} dimensions)'.format(dim), size=titles_font_size)
    plt.xlim(0,100)
    plt.legend(fontsize = 12)
    plt.show()
    _, pvalue = ranksums(similarity_same_group,similarity_different_group)
    print('Wilcoxon rank sum test p-value: {}.'.format(np.round(pvalue,3)))

def manifolds_comparison(early,late,dim=6,chunk_length=500):
    videos = early + late
    similarity_same_group = []
    similarity_different_group = []
    for i in range(len(early)):
        for j in range(len(late)):
            similarity_different_group.append(PCA_crossprojection_similarity_dims(early[i],late[j],dim)*100)
    for group in [early,late]:
        for i in range(len(group)):
            for j in range(len(group)):
                if i!=j:
                    similarity_same_group.append(PCA_crossprojection_similarity_dims(group[i],group[j],dim)*100)
    
    random_similarity = []
    for video in videos:
            random_similarity.append(PCA_random_crossprojection_similarity_dims(video,dim)*100)
    
    
    
    cross_similarity = []
    for video in videos:
        n,m = video.shape
        n_chunks = int(np.round(m/chunk_length))
        chunks = []
        for i in range(n_chunks-1):
            chunks.append(video[:,chunk_length*i:chunk_length*(i+1)])
    
        for chunk1 in chunks:
            for chunk2 in chunks:
                if (chunk1 != chunk2).any():
                    cross_similarity.append(PCA_crossprojection_similarity_dims(chunk1,chunk2,dim)*100)
                    random_similarity.append(PCA_random_crossprojection_similarity_dims(chunk1,dim)*100)
    if dim>0 and dim<5:
        n_bins = 20
    elif dim>=5 and dim<9:
        n_bins = 30
    elif dim>=9:
        n_bins = 70
    bins=np.histogram(np.hstack((similarity_same_group,similarity_different_group,cross_similarity,random_similarity)), bins=n_bins)[1]             
        
    plt.figure()
    plt.hist(random_similarity, density=True, bins=bins, color = 'grey', ec = 'grey', alpha=0.6,label='Random')
    plt.hist(similarity_same_group, density=True, bins=bins, color = 'c', ec = 'c', alpha=0.6,label='Within group')
    plt.hist(similarity_different_group, density=True, bins=bins, color = 'm', ec = 'm', alpha=0.6,label='Across group')
    plt.hist(cross_similarity, density=True, bins=bins, color = 'yellow', ec = 'yellow', alpha=0.6,label='Within video')
    #plt.plot([], [], ' ', label="Dimensions: {}".format(dims))
    plt.xlabel('Cross-projection similarity (%)',size=plots_font_size)
    plt.ylabel('Manifold pairs',size=plots_font_size)
    plt.title('{}D Manifolds comparison'.format(dim), size=titles_font_size)
    plt.xlim(0,100)
    plt.legend(fontsize = 12)
    plt.gca().axes.yaxis.set_ticks([])
    plt.show()
    
    print('Wilcoxon rank sum tests p-values')
    _, pvalue = ranksums(similarity_same_group,similarity_different_group)
    print('Within group and across group: {}.'.format(np.round(pvalue,3)))
    _, pvalue = ranksums(cross_similarity,similarity_different_group)
    print('Within video and across group: {}.'.format(np.round(pvalue,3)))
    _, pvalue = ranksums(similarity_same_group,cross_similarity)
    print('Within video and within group: {}.'.format(np.round(pvalue,3)))
    #print(np.percentile(cross_similarity,99))


def PCA_crossprojection_similarity_chunks(data1,data2,dim,chunk_length):
    similarity = []
    random_similarity = []
    n1,m1 = data1.shape
    n2,m2 = data2.shape
    n_chunks1 = int(np.round(m1/chunk_length))
    n_chunks2 = int(np.round(m2/chunk_length))
    chunks1 = []
    for i in range(n_chunks1-1):
        chunks1.append(data1[:,chunk_length*i:chunk_length*(i+1)])
    chunks2 = []
    for i in range(n_chunks2-1):
        chunks2.append(data2[:,chunk_length*i:chunk_length*(i+1)])
    for chunk1 in chunks1:
        for chunk2 in chunks2:
            similarity.append(PCA_crossprojection_similarity_dims(chunk1,chunk2,dim)*100)
            similarity.append(PCA_crossprojection_similarity_dims(chunk2,chunk1,dim)*100)
            random_similarity.append(PCA_random_crossprojection_similarity_dims(chunk1,dim)*100)
            random_similarity.append(PCA_random_crossprojection_similarity_dims(chunk2,dim)*100)
    return similarity, random_similarity
                    
def manifolds_comparison_chunks(early,late,dim=6,chunk_length=500,save_fig=None,exclude_across_group_same_babies=True):
    videos = early + late
    similarity_same_group = []
    similarity_different_group = []
    random_similarity = []
    for i in range(len(early)):
        for j in range(len(late)):
            if exclude_across_group_same_babies:
                if i!=j:
                    similarity, random_similar = PCA_crossprojection_similarity_chunks(early[i],late[j],dim,chunk_length)
                    similarity_different_group += similarity
                    random_similarity += random_similar
            else:
                similarity, random_similar = PCA_crossprojection_similarity_chunks(early[i],late[j],dim,chunk_length)
                similarity_different_group += similarity
                random_similarity += random_similar
    for group in [early,late]:
        for i in range(len(group)):
            for j in range(len(group)):
                if i!=j:
                    similarity, random_similar = PCA_crossprojection_similarity_chunks(group[i],group[j],dim,chunk_length)
                    similarity_same_group += similarity
                    random_similarity += random_similar
    
    #random_similarity = []
    #for video in videos:
    #        random_similarity.append(PCA_random_crossprojection_similarity_dims(video,dim)*100)
    
    if exclude_across_group_same_babies:
        across_group_label = 'Across group (excluding same baby comparison)'
    else:
        across_group_label = 'Across group (including same baby comparison)'
    
    self_similarity = []
    for video in videos:
        n,m = video.shape
        n_chunks = int(np.round(m/chunk_length))
        chunks = []
        for i in range(n_chunks-1):
            chunks.append(video[:,chunk_length*i:chunk_length*(i+1)])
    
        for chunk1 in chunks:
            for chunk2 in chunks:
                if (chunk1 != chunk2).any():
                    self_similarity.append(PCA_crossprojection_similarity_dims(chunk1,chunk2,dim)*100)
                    random_similarity.append(PCA_random_crossprojection_similarity_dims(chunk1,dim)*100)
  
    fig = plt.figure(figsize=(10,6))
    plt.hist(random_similarity, density=True, bins='auto', color = 'grey', ec = 'grey', alpha=0.4,label='Random')
    
    plt.hist(self_similarity, density=True, bins='auto', color = 'magenta', ec = 'magenta', alpha=0.6,label='Within video')
    plt.hist(similarity_same_group, density=True, bins='auto', color = 'deepskyblue', ec = 'deepskyblue', alpha=0.6,label='Within group')
    plt.hist(similarity_different_group, density=True, bins='auto', color = 'yellow', ec = 'yellow', alpha=0.6,label=across_group_label)
    
    #plt.plot([], [], ' ', label="Dimensions: {}".format(dims))
    plt.xlabel('Cross-projection similarity (%)',size=plots_font_size)
    plt.ylabel('Manifold pairs density',size=plots_font_size)
    plt.title('{}D Manifolds comparison'.format(dim), size=titles_font_size)
    plt.xlim(0,100)
    plt.legend(fontsize = 12)
    plt.gca().axes.yaxis.set_ticks([])
    plt.show()
    
    if save_fig != None:
        fig.savefig('./figures/{}.png'.format(save_fig))
    '''
    print('')
    print('Wilcoxon rank sum tests p-values')
    print('')
    _, pvalue = ranksums(self_similarity,similarity_different_group)
    print('Within video and across group: {}.'.format(np.round(pvalue,3)))
    _, pvalue = ranksums(similarity_same_group,self_similarity)
    print('Within video and within group: {}.'.format(np.round(pvalue,3)))

    print('')
    _, pvalue = ranksums(similarity_same_group,similarity_different_group)
    print('2-sided rank sum test within and across group: {}.'.format(np.round(pvalue,3)))
    '''
    _, pvalue = Wilcoxon_rank_sum_test(similarity_same_group,similarity_different_group,alternative='greater')
    print('Rank sum test p-value (within greater than across): {}.'.format(np.round(pvalue,6)))
    '''
    _, pvalue = Wilcoxon_rank_sum_test(similarity_different_group,similarity_same_group,alternative='greater')
    print('Rank sum test p-value (across greater than within): {}.'.format(np.round(pvalue,4)))
    '''


def manifolds_comparison_chunks2(early,late,dim=6,chunk_length=500,save_fig=None):
    videos = early + late
    similarity_same_group = []
    similarity_different_group = []
    random_similarity = []
    same_baby_similarity=[]
    for i in range(len(early)):
        for j in range(len(late)):
            if i!=j:
                similarity, random_similar = PCA_crossprojection_similarity_chunks(early[i],late[j],dim,chunk_length)
                similarity_different_group += similarity
                random_similarity += random_similar
            else:
                similarity, random_similar = PCA_crossprojection_similarity_chunks(early[i],late[j],dim,chunk_length)
                same_baby_similarity += similarity
                random_similarity += random_similar
    for group in [early,late]:
        for i in range(len(group)):
            for j in range(len(group)):
                if i!=j:
                    similarity, random_similar = PCA_crossprojection_similarity_chunks(group[i],group[j],dim,chunk_length)
                    similarity_same_group += similarity
                    random_similarity += random_similar
    
    #random_similarity = []
    #for video in videos:
    #        random_similarity.append(PCA_random_crossprojection_similarity_dims(video,dim)*100)

    self_similarity = []
    for video in videos:
        n,m = video.shape
        n_chunks = int(np.round(m/chunk_length))
        chunks = []
        for i in range(n_chunks-1):
            chunks.append(video[:,chunk_length*i:chunk_length*(i+1)])
    
        for chunk1 in chunks:
            for chunk2 in chunks:
                if (chunk1 != chunk2).any():
                    self_similarity.append(PCA_crossprojection_similarity_dims(chunk1,chunk2,dim)*100)
                    random_similarity.append(PCA_random_crossprojection_similarity_dims(chunk1,dim)*100)
  
    fig = plt.figure(figsize=(10,6))
    plt.hist(random_similarity, density=True, bins='auto', color = 'grey', ec = 'grey', 
             alpha=0.4,label='Random')
    #chartreuse mediumslateblue
    plt.hist(self_similarity, density=True, bins='auto', color = 'teal', ec = 'teal', 
             alpha=0.4,label='Same baby, same age')
    
    plt.hist(same_baby_similarity, density=True, bins='auto', color = 'magenta', ec = 'magenta', 
             alpha=0.4,label='Same baby, different age')
    
    plt.hist(similarity_same_group, density=True, bins='auto', color='deepskyblue', ec='deepskyblue',
             alpha=0.4,label='Different babies, same ages')
    
    plt.hist(similarity_different_group, density=True, bins='auto', color = 'yellow', ec = 'yellow',
             alpha=0.4,label='Different babies, different ages')
    
    #plt.plot([], [], ' ', label="Dimensions: {}".format(dims))
    plt.xlabel('Cross-projection similarity (%)',size=plots_font_size)
    plt.ylabel('{}D Manifold pairs density'.format(dim),size=plots_font_size)
    #plt.title('{}D Manifolds comparison'.format(dim), size=titles_font_size)
    plt.xlim(0,100)
    plt.legend(fontsize = 12,loc='upper left')
    plt.gca().axes.yaxis.set_ticks([])
    plt.show()
    
    if save_fig != None:
        fig.savefig('./figures/{}.png'.format(save_fig))
    '''
    print('')
    print('Wilcoxon rank sum tests p-values')
    print('')
    _, pvalue = ranksums(self_similarity,similarity_different_group)
    print('Within video and across group: {}.'.format(np.round(pvalue,3)))
    _, pvalue = ranksums(similarity_same_group,self_similarity)
    print('Within video and within group: {}.'.format(np.round(pvalue,3)))

    print('')
    _, pvalue = ranksums(similarity_same_group,similarity_different_group)
    print('2-sided rank sum test within and across group: {}.'.format(np.round(pvalue,3)))
    '''
    _, pvalue = Wilcoxon_rank_sum_test(similarity_same_group,similarity_different_group,alternative='greater')
    print('Rank sum test p-value (same age greater than different age): {}.'.format(np.round(pvalue,6)))
    '''
    _, pvalue = Wilcoxon_rank_sum_test(similarity_different_group,similarity_same_group,alternative='greater')
    print('Rank sum test p-value (across greater than within): {}.'.format(np.round(pvalue,4)))
    '''
    return random_similarity,self_similarity,same_baby_similarity,similarity_same_group,similarity_different_group

######### CCA #########
    
def canoncorr(X:np.ndarray, Y: np.ndarray, fullReturn: bool = False) -> np.ndarray:
    """
    Canonical Correlation Analysis (CCA)
    line-by-line port from Matlab implementation of `canoncorr`
    X,Y: (samples/observations) x (features) matrix, for both: X.shape[0] >> X.shape[1]
    fullReturn: whether all outputs should be returned or just `r` be returned (not in Matlab)
    
    returns: A,B,r,U,V 
    A,B: Canonical coefficients for X and Y
    U,V: Canonical scores for the variables X and Y
    r:   Canonical correlations
    
    Signature:
    A,B,r,U,V = canoncorr(X, Y)
    """
    n, p1 = X.shape
    p2 = Y.shape[1]
    if p1 >= n or p2 >= n:
        print('Not enough samples, might cause problems')
    # Center the variables
    X = X - np.mean(X,0);
    Y = Y - np.mean(Y,0);
    # Factor the inputs, and find a full rank set of columns if necessary
    Q1,T11,perm1 = qr(X, mode='economic', pivoting=True, check_finite=True)
    rankX = sum(np.abs(np.diagonal(T11)) > np.finfo(type((np.abs(T11[0,0])))).eps*max([n,p1]));
    if rankX == 0:
        raise Exception('stats:canoncorr:BadData = X')
    elif rankX < p1:
        print('stats:canoncorr:NotFullRank = X')
        Q1 = Q1[:,:rankX]
        T11 = T11[rankX,:rankX]
    Q2,T22,perm2 = qr(Y, mode='economic', pivoting=True, check_finite=True)
    rankY = sum(np.abs(np.diagonal(T22)) > np.finfo(type((np.abs(T22[0,0])))).eps*max([n,p2]));
    if rankY == 0:
        raise Exception('stats:canoncorr:BadData = Y')
    elif rankY < p2:
        print('stats:canoncorr:NotFullRank = Y')
        Q2 = Q2[:,:rankY];
        T22 = T22[:rankY,:rankY];
    # Compute canonical coefficients and canonical correlations.  For rankX >
    # rankY, the economy-size version ignores the extra columns in L and rows
    # in D. For rankX < rankY, need to ignore extra columns in M and D
    # explicitly. Normalize A and B to give U and V unit variance.
    d = min(rankX,rankY);
    L,D,M = svd(Q1.T @ Q2, full_matrices=True, check_finite=True, lapack_driver='gesdd')
    M = M.T
    A = inv(T11) @ L[:,:d] * np.sqrt(n-1);
    B = inv(T22) @ M[:,:d] * np.sqrt(n-1);
    r = D[:d]
    # remove roundoff errs
    r[r>=1] = 1
    r[r<=0] = 0
    if not fullReturn:
        return r
    # Put coefficients back to their full size and their correct order
    A[perm1,:] = np.vstack((A, np.zeros((p1-rankX,d))))
    B[perm2,:] = np.vstack((B, np.zeros((p2-rankY,d))))
    
    # Compute the canonical variates
    U = X @ A
    V = Y @ B
    return A, B, r, U, V


def canonical_correlations(A,B,dims=6):
    na,ma = A.shape
    pcaA = PCA()
    pcaA.fit(A.T)
    PCsA = pcaA.components_.T
    
    nb,mb = B.shape
    pcaB = PCA()
    pcaB.fit(B.T)
    PCsB = pcaB.components_.T
    
    # Reduced PCs space
    PCsA_reduced = np.delete(PCsA,list(range(dims,PCsA.shape[1])) ,1)
    PCsB_reduced = np.delete(PCsB,list(range(dims,PCsB.shape[1])) ,1)

    lvA = np.dot(PCsA_reduced.T,A) # Project the angles onto the principal components
    lvB = np.dot(PCsB_reduced.T,B) # Project the angles onto the principal components

    m = min(ma,mb)
    corr = canoncorr(lvA[:,0:m].T,lvB[:,0:m].T)

    return corr

def get_max_CC(data,chunk_length=200,dims=6):
    n,m = data.shape
    n_chunks = int(np.round(m/chunk_length))
    chunks = []
    for i in range(n_chunks-1):
        chunks.append(data[:,chunk_length*i:chunk_length*(i+1)])
    CCs = []
    for chunk1 in chunks:
        for chunk2 in chunks:
            if (chunk1 != chunk2).any():
                CCs.append(canonical_correlations(chunk1,chunk2,dims))
    CCs = np.array(CCs)
    max_CCs = np.percentile(CCs,99,axis=0)
    return np.array(max_CCs)

def CCs_across_within_groups(early,late,dims=6,plot=True):
    CCs_same_group = []
    CCs_different_group = []
    for i in range(len(early)):
        for j in range(len(late)):
            CCs = canonical_correlations(early[i],late[j],dims)
            CCs_different_group.append(CCs)
    for group in [early,late]:
        for i in range(len(group)):
            for j in range(len(group)):
                if i!=j:
                    CCs = canonical_correlations(group[i],group[j],dims)
                    CCs_same_group.append(CCs) 
    if plot:
        plot_CCs_across_within_groups(CCs_same_group,CCs_different_group)
    return CCs_same_group,CCs_different_group

def CCs_across_within_groups_normalized(early,late,dims=6,plot=True):
    early_max_CCs = []
    late_max_CCs = []
    for baby in early:
        early_max_CCs.append(get_max_CC(baby))
    for baby in late:
        late_max_CCs.append(get_max_CC(baby))
    CCs_same_group = []
    CCs_different_group = []
    for i in range(len(early)):
        for j in range(len(late)):
            CCs = canonical_correlations(early[i],late[j],dims)
            CCs_norm = np.array(CCs)/np.maximum(early_max_CCs[i],late_max_CCs[j])
            CCs_different_group.append(CCs_norm)
    for group,max_CCs in zip([early,late],[early_max_CCs,late_max_CCs]):
        for i in range(len(group)):
            for j in range(len(group)):
                if i!=j:
                    CCs = canonical_correlations(group[i],group[j],dims)
                    CCs_norm = np.array(CCs)/np.maximum(max_CCs[i],max_CCs[j])
                    CCs_same_group.append(CCs_norm) 
    if plot:
        plot_CCs_across_within_groups(CCs_same_group,CCs_different_group)
    return CCs_same_group,CCs_different_group


def plot_CCs_across_within_groups(CCs_same_group,CCs_different_group):

    same_group_mean = np.mean(np.array(CCs_same_group),axis=0)
    different_group_mean = np.mean(np.array(CCs_different_group),axis=0)
    same_group_std = np.std(np.array(CCs_same_group),axis=0)
    different_group_std = np.std(np.array(CCs_different_group),axis=0)
    
    modes = np.arange(len(same_group_mean))+1
    plt.figure(figsize=(8,5))
    linewidth = 3
    for CCs in CCs_same_group:
        plt.plot(modes,CCs,'b', alpha=0.2)
    plt.plot(modes,same_group_mean,'b',linewidth=linewidth)
    plt.plot(modes,same_group_mean+same_group_std,'b',linestyle='dashed',linewidth=linewidth)
    plt.plot(modes,same_group_mean-same_group_std,'b',linestyle='dashed',linewidth=linewidth)
    for CCs in CCs_different_group:
        plt.plot(modes,CCs,'r', alpha=0.2)
    plt.plot(modes,different_group_mean,'r',linewidth=linewidth)
    plt.plot(modes,different_group_mean+different_group_std,'r',linestyle='dashed',linewidth=linewidth)
    plt.plot(modes,different_group_mean-different_group_std,'r',linestyle='dashed',linewidth=linewidth)
    plt.xlabel('Mode',size=plots_font_size)
    plt.ylabel('CC',size=plots_font_size)
    plt.title('Within group and across groups CCs',size=titles_font_size)
    plt.xticks(modes,fontsize=12)
    plt.yticks(fontsize=12)
    #plt.ylim(-0.05,1)
    red_patch = mpatches.Patch(color='r', label='Across group')
    blue_patch = mpatches.Patch(color='b', label='Within group')
    plt.legend(handles=[red_patch,blue_patch],fontsize = 12)
    plt.show()
    
########## CROSSPROJECTION CORRELATION #############

def crossprojection_correlation(data1,data2,n_PCs=6):
   
    n1,m1 = data1.shape
    pca1 = PCA()
    pca1.fit(data1.T)
    PCs1 = pca1.components_.T
    pca2 = PCA()
    pca2.fit(data2.T)
    PCs2 = pca2.components_.T
    
    # Reduced PCs space
    PCs1_reduced = np.delete(PCs1,list(range(n_PCs,PCs1.shape[1])) ,1)
    PCs2_reduced = np.delete(PCs2,list(range(n_PCs,PCs2.shape[1])) ,1)
    projected1_onto1 = np.dot(PCs1_reduced.T,data1) # Project the angles onto the principal components
    projected1_onto2 = np.dot(PCs2_reduced.T,data1) # Project the angles onto the principal components
                    
    corr = []
    for a_row,b_row in zip(projected1_onto1, projected1_onto2):
        corrcoef,p_value = scipy.stats.pearsonr(a_row,b_row)
        corr.append(abs(corrcoef))
    return corr

def random_crossprojection_correlation(data1,n_PCs=6):
   
    n1,m1 = data1.shape
    pca1 = PCA()
    pca1.fit(data1.T)
    PCs1 = pca1.components_.T
    PCs2 = PCs1.copy()
    np.random.shuffle(PCs2)

    # Reduced PCs space
    PCs1_reduced = np.delete(PCs1,list(range(n_PCs,PCs1.shape[1])) ,1)
    PCs2_reduced = np.delete(PCs2,list(range(n_PCs,PCs2.shape[1])) ,1)
    projected1_onto1 = np.dot(PCs1_reduced.T,data1) # Project the angles onto the principal components
    projected1_onto2 = np.dot(PCs2_reduced.T,data1) # Project the angles onto the principal components
                    
    corr = []
    for a_row,b_row in zip(projected1_onto1, projected1_onto2):
        corrcoef,p_value = scipy.stats.pearsonr(a_row,b_row)
        corr.append(abs(corrcoef))
    return corr


def crossprojection_correlation_videos_distributions(early,late,dim=6):
    videos = early + late
    corr_same_group = []
    corr_different_group = []
    for i in range(len(early)):
        for j in range(len(late)):
            corr_different_group += crossprojection_correlation(early[i],late[j],dim)
    for group in [early,late]:
        for i in range(len(group)):
            for j in range(len(group)):
                if i!=j:
                    corr_same_group += crossprojection_correlation(group[i],group[j],dim)
    
    corr_different_group = np.array(corr_different_group)*100
    corr_same_group = np.array(corr_same_group)*100
    
    videos_random_corr = []
    iters = 200
    for i,video in enumerate(videos):
        #random_corr = np.zeros((iters,dim))
        for j in range(iters):
            videos_random_corr += random_crossprojection_correlation(video,dim)
        #videos_random_corr += (np.percentile(random_similarity,99,axis=0)*100).tolist()
    
    videos_random_corr = np.array(videos_random_corr)*100
    
    videos_random_corr_ = np.random.choice(videos_random_corr,500)
    bins=np.histogram(np.hstack((corr_same_group,corr_different_group)), bins=5)[1]
    plt.figure()
    plt.hist(videos_random_corr_, bins='auto', color = 'grey', ec = 'grey', alpha=0.7,label='Random')
    plt.hist(corr_same_group, bins=bins, color = 'b', ec = 'b', alpha=0.7,label='Within group')
    plt.hist(corr_different_group, bins=bins, color = 'r', ec = 'r', alpha=0.7,label='Across group')
    #plt.plot([], [], ' ', label="Dimensions: {}".format(dims))
    plt.xlabel('Cross-video correlations (%)',size=plots_font_size)
    plt.ylabel('Video pairs',size=plots_font_size)
    plt.title('Crossprojection correlation (%) ({} dimensions)'.format(dim), size=titles_font_size)
    plt.xlim(0,100)
    plt.legend(fontsize = 12)
    plt.show()
    _, pvalue = ranksums(corr_same_group,corr_different_group)
    print('Wilcoxon rank sum test p-value: {}.'.format(np.round(pvalue,3)))
    
# Wilcoxom rank sum test

def Wilcoxon_rank_sum_test(x,y,alternative='two-sided'):
    x, y = map(np.asarray, (x, y))
    n1 = len(x)
    n2 = len(y)
    alldata = np.concatenate((x, y))
    ranked = rankdata(alldata)
    x = ranked[:n1]
    s = np.sum(x, axis=0)
    expected = n1 * (n1+n2+1) / 2.0
    z = (s - expected) / np.sqrt(n1*n2*(n1+n2+1)/12.0)
    z, prob = _normtest_finish(z, alternative)
    return z,prob
    
def _normtest_finish(z, alternative):
    """Common code between all the normality-test functions."""
    if alternative == 'less':
        prob = distributions.norm.cdf(z)
    elif alternative == 'greater':
        prob = distributions.norm.sf(z)
    elif alternative == 'two-sided':
        prob = 2 * distributions.norm.sf(np.abs(z))
    else:
        raise ValueError("alternative must be "
                         "'less', 'greater' or 'two-sided'")

    if z.ndim == 0:
        z = z[()]

    return z, prob