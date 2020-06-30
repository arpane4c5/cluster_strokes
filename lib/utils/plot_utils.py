#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 01:05:32 2020

@author: arpan
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import trajectory_utils as traj_utils
from sklearn.preprocessing import normalize


def plot_sse_score(sse_list, x, plot_dir):
    '''Plot the sum of squared error for a range of clusters. The error drops 
    as the number of clusters increases. 
    
    '''
    plt.figure(2)
    plt.xlabel("#Clusters")
    plt.ylabel("SSE")
    plt.plot(x, sse_list, label='sse score', c='r')
    plt.title('SSE score Vs No. of clusters')
    #plt.savefig(os.path.join(plot_dir, 'sse_plot.png'))
    plt.show()

def plot_clusters(data, labels, best_tuple, plot_dir, plotname):
    n_clusters = max(labels) + 1
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    markers = list("12348spphH+xXdD")
    for color,n_cluster in enumerate(best_tuple):
        # print("n_cluster", n_cluster)
        # print("marker", markers[n_cluster])
        #label_str = "Ambiguous"
        #if color==0:
        #    label_str = "Left"
        #elif color==1:
        #    label_str = "Right"
            
        cls_data = data[labels==n_cluster]
        plt.scatter(cls_data[:,0], cls_data[:,1], c=colors[color%8], marker = markers[color], label="C"+str(color+1))
    plt.legend(loc='upper right')
#    if plotname=='cluster_pca_ordered.png':
#        plt.xlim((-1,1))
#        plt.ylim((-1,1))
    plt.savefig(os.path.join(plot_dir, str(n_clusters)+'_'+plotname))
    plt.show()
    
def plot_trajectories3D(stroke_vecs, stroke_names):
    '''
    '''
    stroke_lens = [len(stroke) for strokes in stroke_vecs for stroke in strokes]
    stroke_names = [name for names in stroke_names for name in names]
    all_feats = np.vstack([np.vstack(stroke) for strokes in stroke_vecs for stroke in strokes])
    
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    
    all_feats = normalize(all_feats, norm='l2')
    pca_applied = traj_utils.apply_PCA(all_feats, dims=3)
    
    indx = 0
    for i, slen in enumerate(stroke_lens):
        #print("Stroke : {}".format(stroke_names[i]))
        print("Stroke : {}".format(i))
        fig = plt.figure()
        ax = fig.gca(projection='3d')
#        plt.plot(pca_applied[indx:(indx+slen), 0], pca_applied[indx:(indx+slen), 1], \
#                             color=colors[i % len(colors)], marker='.', lw=1 )
        x = pca_applied[indx:(indx+slen), 0]
        y = pca_applied[indx:(indx+slen), 1]
        z = pca_applied[indx:(indx+slen), 2]
        ax.quiver(x[:-1], y[:-1], z[:-1], x[1:]-x[:-1], y[1:]-y[:-1], z[1:]-z[:-1], \
                   color=colors[0] )
        
        ax.set_xlim([np.min(pca_applied[:,0]), np.max(pca_applied[:,0])])
        ax.set_ylim([np.min(pca_applied[:,1]), np.max(pca_applied[:,1])])
        ax.set_zlim([np.min(pca_applied[:,2]), np.max(pca_applied[:,2])])
        
        indx += slen
        
        fig_name = stroke_names[i]+'.png'
        #fig_name = fig_name_parts[0]+'_'+fig_name_parts[1].split('_', 1)[1]+'.png'
        #fig.savefig(os.path.join(os.getcwd(), 'lib/visualize/graphs3D_v1', fig_name))
        plt.show()
    
def visualize_evecs(eVals, eVecs):
    
    chooseVecs = [1]    # 2nd EVec at col indx 1
    for choice in chooseVecs:
        # get second EigenVector and visualize the sorted vertices
        evec = eVecs[:, choice]
        
        idx = np.argsort(evec)
        evec = evec[idx]
        
        x = list(range(evec.shape[0]))
        y = evec.tolist()
        plt.figure(choice)
        plt.xlabel("Rank of x"+str(choice+1))
        plt.ylabel("Value of x"+str(choice+1))
        plt.plot(x , y, c='r')
        plt.title('Components of EVec'+str(choice+1))
        #plt.savefig(os.path.join(plot_dir, 'sse_plot.png'))
        plt.show()
    return