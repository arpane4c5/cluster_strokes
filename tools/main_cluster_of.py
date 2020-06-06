#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 03:02:08 2020

@author: arpan

@Description: main file for clustering of optical flow from strokes (Highlights data).
"""

import _init_paths

import os
import seaborn as sns
import numpy as np
import pandas as pd
from utils import extract_of
from utils import trajectory_utils as traj_utils
from utils import spectral_utils
from utils import plot_utils
from evaluation import eval_of_clusters

from sklearn.preprocessing import normalize

sns.set()


# Local Paths
LABELS = "/home/arpan/VisionWorkspace/Cricket/scripts/supporting_files/sample_set_labels/sample_labels_shots/ICC WT20"
DATASET = "/home/arpan/VisionWorkspace/VideoData/sample_cricket/ICC WT20"
# Server Paths
if os.path.exists("/opt/datasets/cricket/ICC_WT20"):
    LABELS = "/home/arpan/VisionWorkspace/shot_detection/supporting_files/sample_set_labels/sample_labels_shots/ICC WT20"
    DATASET = "/opt/datasets/cricket/ICC_WT20"

ANNOTATION_FILE = "stroke_labels.txt"



def get_hoof_spectral(dest_dir):
    '''Extract the HOOF features from strokes and perform Spectral Clustering using 
    a similarity/distance threshold. 
    '''
    #bin_counts = list(range(5, 21, 2))
    #thresh_list = list(range(2, 30, 3))
    
    bin_counts = [5]
    thresh_list = [2]
    
    bins_vals = []
    thresh_vals = []
    best_acc = []
    
    for bins in bin_counts:
        for thresh in thresh_list:
            print(60*'*')
            # Extract trajectory features (HOOF) using mag threshold and nbinn
            trajectories, stroke_names = extract_of.extract_all_feats(DATASET, \
                                                            LABELS, bins, thresh)
            
            destPath = os.path.join(dest_dir, "bins_"+str(bins)+"_th_"+str(thresh))
            if not os.path.exists(destPath):
                os.makedirs(destPath)
            #np.save(os.path.join(destPath, "feats_bins"+str(bins)+"_th_"+str(thresh)+".npy"), all_feats)
            #all_feats = np.load(os.path.join(destPath, "feats_bins"+str(bins)+"_th_"+str(thresh)+".npy"))
            print("bin:{}, thresh:{} ".format(bins, thresh))
            all_feats = traj_utils.get_trajectory_mean(trajectories)
            all_feats = normalize(all_feats, norm='l2')
            ###################################################################
            #plot_utils.plot_trajectories3D(trajectories, stroke_names)
            
            nclusters = [3]     # list of no of clusters
            threshold = 50000
            # returns a list of cluster label assignments for trajectories
            labels_spec = spectral_utils.spectral(all_feats, trajectories, nclusters, threshold)
            # Write the strokes in their respective folders
            eval_of_clusters.evaluate(labels_spec, stroke_names, bins, thresh, dest_dir, DATASET)

    

def get_hoof_kmeans(dest_dir):
    '''Extract the HOOF feature sequences from strokes and perform K-Means Clustering
    on a range of bins, HOOF mag threshold and nCluster values.
    The feature clustered is the mean of the trajectories.
    Parameters:
    -----------
    dest_dir : str
        destination folder name for saving the output
    '''
    bin_counts = list(range(5, 21, 2))
    thresh_list = list(range(2, 30, 3))
    
    bins_vals = []
    thresh_vals = []
    best_acc = []
    
    for bins in bin_counts:
        for thresh in thresh_list:
            print(60*'*')
            trajectories, stroke_names = extract_of.extract_all_feats(DATASET, \
                                                            LABELS, bins, thresh)
            #all_feats[np.isinf(all_feats)] = 0
            destPath = os.path.join(dest_dir, "bins_"+str(bins)+"_th_"+str(thresh))
            if not os.path.exists(destPath):
                os.makedirs(destPath)
            #np.save(os.path.join(destPath, "feats_bins"+str(bins)+"_th_"+str(thresh)+".npy"), all_feats)
            #all_feats = np.load(os.path.join(destPath, "feats_bins"+str(bins)+"_th_"+str(thresh)+".npy"))
            print("bin:{}, thresh:{} ".format(bins, thresh))
            all_feats = traj_utils.get_trajectory_mean(trajectories)
            all_feats = normalize(all_feats, norm='l2')
            ###################################################################
            #plot_utils.plot_trajectories3D(trajectories, stroke_names)
            
            sse_list = []
            x_range = list(range(1, 10))
            print("Clustering for 50 clusters ... ")
            for i in x_range:
                km = traj_utils.kmeans(all_feats, i)
                labels = km.labels_
                sse_list.append(km.inertia_)
                #print("labels_", labels)
            
            plot_utils.plot_sse_score(sse_list, x_range, bins, thresh, dest_dir)
            print("Final clustering for 3 clusters... \n")
            
            km = traj_utils.kmeans(all_feats, 3)
            labels = km.labels_
    
            acc_values, perm_tuples, gt_list, pred_list = \
                eval_of_clusters.get_accuracy(all_feats, labels, ANNOTATION_FILE, \
                                              DATASET, LABELS)
            acc_perc = [sum(k)/all_feats.shape[0] for k in acc_values]
            #best_acc[i,j] = max(acc_perc)
            bins_vals.append(bins)
            thresh_vals.append(thresh)
            best_acc.append(max(acc_perc))
            best_indx = acc_perc.index(max(acc_perc))
            print("Max Acc. : ", max(acc_perc))
            print("Acc values : ", acc_perc)
            print("Acc values : ", acc_values)
            print("perm_tuples : ", perm_tuples)
            
            if bins>2:
                pca_flows = traj_utils.apply_PCA(all_feats)
            else:
                pca_flows = all_feats
            plot_utils.plot_clusters(pca_flows, labels, perm_tuples[best_indx], bins,\
                                     thresh, dest_dir, 'cluster_pca_ordered.png')
            
            pca_flows = traj_utils.apply_tsne(all_feats)
            plot_utils.plot_clusters(pca_flows, labels, perm_tuples[best_indx], bins, \
                                     thresh, dest_dir, 'cluster_tsne_ordered.png')
            
            eval_of_clusters.evaluate(labels, stroke_names, bins, thresh, dest_dir, DATASET)
            
            break
        break
            
    df = pd.DataFrame({"Bins(w)": bins_vals, "Magnitude Threshold (Pixels)":thresh_vals, "Accuracy(percent)":best_acc})
    df = df.pivot("Bins(w)", "Magnitude Threshold (Pixels)", "Accuracy(percent)")
    ax = sns.heatmap(df, vmin=0., vmax=1., annot=True, fmt='.2f', cmap="YlGnBu")
    
    normal_heat = sns.heatmap(df, vmin=0., vmax=1., annot=True, fmt='.2f')
    normal_heat.figure.savefig(os.path.join(dest_dir, "normal_heat_bins10_thresh10.png"))
    for i, g in enumerate(gt_list):
        print(g, pred_list[i])
#    dbs = traj_utils.dbscan(all_feats)
#    print(dbs.labels_)
    

if __name__=='__main__':
    spect_dest = 'res_OF_spec'
    kmeans_dest = 'res_OF_kmeans'
    get_hoof_spectral(kmeans_dest)
    get_hoof_kmeans(spect_dest)
    

    