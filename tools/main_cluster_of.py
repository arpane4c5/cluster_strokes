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
import pickle
from utils import extract_of
from utils import extract_autoenc_feats as autoenc_feats
from utils import trajectory_utils as traj_utils
from utils import spectral_utils
from utils import plot_utils
from evaluation import eval_of_clusters

from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import normalize
from scipy.stats import norm

sns.set()


# Local Paths
LABELS = "/home/arpan/VisionWorkspace/Cricket/scripts/supporting_files/sample_set_labels/sample_labels_shots/ICC WT20"
DATASET = "/home/arpan/VisionWorkspace/VideoData/sample_cricket/ICC WT20"
# Server Paths
if os.path.exists("/opt/datasets/cricket/ICC_WT20"):
    LABELS = "/home/arpan/VisionWorkspace/shot_detection/supporting_files/sample_set_labels/sample_labels_shots/ICC WT20"
    DATASET = "/opt/datasets/cricket/ICC_WT20"

model_path = "checkpoints/autoenc_gru_resnet50_ep10_w6_Adam.pt"
SEQ_SIZE = 6
INPUT_SIZE = 2048
HIDDEN_SIZE = 32#64#1024
NUM_LAYERS = 2
BATCH_SIZE = 2*SEQ_SIZE # set to the number of images of a seqence # 36
ANNOTATION_FILE = "stroke_labels.txt"

def get_autoenc_kmeans(dest_dir):
    print(60*'*')
    # Extract autoencoder features 
    trajectories, stroke_names = autoenc_feats.extract_sequence_feats(model_path, \
                                DATASET, LABELS, BATCH_SIZE, SEQ_SIZE, INPUT_SIZE, \
                                HIDDEN_SIZE, NUM_LAYERS, partition='all', nstrokes=50)
#    
    destPath = dest_dir
    if not os.path.exists(destPath):
        os.makedirs(destPath)
#    with open(os.path.join(destPath, "trajectories.pkl"), 'wb') as fp:
#        pickle.dump(trajectories, fp)
#    with open(os.path.join(destPath, "stroke_names.pkl"), 'wb') as fp:
#        pickle.dump(stroke_names, fp)
    with open(os.path.join(destPath, "trajectories.pkl"), 'rb') as fp:
        trajectories = pickle.load(fp)
    with open(os.path.join(destPath, "stroke_names.pkl"), 'rb') as fp:
        stroke_names = pickle.load(fp)
    #all_feats = np.load(os.path.join(destPath, "feats_bins"+str(bins)+"_th_"+str(thresh)+".npy"))
    print("Extraction Model : {}".format(dest_dir))
    all_feats = traj_utils.get_trajectory_mean(trajectories)
    all_feats = normalize(all_feats, norm='l2')
    ###################################################################
    #plot_utils.plot_trajectories3D(trajectories, stroke_names)
    
    nclusters = [2]     # list of no of clusters
    threshold = 50000
    
    ###################################################################
    # Pre-processing : Forming the dataframe with trajectories as rows
    
    # converting list of list to list of stroke trajectories
    X = [stroke for vid_strokes in trajectories for stroke in vid_strokes]
    row_names = [stroke for vid_strokes in stroke_names for stroke in vid_strokes]
    
    X = pd.DataFrame(X, index=row_names)    
    
    sse_list = []
    x_range = list(range(1, 10))
    print("Clustering for 10 clusters ... ")
    for i in x_range:
        km = traj_utils.kmeans(all_feats, i)
        labels = km.labels_
        sse_list.append(km.inertia_)
        #print("labels_", labels)
    
    plot_utils.plot_sse_score(sse_list, x_range, destPath)
    print("Final clustering for 3 clusters... \n")
    
    km = traj_utils.kmeans(all_feats, 3)
    labels = km.labels_

    acc_values, perm_tuples, gt_list, pred_list = \
        eval_of_clusters.get_accuracy(all_feats, labels, ANNOTATION_FILE, \
                                      DATASET, LABELS)
    acc_perc = [sum(k)/all_feats.shape[0] for k in acc_values]
    
#    best_acc.append(max(acc_perc))
    best_indx = acc_perc.index(max(acc_perc))
    print("Max Acc. : ", max(acc_perc))
    print("Acc values : ", acc_perc)
    print("Acc values : ", acc_values)
    print("perm_tuples : ", perm_tuples)
    
    if all_feats.shape[1] > 2:
        pca_flows = traj_utils.apply_PCA(all_feats)
    else:
        pca_flows = all_feats
    plot_utils.plot_clusters(pca_flows, labels, perm_tuples[best_indx], \
                             destPath, 'cluster_pca_ordered.png')
    
    pca_flows = traj_utils.apply_tsne(all_feats)
    plot_utils.plot_clusters(pca_flows, labels, perm_tuples[best_indx], \
                             destPath, 'cluster_tsne_ordered.png')
    
    eval_of_clusters.evaluate(labels, row_names, destPath, DATASET)
    
    ###################################################################
    
    plot_utils.plot_trajectories3D(trajectories, stroke_names)

def get_autoenc_spectral(dest_dir):
    print(60*'*')
    # Extract autoencoder features 
#    trajectories, stroke_names = autoenc_feats.extract_sequence_feats(model_path, \
#                                DATASET, LABELS, BATCH_SIZE, SEQ_SIZE, INPUT_SIZE, \
#                                HIDDEN_SIZE, NUM_LAYERS, partition='all', nstrokes=-1)
    
    destPath = dest_dir
    if not os.path.exists(destPath):
        os.makedirs(destPath)
#    with open(os.path.join(destPath, "trajectories.pkl"), 'wb') as fp:
#        pickle.dump(trajectories, fp)
#    with open(os.path.join(destPath, "stroke_names.pkl"), 'wb') as fp:
#        pickle.dump(stroke_names, fp)
    with open(os.path.join(destPath, "trajectories.pkl"), 'rb') as fp:
        trajectories = pickle.load(fp)
    with open(os.path.join(destPath, "stroke_names.pkl"), 'rb') as fp:
        stroke_names = pickle.load(fp)
    #all_feats = np.load(os.path.join(destPath, "feats_bins"+str(bins)+"_th_"+str(thresh)+".npy"))
    print("Extraction Model : {}".format(dest_dir))
    all_feats = traj_utils.get_trajectory_mean(trajectories)
    all_feats = normalize(all_feats, norm='l2')
    ###################################################################
    #plot_utils.plot_trajectories3D(trajectories, stroke_names)
    
    nclusters = [2]     # list of no of clusters
    threshold = 50000
    
    ###################################################################
    # Pre-processing : Forming the dataframe with trajectories as rows
    
    # converting list of list to list of stroke trajectories
    X = [stroke for vid_strokes in trajectories for stroke in vid_strokes]
    row_names = [stroke for vid_strokes in stroke_names for stroke in vid_strokes]
    
    X = pd.DataFrame(X, index=row_names)
    
    # affinity mat with normalized rows (rows summed to 1, diag elements 0)
#    affinity_mat = spectral_utils.compute_affinity_matrix(X, similarity='euclidean')
#    affinity_mat = spectral_utils.compute_affinity_matrix_par(X, njobs=3, similarity='euclidean')
#    
#    np.save(os.path.join(destPath, "aff_mat_bins"+".npy"), affinity_mat)
    affinity_mat = np.load(os.path.join(destPath, "aff_mat_bins"+".npy"))
    
    # Threshold the affinity_mat n
    threshold = np.mean(affinity_mat)
    affinity_mat_thr = (affinity_mat < threshold) * 1.
    np.fill_diagonal(affinity_mat_thr, 0)
    
    for ncluster in nclusters:
        spectral_model = SpectralClustering(n_clusters=ncluster, affinity='precomputed')
        labels = spectral_model.fit_predict(affinity_mat)
        
        # returns a list of cluster label assignments for trajectories
        spectral_utils.spectral_plot(all_feats, labels, nclusters, threshold)
        # Write the strokes in their respective folders
        eval_of_clusters.evaluate(labels, row_names, dest_dir, DATASET)
        
    # plot the strokes as sequence of points
    plot_utils.plot_trajectories3D(trajectories, stroke_names)
            
def isConnected(adj_mat):
    '''Apply DFS to the graph to check if it is connected or not
    '''
    # traverse
    visited_list = dfs(adj_mat, 0, [])
    # return True is all nodes are visited else return False
    return len(visited_list) == adj_mat.shape[0]


def dfs(adj_mat, node, visited):
    if node not in visited:
        visited.append(node)        
        # Iterate on neighbours of current node that have not been visited
        for n in getUnvisitedNeighbours(adj_mat, node, visited): 
            dfs(adj_mat, n, visited)
            
    return visited


def getUnvisitedNeighbours(adj_mat, v, visited):
    
    unvisitedNeighbours = []
    neighbours = np.where(adj_mat[v, :] != 0)[0]
    # if there are no neighbours, then unvisited is empty
    for neighbour in neighbours:
        if neighbour not in visited:
            unvisitedNeighbours.append(neighbour)
    return unvisitedNeighbours
    
    
def find_edge_threshold(affinity_mat):
    '''Given an adjacency matrix of an undirected graph, determine the min threshold
    such that the graph is connected.
    
    '''
    min_val = int(np.min(affinity_mat[affinity_mat!=0]))
    max_val = int(np.max(affinity_mat))
    thresh_vals = np.linspace(min_val, max_val, 11).tolist()
    
    for i, thresh in enumerate(thresh_vals):
        affinity_mat_thr = (affinity_mat < thresh) * 1.
        if isConnected(affinity_mat_thr):
            break
        
    return i, thresh_vals
    

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
#            trajectories, stroke_names = extract_of.extract_all_feats(DATASET, \
#                                                            LABELS, bins, thresh)
            
            destPath = os.path.join(dest_dir, "bins_"+str(bins)+"_th_"+str(thresh))
#            if not os.path.exists(destPath):
#                os.makedirs(destPath)
#            with open(os.path.join(destPath, "trajectories.pkl"), 'wb') as fp:
#                pickle.dump(trajectories, fp)
#            with open(os.path.join(destPath, "stroke_names.pkl"), 'wb') as fp:
#                pickle.dump(stroke_names, fp)
            with open(os.path.join(destPath, "trajectories.pkl"), 'rb') as fp:
                trajectories = pickle.load(fp)
            with open(os.path.join(destPath, "stroke_names.pkl"), 'rb') as fp:
                stroke_names = pickle.load(fp)
            #all_feats = np.load(os.path.join(destPath, "feats_bins"+str(bins)+"_th_"+str(thresh)+".npy"))
            print("bin:{}, thresh:{} ".format(bins, thresh))
            all_feats = traj_utils.get_trajectory_mean(trajectories)
            all_feats = normalize(all_feats, norm='l2')
            ###################################################################
            #plot_utils.plot_trajectories3D(trajectories, stroke_names)
            
            nclusters = [2]     # list of no of clusters
            
            ###################################################################
            # Pre-processing : Forming the dataframe with trajectories as rows
            
            # converting list of list to list of stroke trajectories
            X = [stroke for vid_strokes in trajectories for stroke in vid_strokes]
            row_names = [stroke for vid_strokes in stroke_names for stroke in vid_strokes]
            
            X = pd.DataFrame(X, index=row_names)
            
            # affinity mat with normalized rows (rows summed to 1, diag elements 0)
            affinity_mat_orig = spectral_utils.compute_affinity_matrix(X, njobs=1, similarity='dtw')
##            
#            np.save(os.path.join(destPath, "aff_mat_l2_all_bins"+str(bins)+"_th_"+str(thresh)+".npy"), affinity_mat)
#            affinity_mat = np.load(os.path.join(destPath, "aff_mat_l2_all_bins"+str(bins)+"_th_"+str(thresh)+".npy"))
            g_mean, g_sigma = norm.fit(affinity_mat_orig)
            affinity_mat = np.exp(- affinity_mat_orig ** 2 / (2. * g_sigma ** 2))
#            # Threshold the affinity_mat 
            th_idx, threshold_vals = find_edge_threshold(affinity_mat)
            threshold = threshold_vals[th_idx]
            for i in range(th_idx, len(threshold_vals)):
#                affinity_mat_thr = np.ones_like(affinity_mat)
#                affinity_mat_thr[affinity_mat == 0] = 0.
#                affinity_mat_thr[affinity_mat > threshold_vals[i]] = 0.
            
                # form the laplacian matrix L = Diagonal matrix - Adjacency matrix
                lap_mat = np.copy(affinity_mat)
                lap_mat = -lap_mat
                np.fill_diagonal(lap_mat, np.sum(affinity_mat, axis=0))
                eVals, eVecs = np.linalg.eig(lap_mat)
                # roundoff the values to prevent floating point epsilon errors
                eVals, eVecs = np.round(eVals, 8), np.round(eVecs, 8)
                # sort in ascending order
                idx = eVals.argsort()
                eVals = eVals[idx]
                eVecs = eVecs[:, idx]
                
                plot_utils.visualize_evecs(eVals, eVecs)
                break
            
            
            for ncluster in nclusters:
                spectral_model = SpectralClustering(n_clusters=ncluster, 
                                                    affinity='precomputed',
                                                    random_state=0)
                labels = spectral_model.fit_predict(affinity_mat)
                
                # returns a list of cluster label assignments for trajectories
                spectral_utils.spectral_plot(all_feats, labels, nclusters, threshold)
                # Write the strokes in their respective folders
                eval_of_clusters.evaluate(labels, row_names, destPath, DATASET)
                
            spectral_model_3 = SpectralClustering(n_clusters=3, 
                                                    affinity='precomputed',
                                                    random_state=0)
            labels_3 = spectral_model_3.fit_predict(affinity_mat)
            acc_values, perm_tuples, gt_list, pred_list = \
            eval_of_clusters.get_accuracy(all_feats, labels_3, ANNOTATION_FILE, \
                                          DATASET, LABELS)
            acc_perc = [sum(k)/all_feats.shape[0] for k in acc_values]
            eval_of_clusters.evaluate(labels_3, row_names, destPath, DATASET)
            
            

            #best_acc[i,j] = max(acc_perc)
            bins_vals.append(bins)
            thresh_vals.append(thresh)
            #best_acc.append(max(acc_perc))
            best_indx = acc_perc.index(max(acc_perc))
            print("Max Acc. : ", max(acc_perc))
            print("Acc values : ", acc_perc)
            print("Acc values : ", acc_values)
            print("perm_tuples : ", perm_tuples)
            
            
#            # Building the clustering model 
#            spectral_model_rbf = SpectralClustering(n_clusters = 3, affinity ='rbf') 
#            # Training the model and Storing the predicted cluster labels 
#            labels_rbf = spectral_model_rbf.fit_predict(X_principal)
#            
#            # Building the clustering model 
#            spectral_model_nn = SpectralClustering(n_clusters = 3, affinity ='nearest_neighbors') 
#              
#            # Training the model and Storing the predicted cluster labels 
#            labels_nn = spectral_model_nn.fit_predict(X_principal) 
                
    plot_utils.plot_trajectories3D(trajectories, stroke_names)
    

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
            
            row_names = [stroke for vid_strokes in stroke_names for stroke in vid_strokes]
            
            sse_list = []
            x_range = list(range(1, 10))
            print("Clustering for 50 clusters ... ")
            for i in x_range:
                km = traj_utils.kmeans(all_feats, i)
                labels = km.labels_
                sse_list.append(km.inertia_)
                #print("labels_", labels)
            
            plot_utils.plot_sse_score(sse_list, x_range, destPath)
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
            #best_acc.append(max(acc_perc))
            best_indx = acc_perc.index(max(acc_perc))
            print("Max Acc. : ", max(acc_perc))
            print("Acc values : ", acc_perc)
            print("Acc values : ", acc_values)
            print("perm_tuples : ", perm_tuples)
            
            if bins>2:
                pca_flows = traj_utils.apply_PCA(all_feats)
            else:
                pca_flows = all_feats
            plot_utils.plot_clusters(pca_flows, labels, perm_tuples[best_indx], \
                                     destPath, 'cluster_pca_ordered.png')
            
            pca_flows = traj_utils.apply_tsne(all_feats)
            plot_utils.plot_clusters(pca_flows, labels, perm_tuples[best_indx], \
                                     destPath, 'cluster_tsne_ordered.png')
            
            eval_of_clusters.evaluate(labels, row_names, destPath, DATASET)
            
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
#    get_autoenc_spectral("res_autoenc_seq"+str(SEQ_SIZE))
    get_autoenc_kmeans("res_autoenc_gru_seq"+str(SEQ_SIZE))    
#    get_hoof_spectral(spect_dest)
#    get_hoof_kmeans(kmeans_dest)
    

    