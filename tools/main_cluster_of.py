#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 03:02:08 2020

@author: arpan

@Description: main file for clustering of optical flow from strokes.
"""

import _init_paths

import os
import seaborn as sns
import numpy as np
import pandas as pd
from utils import extract_of
from utils import cluster_of_utils as clus_utils
from evaluation import eval_of_clusters

from sklearn.preprocessing import normalize
sns.set()


c3dWinSize = 22
# Local Paths
LABELS = "/home/arpan/VisionWorkspace/Cricket/scripts/supporting_files/sample_set_labels/sample_labels_shots/ICC WT20"
DATASET = "/home/arpan/VisionWorkspace/VideoData/sample_cricket/ICC WT20"
#c3dFC7FeatsPath = "/home/arpan/VisionWorkspace/Cricket/localization_finetuneC3D/c3dFinetuned_feats_"+str(c3dWinSize)
# Server Paths
if os.path.exists("/opt/datasets/cricket/ICC_WT20"):
    LABELS = "/home/arpan/VisionWorkspace/shot_detection/supporting_files/sample_set_labels/sample_labels_shots/ICC WT20"
    DATASET = "/opt/datasets/cricket/ICC_WT20"
    #c3dFC7FeatsPath = "/home/arpan/VisionWorkspace/localization_finetuneC3D/c3dFinetuned_feats_"+str(c3dWinSize)

ANNOTATION_FILE = "stroke_labels.txt"
flow_numpy_path = 'flow_numpy_files'
RESULTS_DIR = 'res_OF_clusters'


if __name__ == '__main__':
    
    bin_counts = list(range(2, 21, 2))
    thresh_list = list(range(2, 30, 3))
    
    bins_vals = []
    thresh_vals = []
    best_acc = []
    
    for bins in bin_counts:
        for thresh in thresh_list:
            print(60*'*')
            #all_feats = extract_of.extract_all_feats(DATASET, LABELS, bins, thresh)
            #all_feats[np.isinf(all_feats)] = 0
            destPath = os.path.join(RESULTS_DIR, "bins_"+str(bins)+"_th_"+str(thresh))
            #if not os.path.exists(destPath):
            #    os.makedirs(destPath)
            #np.save(os.path.join(destPath, "feats_bins"+str(bins)+"_th_"+str(thresh)+".npy"), all_feats)
            all_feats = np.load(os.path.join(destPath, "feats_bins"+str(bins)+"_th_"+str(thresh)+".npy"))
            print("bin:{}, thresh:{} ".format(bins, thresh))
            all_feats = normalize(all_feats, norm='l2')
            ###################################################################################
            #all_feats = np.load(os.path.join("flow_shot", "norm_train_feats_50.npy"))
            
            sse_list = []
            x_range = list(range(1, 50))
            print("Clustering for 50 clusters ... ")
            for i in x_range:
                km = clus_utils.kmeans(all_feats, i)
                labels = km.labels_
                sse_list.append(km.inertia_)
                #print("labels_", labels)
            
            clus_utils.plot_sse_score(sse_list, x_range, bins, thresh)
            print("Final clustering for 3 clusters... \n")
            
            km = clus_utils.kmeans(all_feats, 3)
            labels = km.labels_
    
            acc_values, perm_tuples, gt_list, pred_list = \
                eval_of_clusters.get_accuracy(all_feats, labels, ANNOTATION_FILE)
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
                pca_flows = clus_utils.apply_PCA(all_feats)
            else:
                pca_flows = all_feats
            clus_utils.plot_clusters(pca_flows, labels, perm_tuples[best_indx], bins, thresh, 'cluster_pca_ordered.png')
            
            pca_flows = clus_utils.apply_tsne(all_feats)
            clus_utils.plot_clusters(pca_flows, labels, perm_tuples[best_indx], bins, thresh, 'cluster_tsne_ordered.png')
            
            #eval_of_clusters.evaluate(labels, bins, thresh)
            
    
    df = pd.DataFrame({"Bins(w)": bins_vals, "Magnitude Threshold (Pixels)":thresh_vals, "Accuracy(percent)":best_acc})
    df = df.pivot("Bins(w)", "Magnitude Threshold (Pixels)", "Accuracy(percent)")
    ax = sns.heatmap(df, vmin=0., vmax=1., annot=True, fmt='.2f', cmap="YlGnBu")
    
    normal_heat = sns.heatmap(df, vmin=0., vmax=1., annot=True, fmt='.2f')
    normal_heat.figure.savefig("normal_heat_bins10_thresh10.png")
    for i, g in enumerate(gt_list):
        print(g, pred_list[i])
    dbs = clus_utils.dbscan(all_feats)
    print(dbs.labels_)