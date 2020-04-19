#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 03:09:43 2020

@author: arpan

@Description: Optical Flow Feature clustering and supporting functions
"""

import os
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt


def apply_PCA(flows):
    pca = PCA(n_components=2)
    pca.fit(flows)
    flows_pca = pca.transform(flows)
    return flows_pca

def apply_tsne(flows):
    return TSNE(n_components=2).fit_transform(flows)

def kmeans(flows, clusters=4):
    km = KMeans(n_clusters=clusters, algorithm='elkan', random_state=0)
    km.fit(flows)
    return km

def dbscan(flows, clusters=4):
    db = DBSCAN(eps=0.05, min_samples=10).fit(flows)
    return db


def plot_clusters(data, labels, best_tuple, bins, thresh, plotname):
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
    if plotname=='cluster_pca_ordered.png':
        plt.xlim((-1,1))
        plt.ylim((-1,1))
    plt.savefig(os.path.join(RESULTS_DIR, "bins_"+str(bins)+"_th_"+str(thresh), str(n_clusters)+'_'+plotname))
    plt.show()

def plot_sse_score(sse_list, x, bins, thresh):
    plt.figure(2)
    plt.xlabel("#Clusters")
    plt.ylabel("SSE")
    plt.plot(x, sse_list, label='sse score', c='r')
    plt.title('SSE score Vs No. of clusters')
    plt.savefig(os.path.join(RESULTS_DIR, "bins_"+str(bins)+"_th_"+str(thresh), 'sse_plot.png'))
    plt.show()
    
    
def visualize_of(mag, ang, frameNo):
    c = 3
    h, w = mag.shape
    hsv = np.zeros((h, w, c), dtype=np.uint8)
    hsv[...,1] = 255
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    plt.figure()
    plt.imshow(bgr)
    cv2.imwrite(os.path.join(flow_numpy_path, "of_"+str(frameNo)+".png"), bgr)
    #plt.show()