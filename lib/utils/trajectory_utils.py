#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 03:09:43 2020

@author: arpan

@Description: Optical Flow Feature clustering and supporting functions
"""


import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN

def get_trajectory_mean(trajectories):
    all_feats = None
    for vid_strokes in trajectories:
        for stroke in vid_strokes:
            #np.zeros((stroke[0].shape[0]))
            stroke_feats = np.vstack(stroke)
            sum_feats = np.sum(stroke_feats, axis=0)/(stroke_feats.shape[0]+1)
            if all_feats is None:
                all_feats = sum_feats
            else:
                all_feats = np.vstack((all_feats, sum_feats))
    return all_feats

def apply_PCA(flows, dims=2):
    '''
    Function to apply PCA to high dimensional data. The target dimensions are specified.
    Parameters:
    -----------
    flows : np.array
        matrix to size (nStrokes, nbins) , nbins is the source no of dimensions
    dims : int
        target number of dimensions (<=nbins)
    
    Returns:
    --------
    np.array of size (nStrokes, dims)
    '''
    pca = PCA(n_components=dims)
    pca.fit(flows)
    flows_pca = pca.transform(flows)
    return flows_pca

def apply_tsne(flows, dims=2):
    '''
    Function to apply TSNE to high dimensional data. The target dimensions are specified.
    Parameters:
    -----------
    flows : np.array
        matrix to size (nStrokes, nbins) , nbins is the source no of dimensions
    dims : int
        target number of dimensions (<=nbins)
    
    Returns:
    --------
    np.array of size (nStrokes, dims)
    '''
    return TSNE(n_components=dims).fit_transform(flows)

def kmeans(flows, clusters=4):
    '''
    Function to cluster a matrix data (a data point is a row) into given no. of clusters.
    Parameters:
    -----------
    flows : np.array 
        normalized matrix for size (nStrokes, nbins) , nbins is the no. of dimensions.
    clusters : int
        no. of clusters target clusters.
    
    Returns:
    --------
    KMeans object after clustering is performed with random_state of 0
    '''
    km = KMeans(n_clusters=clusters, algorithm='elkan', random_state=0)
    km.fit(flows)
    return km

def dbscan(flows, epsilon=0.05, min_samp=10):
    '''
    Function to cluster a matrix data (a data point is a row) into given no. of clusters.
    Using DBSCAN.
    Parameters:
    -----------
    flows : np.array 
        normalized matrix for size (nStrokes, nbins) , nbins is the no. of dimensions.
    epsilon : float
        distance for acquiring neighbouring data points into a cluster
    min_samp : int
        minimum no. of samples to be assigned for a group to be called a cluster, 
        else it is an outlier.
    
    Returns:
    --------
    DBSCAN object after clustering is performed
    '''
    db = DBSCAN(eps=0.05, min_samples=10).fit(flows)
    return db

