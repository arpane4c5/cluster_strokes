#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 03:51:10 2020

@author: arpan

@Description: Utility functions for Spectral clustering. Similarity/Distance metrics
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial import distance
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler, normalize 
from sklearn.decomposition import PCA 
from matplotlib import pyplot as plt

def compute_affinity_matrix(trajectories):
    '''
    Compute the Affinity Matrix (Adjacency matrix) for input to the Spectral 
    Clustering function.
    
    Parameters:
    -----------
    trajectories : list of list of float vectors
        stroke features for all the vids. Each sublist is for a video, which contains
        list of stroke features. Size is no. of videos.
    threshold : float
        threshold the unnormalized values of edge distance computed.
        
    Returns:
    --------
    affinity_mat: 2D Numpy array with 
    
    '''
    X = [stroke for vid_strokes in trajectories for stroke in vid_strokes]
    
    #for row in range(len(X)):
    
#    # Convert to list and Add padding with with 'inf' values
#    max_len = max([len(stroke) for stroke in X])
    
    X = pd.DataFrame(X)
    
    # form matrix 
    affinity_mat = np.zeros((len(X), len(X)))
    
    for i in range(len(X)):
        print("Stroke {} ...".format(i+1))
        for j in range(len(X)):
            if i<j:
                affinity_mat[i,j]=compute_pair_similarity(X.iloc[i,:], X.iloc[j,:])
            elif i>j:
                affinity_mat[i,j] = affinity_mat[j,i]
    
    # row-wise normalized, sum to 1
    #affinity_mat = affinity_mat/affinity_mat.sum(axis=1, keepdims=1)
    
    return affinity_mat

def similarity_function(t1, t2):
    '''Function takes two vectors and finds the similarity between them, such as 
    Euclidean distance etc.
    
    Parameters:
    -----------
    t1 and t2 : 1D numpy arrays
        1D arrays of equal sizes representing feature points of trajectory sequences
        at time t1 and t2, respectively, in the trajectories.
        
    Returns:
    --------
    float : value representing a similarity/distance between the two vectors.
    '''
    assert t1.shape[0] == t2.shape[0], "Size doesn't match {} : {}".format(\
                   t1.shape[0], t2.shape[0])
    
    # return dot product, try cosine similarity
    similarity = 0
    for i in range(t1.shape[0]):
#        similarity += t1[i].dot(t2[i])
        similarity += distance.euclidean(t1.iloc[i], t2.iloc[i])
    # divide by vector size to normalize.
    return similarity/t1.shape[0]
    

def compute_pair_similarity(t1, t2):
    '''
    Pass two trajectories (as pandas Series) and get similarity between the two. 
    The similarity function to produce high values for similar trajectories and 
    low values for dissimilar trajectories
    '''    
    t1 = t1.dropna()
    t2 = t2.dropna()
    if t1.shape[0] > t2.shape[0]:
        max_t, min_t = t1, t2
    else:
        max_t, min_t = t2, t1
        
    max_len, min_len = max_t.shape[0], min_t.shape[0]
    similarities = []
    for i in range(max_len - min_len +1):
        similarities.append(similarity_function(max_t.iloc[i:(i+min_len)], min_t))
        
    return min(similarities)  # take min if similarity is L2 Distance
    # Each t is a Series object with t1.shape (120,) and t2.shape (87,)
    # Each element of Series is numpy vector of bins size
    # run over the two array lists 


def spectral(all_feats, trajectories, nclusters, threshold):
    # Preprocessing the data to make it visualizable 
      
    # Scaling the Data 
    scaler = StandardScaler() 
    X_scaled = scaler.fit_transform(all_feats) 
      
    # Normalizing the Data 
    X_normalized = normalize(X_scaled) 
      
    # Converting the numpy array into a pandas DataFrame 
    X_normalized = pd.DataFrame(X_normalized) 
      
    # Reducing the dimensions of the data 
    pca = PCA(n_components = 2) 
    X_principal = pca.fit_transform(X_normalized) 
    X_principal = pd.DataFrame(X_principal) 
    X_principal.columns = ['P1', 'P2'] 
      
    X_principal.head() 

    # affinity mat with normalized rows (rows summed to 1, diag elements 0)
    affinity_mat = compute_affinity_matrix(trajectories)
    
    # Threshold the affinity_mat n
    affinity_mat_thr = (affinity_mat < threshold) * 1.
    
    for ncluster in nclusters:
        spectral_model = SpectralClustering(n_clusters=ncluster, affinity='precomputed')
        labels = spectral_model.fit_predict(affinity_mat)
    
#    # Building the clustering model 
#    spectral_model_rbf = SpectralClustering(n_clusters = 3, affinity ='rbf') 
#    
#    # Training the model and Storing the predicted cluster labels 
#    labels_rbf = spectral_model_rbf.fit_predict(X_principal)
#    
#    
#    # Building the clustering model 
#    spectral_model_nn = SpectralClustering(n_clusters = 3, affinity ='nearest_neighbors') 
#      
#    # Training the model and Storing the predicted cluster labels 
#    labels_nn = spectral_model_nn.fit_predict(X_principal) 

    
    # Building the label to colour mapping 
    colours = {} 
    colours[0] = 'b'
    colours[1] = 'y'
    colours[2] = 'g'
      
    # Building the colour vector for each data point 
    cvec = [colours[label] for label in labels] 
    #cvec_nn = [colours[label] for label in labels_nn] 
      
    # Plotting the clustered scatter plot 
      
    b = plt.scatter(X_principal['P1'], X_principal['P2'], color ='b'); 
    y = plt.scatter(X_principal['P1'], X_principal['P2'], color ='y'); 
    g = plt.scatter(X_principal['P1'], X_principal['P2'], color ='g'); 
      
#    plt.figure(figsize =(9, 9)) 
    plt.figure() 
    plt.scatter(X_principal['P1'], X_principal['P2'], c = cvec) 
    plt.legend((b, y, g), ('Label 0', 'Label 1', 'Label 2')) 
    plt.show() 
    
#    plt.figure()
#    plt.scatter(X_principal['P1'], X_principal['P2'], c = cvec_nn) 
#    plt.legend((b, y, g), ('Label 0', 'Label 1', 'Label 2')) 
#    plt.show() 
    return labels
