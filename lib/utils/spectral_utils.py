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
from sklearn.preprocessing import StandardScaler, normalize 
from sklearn.decomposition import PCA 
from matplotlib import pyplot as plt
from fastdtw import fastdtw
from joblib import Parallel, delayed
from hausdorff import hausdorff_distance


def compute_affinity_matrix(X, njobs = 1, similarity="euclidean"):
    '''
    Compute the Affinity Matrix (Adjacency matrix) for input to the Spectral 
    Clustering function.
    
    Parameters:
    -----------
    X : pd.DataFrame
        DataFrame of size (nStrokes, nMaxFeats). Rows have vectors of trajectory points
        Many will be NA cells due to different stroke lengths.
    njobs : int
        Parallelize over these many cores
    similarity : str
        type of similarity to be used for comparing two different stroke trajectories
        Types : 
            'euclidean' : match the vectors based on sliding window L2 distances
            'dtw' : Dynamic Time Warping based matching
            ''
    
    Returns:
    --------
    affinity_mat: 2D Numpy array with similarity values as given in the parameter
    
    '''

    if njobs > 1:
        print("Parallel computation of similarity with nJobs : {} ...".format(njobs))    
        batch_out = Parallel(n_jobs=njobs)(delayed(compute_pair_similarity) \
                         (X.iloc[i, :], X.iloc[j, :], similarity) \
                         for i in range(len(X)) \
                         for j in range(len(X)) if i<j)
    
    # form matrix 
    affinity_mat = np.zeros((len(X), len(X)))
    k = 0
    for i in range(len(X)):
        print("Stroke {} ...".format(i+1))
        for j in range(len(X)):
            if i<j:
                if njobs > 1:
                    affinity_mat[i,j]= batch_out[k]
                    k +=1
                else:
                    affinity_mat[i,j] = compute_pair_similarity(X.iloc[i,:], \
                                        X.iloc[j,:], similarity)
            elif i>j:
                affinity_mat[i,j] = affinity_mat[j,i]
    
    # row-wise normalized, sum to 1
    #affinity_mat = affinity_mat/affinity_mat.sum(axis=1, keepdims=1)
    
    return affinity_mat

def normalized_l2(t1, t2):
    '''Function takes two sequence of vectors and finds the Euclidean distance between
    corresponding vectors. Summed and normalized by length of sequence
    
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
    
    t1 = np.asarray(t1.tolist())
    t2 = np.asarray(t2.tolist())
    # return dot product, try cosine similarity
    similarity = 0
    for i in range(t1.shape[0]):
#        similarity += t1[i].dot(t2[i])
        similarity += distance.euclidean(t1[i, :], t2[i, :])
    # divide by vector size to normalize.
    return similarity/(t1.shape[0]*t1.shape[1])

def compute_hausdorff_similarity(t1, t2):
    
    # convert Series object to 2D matrix 
    t1 = np.asarray(t1.tolist())
    t2 = np.asarray(t2.tolist())
    return hausdorff_distance(t1, t2, distance="euclidean")
    

def compute_dtw_similarity(t1, t2):
    t1 = t1.dropna()
    t2 = t2.dropna()
    
    t1 = np.stack(t1)
    t2 = np.stack(t2)
    
    traj_dist, _ = fastdtw(t1, t2, dist = distance.euclidean)
    return traj_dist

def compute_pair_similarity(t1, t2, similarity="euclidean"):
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
    if similarity == 'euclidean':
        # maybe mean of first and last window placement for partial mapping
        #for i in range(max_len - min_len +1):
        sim1 = normalized_l2(max_t.iloc[:min_len], min_t)
        sim2 = normalized_l2(max_t.iloc[(max_len-min_len):], min_t)
        similarities.append((sim1 + sim2) / 2.)
        #    break  # comment out to run a sliding window approach and return max or min
    elif similarity == 'dtw':
        similarities.append(compute_dtw_similarity(t1, t2) / (min_len*min_t[0].shape[0]) )
    elif similarity == 'hausdorff':
        similarities.append(compute_hausdorff_similarity(t1, t2) / (min_len*min_t[0].shape[0]))
        
        
    return similarities[0]  # take min if similarity is L2 Distance
    # Each t is a Series object with t1.shape (120,) and t2.shape (87,)
    # Each element of Series is numpy vector of bins size
    # run over the two array lists 


def spectral_plot(all_feats, labels, nclusters, threshold):
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
#    return labels
