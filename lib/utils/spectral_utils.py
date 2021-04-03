#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 03:51:10 2020

@author: arpan

@Description: Utility functions for Spectral clustering. Similarity/Distance metrics
"""

import os
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


def isConnected(adj_mat):
    '''Apply DFS to the graph to check if it is connected or not.
    Start traversal from a node and check if all the nodes have been visited.
    
    Parameters:
    -----------
    adj_mat : np.array(NStrokes, NStrokes)
        adjacency matrix for unweighted graph. Obtained by thresholding the
        similarity/distance matrix. 
    
    Returns:
    --------
    boolean value representing whether the graph is connected or not.
    '''
    # traverse
    visited_list = dfs(adj_mat, 0, [])
    # return True is all nodes are visited else return False
    return len(visited_list) == adj_mat.shape[0]

def dfs(adj_mat, node, visited):
    '''Apply DFS to the graph, starting from a node and return a list of visited 
    nodes for that component. The list of visited nodes can be used to find if the 
    graph is connected or not.
    
    Parameters:
    -----------
    adj_mat : np.array(NStrokes, NStrokes)
        adjacency matrix for undirected unweighted graph. 
    node : int
        starting node no. (default can be 0)
    visited : list
        Initially empty. It is populated with traversed nodes.
    
    Returns:
    --------
    visited nodes list (list of int)
    '''
    if node not in visited:
        visited.append(node)        
        # Iterate on neighbours of current node that have not been visited
        for n in getUnvisitedNeighbours(adj_mat, node, visited): 
            dfs(adj_mat, n, visited)
            
    return visited


def getUnvisitedNeighbours(adj_mat, v, visited):
    '''Given an adjacency matrix, node v, and list of already visited nodes,
    create a list of unvisited neighbours of v and return the list.
    
    Parameters:
    -----------
    adj_mat : np.array(NStrokes, NStrokes)
        adjacency matrix for undirected unweighted graph. 
    v : int
        current node that is being visited.
    visited : list of int
        contains node nos. that have already been visited.
        
    Returns:
    --------
    unvisitedNeighbours : list of int
        Node indices that have not been visited and are neighours of v
    '''
    unvisitedNeighbours = []
    neighbours = np.where(adj_mat[v, :] != 0)[0]
    # if there are no neighbours, then unvisited is empty
    for neighbour in neighbours:
        if neighbour not in visited:
            unvisitedNeighbours.append(neighbour)
    return unvisitedNeighbours
    
    
def find_edge_threshold(affinity_mat):
    '''Given an affinity matrix for an undirected graph, determine the min threshold
    such that the graph is connected.
    Parameters:
    -----------
    affinity_mat : np.array(NStrokes, NStrokes)
        similarity/distances for stroke pairs. Symmetric matrix. 
        
    Returns:
    --------
    i : int
        index for threshold value
    threshold_vals : np.array(10)
        list 10 equal sized bins in range (min_val, max_val), where min_val, max_val
        are the maximum affinity value and minimum (>0) affinity values in aff matrix.
    
    '''
    min_val = np.min(affinity_mat[affinity_mat!=0])
    max_val = np.max(affinity_mat)
    thresh_vals = np.linspace(max_val, min_val, 11).tolist()
    
    for i, thresh in enumerate(thresh_vals):
        affinity_mat_thr = (affinity_mat > thresh) * 1.
        if isConnected(affinity_mat_thr):
            break
        
    return i, thresh_vals

def compute_affinity_matrix(X, njobs = 1, similarity="euclidean", lcs_eps=0.6, lcs_delta=13):
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
            'hausdorff' : Hausdorff Distance
            ''
    
    Returns:
    --------
    affinity_mat: 2D Numpy array with similarity values as given in the parameter
    
    '''

    if njobs > 1:
        print("Parallel computation of similarity with nJobs : {} ...".format(njobs))    
        batch_out = Parallel(n_jobs=njobs)(delayed(compute_pair_similarity) \
                         (X.iloc[i, :], X.iloc[j, :], similarity, lcs_eps, lcs_delta) \
                         for i in range(len(X)) \
                         for j in range(len(X)) if i<j)
    
    # form matrix 
    affinity_mat = np.zeros((len(X), len(X)))
    k = 0
    for i in range(len(X)):
#        print("Stroke {} ...".format(i+1))
        for j in range(len(X)):
            if i<j:
                if njobs > 1:
                    affinity_mat[i,j]= batch_out[k]
                    k +=1
                else:
                    affinity_mat[i,j] = compute_pair_similarity(X.iloc[i,:], \
                                        X.iloc[j,:], similarity, lcs_eps, lcs_delta)
            elif i>j:
                affinity_mat[i,j] = affinity_mat[j,i]
    
    # row-wise normalized, sum to 1
    #affinity_mat = affinity_mat/affinity_mat.sum(axis=1, keepdims=1)
    
    return affinity_mat

def compute_pair_similarity(t1, t2, similarity="euclidean", eps=0.6, delta=13):
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
        for i in range(max_len - min_len +1):
            similarities.append(normalized_l2(max_t.iloc[i:(i+min_len)], min_t))
        similarities = [np.mean(np.asarray(similarities)).tolist()]
#        sim1 = normalized_l2(max_t.iloc[:min_len], min_t)
#        sim2 = normalized_l2(max_t.iloc[(max_len-min_len):], min_t)
#        similarities.append((sim1 + sim2) / 2.)
#        similarities.append(sim2)
        #    break  # comment out to run a sliding window approach and return max or min
    elif similarity == 'dtw':
        similarities.append(compute_dtw_similarity(t1, t2) / (min_len*min_t[0].shape[0]) )
    elif similarity == 'hausdorff':
        similarities.append(compute_hausdorff_similarity(t1, t2) / (min_len*min_t[0].shape[0]))
    elif similarity == 'corr':
        similarities.append(compute_svd_similarity(t1, t2))
    elif similarity == 'temp_corr':
        similarities.append(compute_temp_svd_similarity(t1, t2))
    elif similarity == 'lcss':
        similarities.append(compute_lcss_similarity(t1, t2, eps, delta))
        
    return similarities[0]  # take min if similarity is L2 Distance
    # Each t is a Series object with t1.shape (120,) and t2.shape (87,)
    # Each element of Series is numpy vector of bins size
    # run over the two array lists 

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

def compute_svd_similarity(t1, t2):
    t1 = np.asarray(t1.tolist())
    t2 = np.asarray(t2.tolist())
    
#    t1_u, t1_sigma, t1_v = np.linalg.svd(t1)
#    t2_u, t2_sigma, t2_v = np.linalg.svd(t2)
#    
#    return distance.euclidean(t1_sigma, t2_sigma)
    t1_corr = np.dot(t1.transpose(), t1)
    t2_corr = np.dot(t2.transpose(), t2)
    similarity = 0
    for i in range(t1_corr.shape[0]):
#        similarity += t1[i].dot(t2[i])
        similarity += distance.euclidean(t1_corr[i, :], t2_corr[i, :])
    # divide by vector size to normalize.
    return similarity/(t1_corr.shape[0]*t1_corr.shape[1])

def compute_temp_svd_similarity(t1, t2):
    t1 = np.asarray(t1.tolist())
    t2 = np.asarray(t2.tolist())
    
#    t1_u, t1_sigma, t1_v = np.linalg.svd(t1)
#    t2_u, t2_sigma, t2_v = np.linalg.svd(t2)
#    
#    return distance.euclidean(t1_sigma, t2_sigma)
    L1 = temp_laplacian_mat(t1.shape[0], 15)
    L2 = temp_laplacian_mat(t2.shape[0], 15)
    t1_corr = np.dot(np.dot(t1.transpose(), L1), t1)
    t2_corr = np.dot(np.dot(t2.transpose(), L2), t2)
    similarity = 0
    for i in range(t1_corr.shape[0]):
#        similarity += t1[i].dot(t2[i])
        similarity += distance.euclidean(t1_corr[i, :], t2_corr[i, :])
    # divide by vector size to normalize.
    return similarity/(t1_corr.shape[0]*t1_corr.shape[1])

def temp_laplacian_mat(dim, width=3):
    '''Create a temporal laplacian matrix L as follows:
        L = D - W , where W is square matrix with diag(W)=0 and temporal associations
        defined as binary/float values 
    dim : int
        size of the temporal dimension for a stroke.
    '''
    W = np.zeros((dim, dim))
    assert dim >= width, "Laplacian matrix size should be more than temporal width"
    for w in range(1, 1+(width//2)) :
        i = list(range(dim - w))
        j = list(range(w, dim))
        W[i, j] = 1
        W[j, i] = 1
        
    D_bar = np.diag(np.sum(W, axis=1))
    L = D_bar - W
    return L
    
def compute_lcss_similarity(t1, t2, eps, delta):
    t1 = np.asarray(t1.tolist())
    t2 = np.asarray(t2.tolist())
    C = LCS(t1, t2, eps, delta)
    # find Distance
    D = 1 - (C[-1][-1] / max(t1.shape[0], t2.shape[0]))
    return D

def LCS(X, Y, eps, delta):
    m = len(X)
    n = len(Y)
    # An (m+1) times (n+1) matrix
    C = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if distance.euclidean(X[i-1], Y[j-1]) < eps and abs(i-j) < delta:  # X[i-1] == Y[j-1] 
                C[i][j] = C[i-1][j-1] + 1
            else:
                C[i][j] = max(C[i][j-1], C[i-1][j])
    return C

#def backTrack(C, X, Y, i, j):
#    if i == 0 or j == 0:
#        return ""
#    elif X[i-1] == Y[j-1]:
#        return backTrack(C, X, Y, i-1, j-1) + X[i-1]
#    else:
#        if C[i][j-1] > C[i-1][j]:
#            return backTrack(C, X, Y, i, j-1)
#        else:
#            return backTrack(C, X, Y, i-1, j)

def unit_vector(vector):
    """     https://stackoverflow.com/a/13849249/71522
    Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def spectral_plot(all_feats, labels, plot_dir):
    # Preprocessing the data to make it visualizable 
    nclusters = max(labels) + 1
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
    
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

    # Building the colour vector for each data point 
#    cvec = [colors[label] for label in labels]
    #cvec_nn = [colours[label] for label in labels_nn] 
    scatter_objs, labs = [], []
    plt.figure()
    # Plotting the clustered scatter plot 
    for i in range(nclusters):
        scatter_objs.append(plt.scatter(X_principal['P1'].iloc[labels==i], 
                                        X_principal['P2'].iloc[labels==i], 
                                        marker='.',
                                        color = colors[i]))
        labs.append("Label {}".format(i))
#        y = plt.scatter(X_principal['P1'], X_principal['P2'], color ='y'); 
#        g = plt.scatter(X_principal['P1'], X_principal['P2'], color ='g'); 
      
#    plt.figure(figsize =(9, 9)) 
#    plt.scatter(X_principal['P1'], X_principal['P2'], c = cvec) 
    plt.legend(scatter_objs, labs, loc='upper right')
    plt.savefig(os.path.join(plot_dir, str(nclusters)+'_'+'pca_spectral.png'), dpi=300)
#    plt.show()
    plt.close()