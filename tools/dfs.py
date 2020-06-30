#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 15:41:58 2020

@author: arpan
"""

import numpy as np


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
    thresh_vals = np.linspace(0, int(np.max(affinity_mat)), 11).tolist() 
    
    for thresh in thresh_vals:
        affinity_mat_thr = (affinity_mat < thresh) * 1.
        if isConnected(affinity_mat_thr):
            break
        
    return thresh