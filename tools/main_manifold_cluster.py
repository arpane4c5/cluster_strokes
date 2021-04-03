#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sept 12 03:02:08 2020

@author: arpan

@Description: main file for clustering global stroke feature trajectories (Highlights data).
"""

import _init_paths

import os
import seaborn as sns
import numpy as np
import pandas as pd
import pickle
from features.extract_hoof_feats import extract_stroke_feats
from features.extract_idt_feats import read_partition_feats
from features import extract_autoenc_feats as autoenc_feats
from utils import trajectory_utils as traj_utils
from utils import spectral_utils
from utils import plot_utils
from utils import autoenc_utils
from evaluation import eval_of_clusters

from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import normalize
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

sns.set(rc={'figure.figsize':(11.7,8.27)})
palette = sns.color_palette("bright", 10)

#sns.set()

# Local Paths
LABELS = "/home/arpan/VisionWorkspace/Cricket/scripts/supporting_files/sample_set_labels/sample_labels_shots/ICC WT20"
DATASET = "/home/arpan/VisionWorkspace/VideoData/sample_cricket/ICC WT20"
IDT_FEATS = "/home/arpan/VisionWorkspace/Cricket/StrokeAttention/logs/idt_strokes"
CLASS_IDS = "/home/arpan/VisionWorkspace/Cricket/cluster_strokes/configs/Class Index_Strokes.txt"
ANNOTATION_FILE_3C = "/home/arpan/VisionWorkspace/Cricket/CricketStrokeLocalizationBOVW/stroke_labels.txt"
ANNOTATION_FILE_5C = "/home/arpan/VisionWorkspace/Cricket/CricketStrokeLocalizationBOVW/shots_classes.txt"

# Server Paths
if os.path.exists("/opt/datasets/cricket/ICC_WT20"):
    LABELS = "/home/arpan/VisionWorkspace/shot_detection/supporting_files/sample_set_labels/sample_labels_shots/ICC WT20"
    DATASET = "/opt/datasets/cricket/ICC_WT20"    
    IDT_FEATS = "/home/arpan/DATA_Drive2/Cricket/Workspace/StrokeAttention/logs/idt_strokes"
    CLASS_IDS = "configs/Class Index_Strokes.txt"
    ANNOTATION_FILE_3C = "/home/arpan/DATA_Drive2/Cricket/Workspace/CricketStrokeLocalizationBOVW/stroke_labels.txt"
    ANNOTATION_FILE_5C = "/home/arpan/DATA_Drive2/Cricket/Workspace/CricketStrokeLocalizationBOVW/shots_classes.txt"

model_path = "checkpoints/autoenc_gru_resnet50_ep10_w6_Adam.pt"
SEQ_SIZE = 6
INPUT_SIZE = 2048
HIDDEN_SIZE = 32#64#1024
NUM_LAYERS = 2
BATCH_SIZE = 2*SEQ_SIZE # set to the number of images of a seqence # 36
traj_len = 25


def apply_kmeans(mean_feats, row_names, log_dir, nclasses=3, write_strokes=False):
    
    # clustering for a range of K values and plot SSE scores
    sse_list = []
    x_range = list(range(1, 10))
    print("KMeans Clustering for 10 clusters for SSE scores... ")
    for i in x_range:
        km = traj_utils.kmeans(mean_feats, i)
        labels = km.labels_
        sse_list.append(km.inertia_)
    # Plot SSE scores 
    plot_utils.plot_sse_score(sse_list, x_range, log_dir)
    
    ###########################################################################
    if nclasses == 3:
        annotation_path = ANNOTATION_FILE_3C
    elif nclasses == 5:
        annotation_path = ANNOTATION_FILE_5C
        
    print("KMeans Clustering for {} clusters... ".format(nclasses))
    km = traj_utils.kmeans(mean_feats, nclasses)
    labels = km.labels_

    acc_values, perm_tuples, gt_list, pred_list = \
        eval_of_clusters.get_accuracy(mean_feats, labels, annotation_path, \
                                      DATASET, LABELS)
    acc_perc = [sum(k)/mean_feats.shape[0] for k in acc_values]
    
#    best_acc.append(max(acc_perc))
    best_indx = acc_perc.index(max(acc_perc))
    print("Max Acc. : ", max(acc_perc))
    best_perm = perm_tuples[best_indx]
    print("best perm_tuple : ", best_perm)    
    
    acc = eval_of_clusters.create_confusion_matrix(best_perm, pred_list, gt_list, nclasses)
    
    ###########################################################################
    # Plot PCA, TSNE and write predicted strokes
    if mean_feats.shape[1] > 2:
        pca_flows = traj_utils.apply_PCA(mean_feats)
    else:
        pca_flows = mean_feats
    plot_utils.plot_clusters(pca_flows, labels, perm_tuples[best_indx], \
                             log_dir, 'km_pca.png')
    
    pca_flows = traj_utils.apply_tsne(mean_feats)
    plot_utils.plot_clusters(pca_flows, labels, perm_tuples[best_indx], \
                             log_dir, 'km_tsne.png')
    
    # Write sample strokes in folders representing clusters
    if write_strokes:
        eval_of_clusters.evaluate(labels, row_names, 
                                  os.path.join(log_dir, 'KM_C'+str(nclasses)),
                                  DATASET)
    return acc

def plot_manifold(data, labels, best_tuple, plot_dir, plotname):
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
    plt.savefig(os.path.join(plot_dir, str(n_clusters)+'_'+plotname), dpi=300)
#    plt.show()
    plt.close()
    

def apply_manifold_cluster(mean_feats, row_names, log_dir, nclasses=3, write_strokes=False):
    
    ###########################################################################
    # Explaining variances and choosing no. of principal components
    N = mean_feats.shape[1]
    pca = PCA(n_components=N)
    pca.fit(mean_feats)
    feats_pca = pca.transform(mean_feats)
    ex_var = pca.explained_variance_ratio_      # np.sum(ex_var) is 1
    cum_ex_var = [np.sum(ex_var[:i]) for i in range(1, ex_var.shape[0]+1) ]
    plot_utils.plot_pca_explained_variance(cum_ex_var, N, log_dir)
    ###########################################################################
    # Choosing no. of principal components 
    capture_perc_var = 0.9      # percentage variation to consider
    # No. of principal components that captures the required variation in data
    r = 1 + np.argmax(np.array(cum_ex_var) > capture_perc_var)
    print("Percentage of variation captured : {}".format(capture_perc_var))
    print("No. of principal components chosen : {}".format(r))
    
    ###########################################################################
    assert r >= 2, "There should be atleast 2 principal components (r >= 2)."
    # Applying PCA
    pca = PCA(n_components=r)
    pca.fit(mean_feats)
    feats_pca = pca.transform(mean_feats)
    # Applying TSNE
    p = 2 # IS p == r ?
    feats_tsne = TSNE(n_components=p).fit_transform(feats_pca)
    
    ###########################################################################
    # Find optimum epsilon for DBSCAN
    plot_utils.plot_optimum_eps(feats_tsne, log_dir) 
    # eps (optimal) = 1.5 (approx from graph)
    
    # Applying DBSCAN
#    eps = np.linspace(5, 1, 10)     # 
    eps = np.array([1.5])
    min_samples = range(30, 5, -2)
    for ep in eps.tolist():
        for min_samp in min_samples:
            db = DBSCAN(eps=ep, min_samples=min_samp).fit(feats_tsne)
            n_clust = len(set(db.labels_))
            if n_clust > 1:     # ignoring -1 labels
                print("eps : {:.4f} :: min_s : {} :: C : {}".format(ep, min_samp, n_clust))
    
    
    # generates 28 clusters
    db = DBSCAN(eps=1.5, min_samples=7).fit(feats_tsne)
    labs = db.labels_
    dbscan_points = feats_tsne[labs != -1]
    labs = labs[labs != -1]
    
    plot_utils.plot_dbscan_cls(dbscan_points, labs, log_dir)
    ###########################################################################
    
    
    ###########################################################################
    if nclasses == 3:
        annotation_path = ANNOTATION_FILE_3C
    elif nclasses == 5:
        annotation_path = ANNOTATION_FILE_5C
        
    print("KMeans Clustering for {} clusters... ".format(nclasses))
    km = traj_utils.kmeans(feats_tsne, nclasses)
#    km_pca = traj_utils.kmeans(feats_pca, nclasses)
    
#    km = GaussianMixture(n_components=nclasses, covariance_type='diag',
#                              random_state=128).fit(feats_tsne)
    if isinstance(km, GaussianMixture):
        labels = km.predict(feats_tsne)
    else:
        labels = km.labels_

    acc_values, perm_tuples, gt_list, pred_list = \
        eval_of_clusters.get_accuracy(mean_feats, labels, annotation_path, \
                                      DATASET, LABELS)
    acc_perc = [sum(k)/mean_feats.shape[0] for k in acc_values]
    
#    best_acc.append(max(acc_perc))
    best_indx = acc_perc.index(max(acc_perc))
    print("Max Acc. : ", max(acc_perc))
    best_perm = perm_tuples[best_indx]
    print("best perm_tuple : ", best_perm)
    
    acc = eval_of_clusters.create_confusion_matrix(best_perm, pred_list, gt_list, nclasses)
    
    ###########################################################################
    # Plot PCA, TSNE and write predicted strokes
#    if mean_feats.shape[1] > 2:
#        pca_flows = traj_utils.apply_PCA(mean_feats)
#    else:
#        pca_flows = mean_feats
    plot_utils.plot_clusters(feats_tsne, labels, perm_tuples[best_indx], \
                             log_dir, 'gmm_tsne.png')
    
#    pca_flows = traj_utils.apply_tsne(mean_feats)
#    plot_utils.plot_clusters(pca_flows, labels, perm_tuples[best_indx], \
#                             log_dir, 'km_tsne.png')
    
    # Write sample strokes in folders representing clusters
#    if write_strokes:
#        eval_of_clusters.evaluate(labels, row_names, 
#                                  os.path.join(log_dir, 'KM_C'+str(nclasses)),
#                                  DATASET)
    return acc
    
def apply_spectral(trajectories, row_names, mean_feats, log_dir, similarity='euclidean', 
                   nclasses=3, threshold=None, write_strokes=False):
    X = pd.DataFrame(trajectories, index=row_names)
    aff_mat_filepath = os.path.join(log_dir, "aff_mat_"+similarity+".npy")
    
    print("Spectral Clustering for {} clusters... ".format(nclasses))
    print("Similarity : ", similarity)
    print("Path : ", log_dir)
    if not os.path.isfile(aff_mat_filepath):
        # affinity mat with normalized rows (rows summed to 1, diag elements 0)
        affinity_mat = spectral_utils.compute_affinity_matrix(X, njobs=4, 
                                                              similarity=similarity)
        np.save(aff_mat_filepath, affinity_mat)
    affinity_mat = np.load(aff_mat_filepath)
    
    ###########################################################################
    # Use thresholding for affinity matrix using DFS and visualize Eigen Vecs
#    g_mean, g_sigma = norm.fit(affinity_mat)
    normed_dist = affinity_mat/np.max(affinity_mat)
    sim_aff = 1-normed_dist
#    affinity_mat = np.exp(- affinity_mat ** 2 / (2. * g_sigma ** 2))
    # Threshold the affinity_mat 
    th_idx, threshold_vals = spectral_utils.find_edge_threshold(sim_aff)
    threshold = threshold_vals[th_idx]

    ###########################################################################
    # Threshold the affinity_mat 
    if threshold is None:
        threshold = np.mean(affinity_mat)
    print("Threshold : {}".format(threshold))
    affinity_mat_thr = (sim_aff > threshold) * 1.
    np.fill_diagonal(affinity_mat_thr, 0)
    
    ###########################################################################
    # Compute the 2nd Eigen Vector and plot the values

#    affinity_mat_thr = np.ones_like(affinity_mat)
#    affinity_mat_thr[affinity_mat == 0] = 0.
#    affinity_mat_thr[affinity_mat > threshold_vals[i]] = 0.
    
    # form the laplacian matrix L = Diagonal matrix - Adjacency matrix
    lap_mat = np.copy(sim_aff)
    np.fill_diagonal(lap_mat, 0)
    lap_mat = -lap_mat
    np.fill_diagonal(lap_mat, -np.sum(lap_mat, axis=0))
    eVals, eVecs = np.linalg.eig(lap_mat)
    # roundoff the values to prevent floating point epsilon errors
    eVals, eVecs = np.round(eVals, 8), np.round(eVecs, 8)
    # sort in ascending order
    idx = eVals.argsort()
    eVals = eVals[idx]
    eVecs = eVecs[:, idx]
    
    plot_utils.visualize_evecs(eVals, eVecs, log_dir)
    
    ###########################################################################
    # Evaluate on the labels for entire dataset
    if nclasses == 3:
        annotation_path = ANNOTATION_FILE_3C
    elif nclasses == 5:
        annotation_path = ANNOTATION_FILE_5C
        
    spectral_model = SpectralClustering(n_clusters=nclasses, affinity='precomputed', 
                                        random_state=123)
    labels = spectral_model.fit_predict(sim_aff)
    
    # returns a list of cluster label assignments for trajectories
#    spectral_utils.spectral_plot(mean_feats, labels, log_dir)
    acc_values, perm_tuples, gt_list, pred_list = \
    eval_of_clusters.get_accuracy(mean_feats, labels, annotation_path, \
                                  DATASET, LABELS)
    acc_perc = [sum(k)/mean_feats.shape[0] for k in acc_values]
    
#    #best_acc[i,j] = max(acc_perc)
    #best_acc.append(max(acc_perc))
    best_indx = acc_perc.index(max(acc_perc))
    print("Max Acc. : ", max(acc_perc))
    best_perm = perm_tuples[best_indx]
    print("best perm_tuple : ", best_perm)    
    acc = eval_of_clusters.create_confusion_matrix(best_perm, pred_list, gt_list, nclasses)
    
    # Write the strokes in their respective folders
    if write_strokes:
        eval_of_clusters.evaluate(labels, row_names, log_dir, DATASET)
        
    return acc
    
  
def cluster_feats(base_name, log_dir, ft_files, ft_snames, ft_type="hoof", mth=2, nbins=10, 
         grid=None, clustering='kmeans', similarity='euclidean', nclasses=3, 
         write_strokes=False):

    seed = 1234
    print(60*"#")
    #####################################################################
    
    # Divide the samples files into training set, validation and test sets
    train_lst, val_lst, test_lst = autoenc_utils.split_dataset_files(DATASET)
    
    # form the names of the list of label files, should be at destination 
    train_lab = [f+".json" for f in train_lst]
    val_lab = [f+".json" for f in val_lst]
    test_lab = [f+".json" for f in test_lst]
    
    # get complete path lists of label files
    tr_labs = [os.path.join(LABELS, f) for f in train_lab]
    val_labs = [os.path.join(LABELS, f) for f in val_lab]
    #####################################################################
    
#    sizes = [utils.getNFrames(os.path.join(DATASET, f+".avi")) for f in train_lst]
#    val_sizes = [utils.getNFrames(os.path.join(DATASET, f+".avi")) for f in val_lst]
    
    print("No. of videos : {}".format(len(train_lst)+len(val_lst)+len(test_lst)))
        
    # Feature Extraction : (GRID OF / HOOF / 2D CNN / 3DCNN / IDT)
    # Get feats for only the training videos. Get ordered histograms of freq
    if ft_type == "ofgrid":
        print("OF Grid Features :: GRIDSIZE : {} ".format(grid))
        log_dir = os.path.join(log_dir, "ofgrid"+str(grid))
    elif ft_type == 'hoof':
        print("HOOF Features :: mth : {}, nBins : {}".format(mth, nbins))
        log_dir = os.path.join(log_dir, "manifold_hoof_b"+str(nbins)+"_mth"+str(mth))
    elif ft_type == '2dcnn':
        print("2DCNN (ResNet50) pretrained model extracted features :: seqsize : 1")
        log_dir = os.path.join(log_dir, "2dres")
    elif ft_type == '3dcnn':
        print("3DCNN (ResNet18) pretrained model extracted features :: seqsize : 16")
        log_dir = os.path.join(log_dir, "3dres")
    elif ft_type == 'attention':
        print("Attention model extracted features :: seqsize : 6")
        log_dir = os.path.join(log_dir, "attn_seq6")
    elif ft_type == 'transformer':
        print("transformer model extracted features :: seq 30 step 29")
        log_dir = os.path.join(log_dir, "transformer_seq30_step29")
    elif ft_type == "of20_SA_C1000":
        print("Soft assignment sequences for OF20 features with KMeans C1000 :")
        log_dir = os.path.join(log_dir, "of20_SA_C1000")
    elif ft_type == "conv_encdec":
        print("Conv Encoder Decoder extracted features :")
        log_dir = os.path.join(log_dir, "conv_encdec_seq8")
    elif ft_type == "conv_vae":
        print("Conv VAE extracted mean features :")
        log_dir = os.path.join(log_dir, "conv_vae_seq8")
    
    # create log folder tree, if it doesn't already exist
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    
    #####################################################################
    # read into dictionary {vidname: np array, ...}
    print("Loading features from disk...")
    #features = utils.readAllPartitionFeatures(c3dFC7FeatsPath, train_lst)
#    mainFeatures = utils.readAllPartitionFeatures(c3dFC7MainFeatsPath, train_lst_main)
#    features.update(mainFeatures)     # Merge dicts
    # get Nx4096 numpy matrix with columns as features and rows as window placement features
#    features, strokes_name_id = select_trimmed_feats(c3dFC7FeatsPath, LABELS, \
#                                                       train_lst, c3dWinSize) 
    
    if not os.path.exists(base_name):
        os.makedirs(base_name)
        #    # Extract Grid OF / HOOF features {mth = 2, and vary nbins}
        features, strokes_name_id = extract_stroke_feats(DATASET, LABELS, train_lst, \
                                                     nbins, mth, True, grid) 

#        BATCH_SIZE, SEQ_SIZE, STEP = 16, 16, 1
#        features, strokes_name_id = extract_feats(DATASET, LABELS, CLASS_IDS, BATCH_SIZE, 
#                                                  SEQ_SIZE, STEP, extractor='3dcnn', 
#                                                  part='train')
#        features, strokes_name_id = read_partition_feats(DATASET, LABELS, 
#                                                         IDT_FEATS+"_TrajLen"+str(traj_len),
#                                                         train_lst, traj_len) 
#        trajectories, stroke_names = autoenc_feats.extract_sequence_feats(model_path, 
#                                DATASET, LABELS, CLASS_IDS, BATCH_SIZE, SEQ_SIZE, 
#                                INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, partition='all', 
#                                nstrokes=50)
        with open(os.path.join(base_name, ft_files['train']), "wb") as fp:
            pickle.dump(features, fp)
        with open(os.path.join(base_name, ft_snames['train']), "wb") as fp:
            pickle.dump(strokes_name_id, fp)

    with open(os.path.join(base_name, ft_files["train"]), "rb") as fp:
        features = pickle.load(fp)
    with open(os.path.join(base_name, ft_snames["train"]), "rb") as fp:
        strokes_name_id = pickle.load(fp)
        
    with open(os.path.join(base_name, ft_files["val"]), "rb") as fp:
        features_val = pickle.load(fp)
    with open(os.path.join(base_name, ft_snames["val"]), "rb") as fp:
        strokes_name_id_val = pickle.load(fp)    
    
    with open(os.path.join(base_name, ft_files["test"]), "rb") as fp:
        features_test = pickle.load(fp)
    with open(os.path.join(base_name, ft_snames["test"]), "rb") as fp:
        strokes_name_id_test = pickle.load(fp)    
    
    # concat all partitions
    features.update(features_val)
    features.update(features_test)
    strokes_name_id.extend(strokes_name_id_val)
    strokes_name_id.extend(strokes_name_id_test)
    #####################################################################
    # get matrix (size (N, vec_size) ) of features from dictionary of stroke feats
    count_nans, count_infs = 0, 0
    vecs = []
    for key in strokes_name_id:
        # TODO: can use mean of feats
        count_nans += np.sum(np.isnan(features[key]))
        count_infs += np.sum(np.isinf(features[key]))
        features[key][np.isnan(features[key])] = 0
        features[key][np.isinf(features[key])] = 0
        vecs.append(features[key])
    print("No. of nan's : {} :: No. of inf's : {}".format(count_nans, count_infs))
    vecs = np.vstack(vecs)
    
    #fc7 layer output size (4096)
    INP_VEC_SIZE = vecs.shape[-1]
    print("INP_VEC_SIZE = ", INP_VEC_SIZE)
    
    #####################################################################
    
    # calculate mean values of trajectories
    all_feats = traj_utils.get_trajectory_mean_from_dict(features, strokes_name_id)
    all_feats = normalize(all_feats, norm='l2')
    
    ###################################################################
    # Convert stroke feats to sequence of stroke vectors
    trajectories = traj_utils.to_trajectory_list(features, strokes_name_id)
    
    if clustering is 'kmeans':
        acc = apply_manifold_cluster(all_feats, strokes_name_id, log_dir, nclasses, write_strokes)
    
    elif clustering is 'spectral':
        threshold = None  # 50000
        acc = apply_spectral(trajectories, strokes_name_id, all_feats, log_dir, 
                             similarity, nclasses, threshold, write_strokes)
    
    ###################################################################
    # plot the strokes as sequence of points
#    plot_utils.plot_trajectories3D(trajectories, strokes_name_id, log_dir)
    
    return acc
#    get_autoenc_spectral("res_autoenc_seq"+str(SEQ_SIZE))
#    main("res_autoenc_gru_seq"+str(SEQ_SIZE))
    
    
def main(sim):    

    base_path = "/home/arpan/VisionWorkspace/Cricket/CricketStrokeLocalizationBOVW/logs"
    if not os.path.exists(base_path):
        base_path = "/home/arpan/DATA_Drive2/Cricket/Workspace/CricketStrokeLocalizationBOVW/logs"
#    base_path = "/home/arpan/VisionWorkspace/Cricket/StrokeAttention/logs"
    log_dir = 'data/cluster_logs'
    clustering = 'kmeans'  # 'kmeans' , 'spectral'
#    similarity =  'corr' # 'corr' #'dtw' # 'euclidean' 'hausdorff'
    similarity = sim
    NCLASSES = 5
    WRITE_STROKES = False
    # Features Types : 'hoof', 'ofgrid', ,'2dcnn', '3dcnn', 'autoenc', 'transformer', 'of20_SA_C1000'
    # 'attention', "conv_vae"
    ft_type = 'hoof'
    nclusters = []
    grids = []
    
    nbins = list(range(20, 21, 10))
#    cluster_sizes = list(range(10, 51, 10))
    keys = ["ofGrid:"+str(t) for t in nbins]
#    beta = [0.6, 1.0, 1.0, 1.0]
#    keys = []
#    for i, t in enumerate(nbins):
#        keys.append("ofGrid:"+str(t)+" :: $\\beta="+str(beta[i])+"$")
    l = {}
    accuracies = []
    # ofGrid Accuracies on validation
    mth = 2
    for i, bins in enumerate(nbins):
        #######################################################################
        # normalized HOOF selection params
        folder_name = "bow_HL_hoof_b"+str(bins)+"_mth"+str(mth)
        ft_files = {"train": "hoof_feats_b"+str(bins)+".pkl",
                    "val": "hoof_feats_val_b"+str(bins)+".pkl",
                    "test" : "hoof_feats_test_b"+str(bins)+".pkl"}
        ft_snames = {"train": "hoof_snames_b"+str(bins)+".pkl",
                    "val": "hoof_snames_val_b"+str(bins)+".pkl",
                    "test" : "hoof_snames_test_b"+str(bins)+".pkl"}
        #######################################################################
        # OF Grid selection params
#        folder_name = "bow_HL_ofAng_grid"+str(bins)
#        ft_files = {"train": "of_feats_grid"+str(bins)+".pkl",
#                    "val": "of_feats_val_grid"+str(bins)+".pkl",
#                    "test" : "of_feats_test_grid"+str(bins)+".pkl"}
#        ft_snames = {"train": "of_snames_grid"+str(bins)+".pkl",
#                    "val": "of_snames_val_grid"+str(bins)+".pkl",
#                    "test" : "of_snames_test_grid"+str(bins)+".pkl"}
        #######################################################################
#        # 2DCNN (ResNet50 extracted) selection params
#        folder_name = "bow_HL_2dres"
#        ft_files = {"train": "2dcnn_feats_train"+".pkl",
#                    "val": "2dcnn_feats_val"+".pkl",
#                    "test" : "2dcnn_feats_test"+".pkl"}
#        ft_snames = {"train": "2dcnn_snames_train"+".pkl",
#                    "val": "2dcnn_snames_val"+".pkl",
#                    "test" : "2dcnn_snames_test"+".pkl"}
        #######################################################################   
#        # 3DCNN (ResNet18 extracted) selection params
#        folder_name = "bow_HL_3dres_seq16"
#        ft_files = {"train": "3dcnn_feats_train"+".pkl",
#                    "val": "3dcnn_feats_val"+".pkl",
#                    "test" : "3dcnn_feats_test"+".pkl"}
#        ft_snames = {"train": "3dcnn_snames_train"+".pkl",
#                    "val": "3dcnn_snames_val"+".pkl",
#                    "test" : "3dcnn_snames_test"+".pkl"}
        #######################################################################
        # Transformer trained and extracted features
#        folder_name = "bovtrans_HA_of20_Hidden200"
#        ft_files = {"train": "trans_feats.pkl",
#                    "val": "trans_feats_val.pkl",
#                    "test" : "trans_feats_test.pkl"}
#        ft_snames = {"train": "trans_snames.pkl",
#                    "val": "trans_snames_val.pkl",
#                    "test" : "trans_snames_test.pkl"}
        #######################################################################
        # Soft assignment sequences on OF20 features
#        folder_name = "bovgru_SA_of20_Hidden256"
#        ft_files = {"train": "C1000_train.pkl",
#                    "val": "C1000_val.pkl",
#                    "test" : "C1000_test.pkl"}
#        ft_snames = {"train": "C1000_snames_train.pkl",
#                    "val": "C1000_snames_val.pkl",
#                    "test" : "C1000_snames_test.pkl"}
        #######################################################################
#        # Soft assignment sequences on OF20 features
#        folder_name = "attn_seq6"
#        ft_files = {"train": "attn_feats.pkl",
#                    "val": "attn_feats_val.pkl",
#                    "test" : "attn_feats_test.pkl"}
#        ft_snames = {"train": "attn_snames.pkl",
#                    "val": "attn_snames_val.pkl",
#                    "test" : "attn_snames_test.pkl"}
#
#        #######################################################################
#        # Conv Enc Dec extracted features
#        folder_name = "conv_encdec_framesNew_seq8"
#        ft_files = {"train": "conv_encdec_train.pkl",
#                    "val": "conv_encdec_val.pkl",
#                    "test" : "conv_encdec_test.pkl"}
#        ft_snames = {"train": "conv_encdec_snames_train.pkl",
#                    "val": "conv_encdec_snames_val.pkl",
#                    "test" : "conv_encdec_snames_test.pkl"}

        #######################################################################
#        # Conv VAE extracted mean features
#        folder_name = "conv_vae_flowviz_seq8"
#        ft_files = {"train": "conv_vae_train.pkl",
#                    "val": "conv_vae_val.pkl",
#                    "test" : "conv_vae_test.pkl"}
#        ft_snames = {"train": "conv_vae_snames_train.pkl",
#                    "val": "conv_vae_snames_val.pkl",
#                    "test" : "conv_vae_snames_test.pkl"}
        
#        folder_name = "bow_HL_3dres_seq16_cl20"
        
##        nclusters.append(cluster_size)
        

        grids.append(bins)
        acc = cluster_feats(os.path.join(base_path, folder_name), log_dir, ft_files, ft_snames,
                   ft_type, mth, bins, bins, clustering, similarity, NCLASSES, 
                   WRITE_STROKES)
        accuracies.append(acc)
#        l[keys[i]] = accuracies
#        break
    print("Accuracies : ", accuracies)
#        
#    fname = os.path.join(base_path, "ofGrid_SA_cl150_old.png")
#    plot_accuracy(cluster_sizes, keys, l, "#Words", "Accuracy", fname)

    
    ###########################################################################
    
    
if __name__=='__main__':
    
    sim = ['dtw', 'hausdorff', 'corr', 'euclidean']
    main(sim[0])
#    main(sim[1])
#    main(sim[2])
#    main(sim[3])


###########################################################################
#    # Building the clustering model 
#    spectral_model_rbf = SpectralClustering(n_clusters = 3, affinity ='rbf') 
#    # Training the model and Storing the predicted cluster labels 
#    labels_rbf = spectral_model_rbf.fit_predict(X_principal)
#    
#    # Building the clustering model 
#    spectral_model_nn = SpectralClustering(n_clusters = 3, affinity ='nearest_neighbors') 
#      
#    # Training the model and Storing the predicted cluster labels 
#    labels_nn = spectral_model_nn.fit_predict(X_principal) 