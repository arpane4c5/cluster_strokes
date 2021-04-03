#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 06:55:21 2021

@author: arpan

Create bar graphs for paper comparing accuracy values of trajectory clustering.
"""

import numpy as np
import os
from matplotlib import pyplot as plt

# cluster_logs/
def compare_diff_feats_1normed():
    ''' 1 - Normed values for spectral clustering
    '''
    # cluster_logs/
    kmeans = [0.44661921708185054, 0.3683274021352313, 0.42170818505338076, 0.3487544483985765, 0.3683274021352313, 0.33629893238434166, 0.26868327402135234, 0.28113879003558717, 0.29359430604982206, 0.3274021352313167, ]
    spec1_norm_dtw = [0.5498220640569395, 0.3807829181494662, 0.398576512455516, 0.36298932384341637, 0.3701067615658363, 0.3594306049822064, 0.3096085409252669, 0.297153024911032, 0.3185053380782918, 0.31494661921708184, ]
    spec1_norm_haus = [0.4359430604982206, 0.41459074733096085, 0.4110320284697509, 0.3701067615658363, 0.3932384341637011, 0.3220640569395018, 0.28647686832740216, 0.28113879003558717, 0.3202846975088968, 0.33274021352313166, ]
    spec1_norm_corr = [0.6352313167259787, 0.3861209964412811, 0.5480427046263345, 0.3540925266903915, 0.4893238434163701, 0.4234875444839858, 0.42704626334519574, 0.31316725978647686, 0.35231316725978645, 0.35765124555160144, ]
    spec1_norm_eucl = [0.5569395017793595, 0.45729537366548045, 0.47153024911032027, 0.37544483985765126, 0.41459074733096085, 0.4110320284697509, 0.297153024911032, 0.31316725978647686, 0.3274021352313167, 0.30071174377224197, ]
    # HOOF TrueDen mth2_b40
    labs = ["HOOF:mth2_b40", "OFGridAng:20", "OFGridMagAng:20", "HOG", "2D ResNet50", 
            "3D ResNet18", "BPBB:GT", "BPBB:HOOF", "BPKP:zeroFilled", "BPKP:meanFilled"]
    ######################################################################################
#    plot_mpa_bar()
    fig, ax = plt.subplots()
    x = np.arange(10)
    
    # The width of the bars (1 = the whole width of the 'year group')
    width = 0.15
    # Create the bar charts!
    ax.bar(x - 3*width/2, kmeans, width, label='KMeans', color='#0343df')
    ax.bar(x - width/2, spec1_norm_dtw, width, label='Spec: DTW', color='#e50000')
    ax.bar(x + width/2, spec1_norm_haus, width, label='Spec: Haus', color='#ffff14')
    ax.bar(x + 3*width/2, spec1_norm_corr, width, label='Spec: Corr', color='#929591')
    ax.bar(x + 5*width/2, spec1_norm_eucl, width, label='Spec: Eucl', color='#e50099')
    
    # Notice that features like labels and titles are added in separate steps
    ax.set_ylabel('MPA')
    ax.set_title('Maximum Permutation Accuracy for stroke sequences')

    ax.set_xticks(x)    # This ensures we have one tick per year, otherwise we get fewer
    ax.set_xticklabels(labs, rotation='vertical')
    ax.set_ylim(bottom=0, top=1)
    ax.legend()
    plt.show()
    fig.savefig(os.path.join('plots', 'diffFeats_spec_accuracy5_plot.png'), bbox_inches='tight', dpi=300)
    

# cluster_logs/
def compare_diff_feats_dsigma():
    ''' d by sigma for spectral clustering
    '''
    # cluster_logs/
#    kmeans = []
    spec1_dtw = [0.5071174377224199, 0.3914590747330961, 0.35587188612099646, 0.3932384341637011, 0.35587188612099646, 0.3594306049822064, 0.302491103202847, 0.3291814946619217, 0.37722419928825623, 0.3202846975088968, ]
    spec1_haus = [0.43416370106761565, 0.3665480427046263, 0.3469750889679715, 0.3505338078291815, 0.33451957295373663, 0.31494661921708184, 0.3469750889679715, 0.2669039145907473, 0.3469750889679715, 0.3505338078291815, ]
    spec1_corr = [0.599644128113879, 0.4199288256227758, 0.498220640569395, 0.40391459074733094, 0.4128113879003559, 0.40569395017793597, 0.3540925266903915, 0.3309608540925267, 0.3238434163701068, 0.34341637010676157, ]
    spec1_eucl = [0.603202846975089, 0.4679715302491103, 0.4359430604982206, 0.4092526690391459, 0.3861209964412811, 0.49110320284697506, 0.3096085409252669, 0.31316725978647686, 0.32562277580071175, 0.34341637010676157, ]
    # HOOF from graph_hoof_trueDensity.py
    labs = ["HOOF:mth2_b40", "OFGridAng:20", "OFGridMagAng:20", "HOG", "2D ResNet50", "3D ResNet18", 
            "BPBB:GT", "BPBB:HOOF", "BPKP:zeroFilled", "BPKP:meanFilled"]
    ######################################################################################
#    plot_mpa_bar()
    fig, ax = plt.subplots()
    x = np.arange(10)
    
    # The width of the bars (1 = the whole width of the 'year group')
    width = 0.15
    # Create the bar charts!
    ax.bar(x - 3*width/2, spec1_dtw, width, label='Spec: DTW', color='#e50000')
    ax.bar(x - width/2, spec1_haus, width, label='Spec: Haus', color='#ffff14')
    ax.bar(x + width/2, spec1_corr, width, label='Spec: Corr', color='#929591')
    ax.bar(x + 3*width/2, spec1_eucl, width, label='Spec: Eucl', color='#e50099')
    
    # Notice that features like labels and titles are added in separate steps
    ax.set_ylabel('MPA')
    ax.set_title('Maximum Permutation Accuracy for stroke sequences')

    ax.set_xticks(x)    # This ensures we have one tick per year, otherwise we get fewer
    ax.set_xticklabels(labs, rotation='vertical')
    ax.set_ylim(bottom=0, top=1)
    ax.legend()
    plt.show()
    fig.savefig(os.path.join('plots', 'diffFeats_spec_dsigma_accuracy5_plot.png'), bbox_inches='tight', dpi=300)
    

    
def compare_diff_feats_rbf():
    
    spec1_dtw = [0.5693950177935944, 0.3879003558718861, 0.3683274021352313, 0.400355871886121, 0.45373665480427045, 0.36298932384341637, 0.3274021352313167, 0.35231316725978645, 0.3736654804270463, 0.33451957295373663]
    
    spec1_haus = [0.44483985765124556, 0.35587188612099646, 0.38434163701067614, 0.4110320284697509, 0.3914590747330961, 0.3718861209964413, 0.3540925266903915, 0.33274021352313166, 0.36476868327402134, 0.33451957295373663]
    # sigma = 1
    spec1_corr = [0.5480427046263345, 0.40747330960854095, 0.5106761565836299, 0.42704626334519574, 0.43238434163701067, 0.4412811387900356, 0.34519572953736655, 0.3309608540925267, 0.3416370106761566, 0.33451957295373663]
    # sigma = 1, HOOF sigma calculated
    spec1_eucl = [0.6370106761565836, 0.41637010676156583, 0.45729537366548045, 0.3505338078291815, 0.38434163701067614, 0.35231316725978645, 0.29537366548042704, 0.3096085409252669, 0.31494661921708184, 0.3113879003558719]
    
    labs = ["HOOF:mth2_b40", "OFGridAng:20", "OFGridMagAng:20", "HOG", "2D ResNet50", "3D ResNet18", 
            "BPBB:GT", "BPBB:HOOF", "BPKP:zeroFilled", "BPKP:meanFilled"]
    ######################################################################################
#    plot_mpa_bar()
    fig, ax = plt.subplots()
    x = np.arange(10)
    width = 0.15
    # Create the bar charts!
    ax.bar(x - 3*width/2, spec1_dtw, width, label='Spec: DTW', color='#e50000')
    ax.bar(x - width/2, spec1_haus, width, label='Spec: Haus', color='#ffff14')
    ax.bar(x + width/2, spec1_corr, width, label='Spec: Corr', color='#929591')
    ax.bar(x + 3*width/2, spec1_eucl, width, label='Spec: Eucl', color='#e50099')
    
    # Notice that features like labels and titles are added in separate steps
    ax.set_ylabel('MPA')
    ax.set_title('Maximum Permutation Accuracy for stroke sequences')

    ax.set_xticks(x)    # This ensures we have one tick per year, otherwise we get fewer
    ax.set_xticklabels(labs, rotation='vertical')
    ax.set_ylim(bottom=0, top=1)
    ax.legend()
    plt.show()
    fig.savefig(os.path.join('plots', 'diffFeats_spec_rbf_accuracy5_plot.png'), bbox_inches='tight', dpi=300)
    



if __name__ == '__main__':

    compare_diff_feats_1normed()
    compare_diff_feats_dsigma()
    compare_diff_feats_rbf()
    
    ######################################################################################
    
