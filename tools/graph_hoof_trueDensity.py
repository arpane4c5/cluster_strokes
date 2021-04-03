#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 06:55:21 2021

@author: arpan

Create bar graphs for paper comparing accuracy values of trajectory clustering.
"""

import numpy as np
import os
import re
from matplotlib import pyplot as plt
import pandas as pd

# cluster_logs/all_hoof/trueDensity/log_standardized.log and log_standardized_3.log
def all_hoof_trueDensity():
    ''' 1 - Normed values for spectral clustering
    '''
    # mth=1:4 and nbins = 10:40
    # cluster_logs/all_hoof/falseDensity/   : standardized feats : 
    kmeans = [0.5213523131672598, 0.46441281138790036, 0.4750889679715303, 
              0.45729537366548045, 0.5302491103202847, 0.4483985765124555, 
              0.46441281138790036, 0.44661921708185054, 0.5373665480427047, 
              0.42704626334519574, 0.45195729537366547, 0.45907473309608543, 
              0.5373665480427047, 0.44661921708185054, 0.4395017793594306, 
              0.4377224199288256]
    spec1_dtw = [0.5693950177935944, 0.5516014234875445, 0.5498220640569395, 
                 0.5604982206405694, 0.5249110320284698, 0.5747330960854092, 
                 0.5444839857651246, 0.5498220640569395, 0.5622775800711743, 
                 0.505338078291815, 0.5355871886120996, 0.5409252669039146, 
                 0.5533807829181495, 0.5195729537366548, 0.5516014234875445, 
                 0.5195729537366548]
    spec1_haus = [0.39679715302491103, 0.4092526690391459, 0.40747330960854095, 
                  0.4822064056939502, 0.4092526690391459, 0.43416370106761565, 
                  0.40569395017793597, 0.4359430604982206, 0.3665480427046263, 
                  0.398576512455516, 0.46619217081850534, 0.4092526690391459, 
                  0.4306049822064057, 0.398576512455516, 0.4288256227758007, 
                  0.41637010676156583]
    spec1_corr = [0.599644128113879, 0.6352313167259787, 0.6263345195729537, 
                  0.6316725978647687, 0.599644128113879, 0.6370106761565836, 
                  0.6263345195729537, 0.6352313167259787, 0.5782918149466192, 
                  0.6405693950177936, 0.6245551601423488, 0.6227758007117438, 
                  0.6192170818505338, 0.6441281138790036, 0.6334519572953736, 
                  0.6334519572953736]
    spec1_eucl = [0.5658362989323843, 0.5693950177935944, 0.5676156583629893, 
                  0.5658362989323843, 0.5676156583629893, 0.5676156583629893, 
                  0.5676156583629893, 0.5569395017793595, 0.5604982206405694, 
                  0.5516014234875445, 0.5622775800711743, 0.5729537366548043, 
                  0.5533807829181495, 0.5569395017793595, 0.5444839857651246, 
                  0.5533807829181495]
    
    labs = ["mth"+str(m)+"_b"+str(b) for m in range(1, 5, 1) for b in range(10, 41, 10)]
    ######################################################################################
#    plot_mpa_bar()
    fig, ax = plt.subplots()
    x = np.arange(16)
     
    # The width of the bars (1 = the whole width of the 'year group')
    width = 0.15
    # Create the bar charts!
    ax.bar(x - 3*width/2, kmeans, width, label='KMeans', color='#0343df')
    ax.bar(x - width/2, spec1_dtw, width, label='Spec: DTW', color='#e50000')
    ax.bar(x + width/2, spec1_haus, width, label='Spec: Haus', color='#ffff14')
    ax.bar(x + 3*width/2, spec1_corr, width, label='Spec: Corr', color='#929591')
    ax.bar(x + 5*width/2, spec1_eucl, width, label='Spec: Eucl', color='#e50099')
    
    # Notice that features like labels and titles are added in separate steps
    ax.set_ylabel('MPA')
    ax.set_title('Maximum Permutation Accuracy for stroke sequences')

    ax.set_xticks(x)    # This ensures we have one tick per year, otherwise we get fewer
    ax.set_xticklabels(labs, rotation='vertical')
    ax.set_ylim(bottom=0, top=1)
    ax.legend()
    plt.show()
    fig.savefig(os.path.join('plots', 'hoofTrueDen_spec_accuracy5_plot.png'), bbox_inches='tight', dpi=300)
    

def all_hoof_trueDensity_sigma():
    
    spec1_dsigma_dtw = [0.5231316725978647, 0.5142348754448398, 0.5071174377224199, 
                        0.505338078291815, 0.5569395017793595, 0.5124555160142349, 
                        0.5106761565836299, 0.5071174377224199, 0.5551601423487544, 
                        0.5071174377224199, 0.5106761565836299, 0.505338078291815, 
                        0.5533807829181495, 0.5088967971530249, 0.5071174377224199, 
                        0.5142348754448398]
    
    spec1_dsigma_haus = [0.4501779359430605, 0.43238434163701067, 0.4359430604982206, 
                         0.4483985765124555, 0.42170818505338076, 0.42526690391459077, 
                         0.45373665480427045, 0.43416370106761565, 0.3932384341637011, 
                         0.42704626334519574, 0.4377224199288256, 0.41637010676156583, 
                         0.4377224199288256, 0.35231316725978645, 0.3469750889679715, 
                         0.40747330960854095]
    spec1_dsigma_corr = [0.47686832740213525, 0.6014234875444839, 0.599644128113879, 
                         0.599644128113879, 0.4750889679715303, 0.6298932384341637, 
                         0.6120996441281139, 0.599644128113879, 0.6263345195729537, 
                         0.603202846975089, 0.6103202846975089, 0.5960854092526691, 
                         0.6227758007117438, 0.608540925266904, 0.597864768683274, 
                         0.594306049822064]
    
    spec1_dsigma_eucl = [0.5765124555160143, 0.6014234875444839, 0.6014234875444839, 
                         0.5925266903914591, 0.5765124555160143, 0.6014234875444839, 
                         0.5854092526690391, 0.603202846975089, 0.5711743772241993, 
                         0.5925266903914591, 0.5854092526690391, 0.5765124555160143, 
                         0.5907473309608541, 0.594306049822064, 0.608540925266904, 
                         0.6120996441281139]
    
    labs = ["mth"+str(m)+"_b"+str(b) for m in range(1, 5, 1) for b in range(10, 41, 10)]
    ######################################################################################
#    plot_mpa_bar()
    fig, ax = plt.subplots()
    x = np.arange(16)
    width = 0.15
    # Create the bar charts!
    ax.bar(x - 3*width/2, spec1_dsigma_dtw, width, label='Spec: DTW', color='#e50000')
    ax.bar(x - width/2, spec1_dsigma_haus, width, label='Spec: Haus', color='#ffff14')
    ax.bar(x + width/2, spec1_dsigma_corr, width, label='Spec: Corr', color='#929591')
    ax.bar(x + 3*width/2, spec1_dsigma_eucl, width, label='Spec: Eucl', color='#e50099')
    
    # Notice that features like labels and titles are added in separate steps
    ax.set_ylabel('MPA')
    ax.set_title('Maximum Permutation Accuracy for stroke sequences')

    ax.set_xticks(x)    # This ensures we have one tick per year, otherwise we get fewer
    ax.set_xticklabels(labs, rotation='vertical')
    ax.set_ylim(bottom=0, top=1)
    ax.legend()
    plt.show()
    fig.savefig(os.path.join('plots', 'hoofTrueDen_spec_dsigma_5_plot.png'), bbox_inches='tight', dpi=300)
    
def all_hoof_trueDensity_rbf():
    
    spec1_dtw = [0.6174377224199288, 0.6512455516014235, 0.6512455516014235, 
                 0.5622775800711743, 0.5871886120996441, 0.5747330960854092, 
                 0.5960854092526691, 0.5693950177935944, 0.5444839857651246, 
                 0.5765124555160143, 0.5195729537366548, 0.5800711743772242, 
                 0.5498220640569395, 0.5, 0.4110320284697509, 0.4110320284697509]
    
    spec1_haus = [0.42526690391459077, 0.5177935943060499, 0.47686832740213525, 
                  0.4626334519572954, 0.4555160142348754, 0.4110320284697509, 
                  0.43238434163701067, 0.44483985765124556, 0.35587188612099646, 
                  0.40569395017793597, 0.4306049822064057, 0.4626334519572954, 
                  0.37722419928825623, 0.3416370106761566, 0.3238434163701068, 
                  0.36476868327402134]
    # sigma = 1
    spec1_corr = [0.5587188612099644, 0.6387900355871886, 0.5907473309608541, 
                  0.5587188612099644, 0.5729537366548043, 0.6316725978647687, 
                  0.6334519572953736, 0.5480427046263345, 0.5765124555160143, 
                  0.501779359430605, 0.5195729537366548, 0.5409252669039146, 
                  0.5800711743772242, 0.5195729537366548, 0.5177935943060499, 
                  0.5729537366548043]
    
    spec1_eucl = [0.697508896797153, 0.6565836298932385, 0.6298932384341637, 
                  0.6174377224199288, 0.6850533807829181, 0.6832740213523132, 
                  0.6583629893238434, 0.6370106761565836, 0.6725978647686833, 
                  0.5177935943060499, 0.4377224199288256, 0.4359430604982206, 
                  0.5907473309608541, 0.5231316725978647, 0.49644128113879005, 
                  0.45729537366548045]
    
    labs = ["mth"+str(m)+"_b"+str(b) for m in range(1, 5, 1) for b in range(10, 41, 10)]
    ######################################################################################
#    plot_mpa_bar()
    fig, ax = plt.subplots()
    x = np.arange(16)
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
    fig.savefig(os.path.join('plots', 'hoofTrueDen_spec_rbf_5_plot.png'), bbox_inches='tight', dpi=300)
    
# cluster_logs/all_hoof/falseDensity/log_standardized_3.log
def all_hoof3_trueDensity():
    ''' 1 - Normed values for spectral clustering
    '''
    # mth=1:4 and nbins = 10:40
    # cluster_logs/all_hoof/falseDensity/   : standardized feats : 
    kmeans = [0.8131672597864769, 0.802491103202847, 0.797153024911032, 
              0.7846975088967971, 0.8042704626334519, 0.798932384341637, 
              0.7864768683274022, 0.7811387900355872, 0.8078291814946619, 
              0.7882562277580071, 0.7793594306049823, 0.7864768683274022, 
              0.802491103202847, 0.7793594306049823, 0.7829181494661922, 
              0.7722419928825622]
    spec1_dtw = [0.7935943060498221, 0.7811387900355872, 0.7224199288256228, 
                 0.5462633451957295, 0.798932384341637, 0.806049822064057, 
                 0.806049822064057, 0.5373665480427047, 0.798932384341637, 
                 0.8078291814946619, 0.791814946619217, 0.5338078291814946, 
                 0.798932384341637, 0.806049822064057, 0.7935943060498221, 
                 0.6957295373665481]
    spec1_haus = [0.5622775800711743, 0.6423487544483986, 0.6316725978647687, 
                  0.5177935943060499, 0.5195729537366548, 0.6014234875444839, 
                  0.6352313167259787, 0.4893238434163701, 0.5640569395017794, 
                  0.5711743772241993, 0.4555160142348754, 0.4359430604982206, 
                  0.5071174377224199, 0.5391459074733096, 0.4483985765124555, 
                  0.4306049822064057]
    spec1_corr = [0.697508896797153, 0.6725978647686833, 0.5088967971530249, 
                  0.505338078291815, 0.699288256227758, 0.6708185053380783, 
                  0.4875444839857651, 0.6530249110320284, 0.6832740213523132, 
                  0.6743772241992882, 0.505338078291815, 0.49110320284697506, 
                  0.6690391459074733, 0.5160142348754448, 0.5160142348754448, 
                  0.5106761565836299]
    spec1_eucl = [0.8096085409252669, 0.7633451957295374, 0.7473309608540926, 
                  0.7384341637010676, 0.8096085409252669, 0.7669039145907474, 
                  0.7544483985765125, 0.7437722419928826, 0.8167259786476868, 
                  0.7633451957295374, 0.7562277580071174, 0.7455516014234875, 
                  0.806049822064057, 0.7811387900355872, 0.7633451957295374, 
                  0.7508896797153025]
    
    labs = ["mth"+str(m)+"_b"+str(b) for m in range(1, 5, 1) for b in range(10, 41, 10)]
    ######################################################################################
#    plot_mpa_bar()
    fig, ax = plt.subplots()
    x = np.arange(16)
    
    # The width of the bars (1 = the whole width of the 'year group')
    width = 0.15
    # Create the bar charts!
    ax.bar(x - 3*width/2, kmeans, width, label='KMeans', color='#0343df')
    ax.bar(x - width/2, spec1_dtw, width, label='Spec: DTW', color='#e50000')
    ax.bar(x + width/2, spec1_haus, width, label='Spec: Haus', color='#ffff14')
    ax.bar(x + 3*width/2, spec1_corr, width, label='Spec: Corr', color='#929591')
    ax.bar(x + 5*width/2, spec1_eucl, width, label='Spec: Eucl', color='#e50099')
    
    # Notice that features like labels and titles are added in separate steps
    ax.set_ylabel('MPA')
    ax.set_title('Maximum Permutation Accuracy for stroke sequences')

    ax.set_xticks(x)    # This ensures we have one tick per year, otherwise we get fewer
    ax.set_xticklabels(labs, rotation='vertical')
    ax.set_ylim(bottom=0, top=1)
    ax.legend()
    plt.show()
    fig.savefig(os.path.join('plots', 'hoofTrueDen_spec_accuracy3_plot.png'), bbox_inches='tight', dpi=300)

def all_hoof3_trueDensity_sigma():
    
    spec1_dtw = [0.8042704626334519, 0.5409252669039146, 0.5373665480427047, 
                 0.5409252669039146, 0.8256227758007118, 0.5355871886120996, 
                 0.5355871886120996, 0.5373665480427047, 0.8238434163701067, 
                 0.5284697508896797, 0.5266903914590747, 0.5266903914590747, 
                 0.8131672597864769, 0.5284697508896797, 0.5231316725978647, 
                 0.5231316725978647]
    
    spec1_haus = [0.5782918149466192, 0.6779359430604982, 0.5338078291814946, 
                  0.5498220640569395, 0.5622775800711743, 0.6103202846975089, 
                  0.5338078291814946, 0.5444839857651246, 0.5640569395017794, 
                  0.599644128113879, 0.5, 0.50355871886121, 0.5302491103202847, 
                  0.5284697508896797, 0.5231316725978647, 0.49110320284697506]
    spec1_corr = [0.6690391459074733, 0.6761565836298933, 0.6779359430604982, 
                  0.6637010676156584, 0.6672597864768683, 0.6743772241992882, 
                  0.6654804270462633, 0.6512455516014235, 0.6779359430604982, 
                  0.6637010676156584, 0.6476868327402135, 0.6387900355871886, 
                  0.6957295373665481, 0.6743772241992882, 0.6548042704626335, 
                  0.6494661921708185]
    
    spec1_eucl = [0.8185053380782918, 0.8113879003558719, 0.798932384341637, 
                  0.7882562277580071, 0.8327402135231317, 0.8185053380782918, 
                  0.8096085409252669, 0.806049822064057, 0.8291814946619217, 
                  0.8185053380782918, 0.8042704626334519, 0.791814946619217, 
                  0.8345195729537367, 0.8202846975088968, 0.791814946619217, 
                  0.7829181494661922]
    
    labs = ["mth"+str(m)+"_b"+str(b) for m in range(1, 5, 1) for b in range(10, 41, 10)]
    ######################################################################################
#    plot_mpa_bar()
    fig, ax = plt.subplots()
    x = np.arange(16)
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
    fig.savefig(os.path.join('plots', 'hoofTrueDen_spec_dsigma_3_plot.png'), bbox_inches='tight', dpi=300)

def all_hoof3_trueDensity_rbf():
    
    spec1_dtw = [0.8220640569395018, 0.5391459074733096, 0.5427046263345195, 
                 0.5427046263345195, 0.8238434163701067, 0.5409252669039146, 
                 0.5409252669039146, 0.5391459074733096, 0.8309608540925267, 
                 0.5444839857651246, 0.5462633451957295, 0.5480427046263345, 
                 0.8256227758007118, 0.5373665480427047, 0.5444839857651246, 
                 0.5427046263345195]
    
    spec1_haus = [0.597864768683274, 0.5533807829181495, 0.5284697508896797, 
                  0.5498220640569395, 0.5622775800711743, 0.5729537366548043, 
                  0.5302491103202847, 0.5427046263345195, 0.5765124555160143, 
                  0.5391459074733096, 0.498220640569395, 0.49644128113879005, 
                  0.5266903914590747, 0.5320284697508897, 0.4786476868327402, 
                  0.4928825622775801]
    # sigma = 1
    spec1_corr = [0.693950177935943, 0.6814946619217082, 0.6725978647686833, 
                  0.6690391459074733, 0.6921708185053381, 0.697508896797153, 
                  0.6708185053380783, 0.6725978647686833, 0.6886120996441281, 
                  0.6921708185053381, 0.6708185053380783, 0.6779359430604982, 
                  0.6779359430604982, 0.6921708185053381, 0.6779359430604982, 
                  0.6743772241992882]
    # sigma = 1
    spec1_eucl = [0.800711743772242, 0.7562277580071174, 0.7455516014234875, 
                  0.7348754448398577, 0.8078291814946619, 0.7615658362989324, 
                  0.7526690391459074, 0.7437722419928826, 0.8149466192170819, 
                  0.7651245551601423, 0.7580071174377224, 0.7384341637010676, 
                  0.8096085409252669, 0.7811387900355872, 0.7669039145907474, 
                  0.7562277580071174]
    
    labs = ["mth"+str(m)+"_b"+str(b) for m in range(1, 5, 1) for b in range(10, 41, 10)]
    ######################################################################################
#    plot_mpa_bar()
    fig, ax = plt.subplots()
    x = np.arange(16)
    width = 0.15
    # Create the bar charts!
    ax.bar(x - 3*width/2, spec1_dtw, width, label='Spec: DTW', color='#e50000')
    ax.bar(x - width/2, spec1_haus, width, label='Spec: Haus', color='#ffff14')
    ax.bar(x + width/2, spec1_corr, width, label='Spec: Corr', color='#929591')
    ax.bar(x + 3*width/2, spec1_eucl, width, label='Spec: Eucl', color='#e50099')
    
    # Notice that features like labels and titles are added in separate steps
    ax.set_ylabel('MPA (RBF)')
    ax.set_title('MPA for 3 classes')

    ax.set_xticks(x)    # This ensures we have one tick per year, otherwise we get fewer
    ax.set_xticklabels(labs, rotation='vertical')
    ax.set_ylim(bottom=0, top=1)
    ax.legend()
    plt.show()
    fig.savefig(os.path.join('plots', 'hoofTrueDen_spec_rbf_3_plot.png'), bbox_inches='tight', dpi=300)


if __name__ == '__main__':

    all_hoof_trueDensity()
    all_hoof_trueDensity_sigma()
    all_hoof_trueDensity_rbf()
    all_hoof3_trueDensity()
    all_hoof3_trueDensity_sigma()
    all_hoof3_trueDensity_rbf()
    
    ######################################################################################
    
    # d by sigma values for hoof feats
    
#    l1 = {"train loss" : train_loss, "test loss": test_loss}
#    l2 = {"train accuracy" : train_acc, "test accuracy" : test_acc}
#    best_ep = test_acc.index(max(test_acc)) + 1
#    loss_file = 'logs/plot_data/C3DFine_seq16.png'
#    acc_file = 'logs/plot_data/C3DFine_acc_seq16.png'
#    plot_traintest_loss(["train loss", "test loss"], l1, "Epochs", "Loss", seq, 16, loss_file)
#    plot_traintest_accuracy(["train accuracy", "test accuracy"], l2, "Epochs", "Accuracy", seq, 
#                            16, best_ep, acc_file)
