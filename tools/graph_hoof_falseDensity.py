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

# cluster_logs/all_hoof/falseDensity/log_standardized.log and log_standardized_3.log
def all_hoof_falseDensity():
    ''' 1 - Normed values for spectral clustering
    '''
    # mth=1:4 and nbins = 10:40
    # cluster_logs/all_hoof/falseDensity/   : standardized feats : 
    kmeans = [0.505338078291815, 0.50355871886121, 0.5177935943060499, 0.5177935943060499, 
              0.5462633451957295, 0.5071174377224199, 0.5195729537366548, 0.5177935943060499, 
              0.5871886120996441, 0.5106761565836299, 0.5142348754448398, 0.5106761565836299, 
              0.5871886120996441, 0.50355871886121, 0.5142348754448398, 0.505338078291815]
    spec1_norm_dtw = [0.5391459074733096, 0.5249110320284698, 0.5302491103202847, 
                      0.5498220640569395, 0.5462633451957295, 0.5320284697508897, 
                      0.5355871886120996, 0.5355871886120996, 0.5355871886120996, 
                      0.5551601423487544, 0.5409252669039146, 0.5462633451957295, 
                      0.5391459074733096, 0.5622775800711743, 0.5729537366548043, 
                      0.5658362989323843]
    spec1_norm_haus = [0.505338078291815, 0.5195729537366548, 0.5231316725978647, 
                       0.5249110320284698, 0.5231316725978647, 0.5338078291814946, 
                       0.5427046263345195, 0.5302491103202847, 0.50355871886121, 
                       0.5693950177935944, 0.5622775800711743, 0.5551601423487544, 
                       0.5142348754448398, 0.5782918149466192, 0.5729537366548043, 
                       0.5711743772241993]
    spec1_norm_corr = [0.5693950177935944, 0.5569395017793595, 0.5480427046263345, 
                       0.5516014234875445, 0.5854092526690391, 0.5533807829181495, 
                       0.5569395017793595, 0.5409252669039146, 0.5871886120996441, 
                       0.5533807829181495, 0.5640569395017794, 0.5444839857651246, 
                       0.5871886120996441, 0.5551601423487544, 0.5516014234875445, 
                       0.5338078291814946]
    spec1_norm_eucl = [0.5640569395017794, 0.47686832740213525, 0.47686832740213525, 
                       0.4750889679715303, 0.5266903914590747, 0.4928825622775801, 
                       0.4875444839857651, 0.4928825622775801, 0.5249110320284698, 
                       0.501779359430605, 0.505338078291815, 0.5, 0.5213523131672598, 
                       0.5088967971530249, 0.5088967971530249, 0.5]
    
    labs = ["mth"+str(m)+"_b"+str(b) for m in range(1, 5, 1) for b in range(10, 41, 10)]
    ######################################################################################
#    plot_mpa_bar()
    fig, ax = plt.subplots()
    x = np.arange(16)
     
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
    fig.savefig(os.path.join('plots', 'hoof_spec_accuracy5_plot.png'), bbox_inches='tight', dpi=300)
    

def all_hoof_falseDensity_sigma():
    
    spec1_dsigma_dtw = [0.4483985765124555, 0.4608540925266904, 0.44661921708185054, 
                    0.45373665480427045, 0.4697508896797153, 0.4608540925266904, 
                    0.4608540925266904, 0.47330960854092524, 0.4608540925266904, 
                    0.4822064056939502, 0.5177935943060499, 0.5373665480427047, 
                    0.47330960854092524, 0.5444839857651246, 0.5213523131672598, 
                    0.5391459074733096]
    
    spec1_dsigma_haus = [0.46619217081850534, 0.49110320284697506, 0.5249110320284698, 
                         0.4501779359430605, 0.5195729537366548, 0.4412811387900356, 
                         0.42170818505338076, 0.43416370106761565, 0.5338078291814946, 
                         0.45195729537366547, 0.4359430604982206, 0.43238434163701067, 
                         0.5516014234875445, 0.4412811387900356, 0.4412811387900356, 
                         0.4555160142348754]
    spec1_dsigma_corr = [0.5213523131672598, 0.5480427046263345, 0.5409252669039146, 
                         0.5498220640569395, 0.5266903914590747, 0.5427046263345195, 
                         0.5373665480427047, 0.5480427046263345, 0.5338078291814946, 
                         0.5462633451957295, 0.47330960854092524, 0.5462633451957295, 
                         0.4555160142348754, 0.5427046263345195, 0.5480427046263345, 
                         0.5480427046263345]
    
    spec1_dsigma_eucl = [0.46619217081850534, 0.45729537366548045, 0.45907473309608543, 
                         0.4555160142348754, 0.44661921708185054, 0.4483985765124555, 
                         0.4483985765124555, 0.44661921708185054, 0.4395017793594306, 
                         0.44483985765124556, 0.44661921708185054, 0.44483985765124556, 
                         0.4395017793594306, 0.4430604982206406, 0.45195729537366547, 
                         0.45195729537366547]
    
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
    fig.savefig(os.path.join('plots', 'hoof_spec_dsigma_5_plot.png'), bbox_inches='tight', dpi=300)
    
def all_hoof_falseDensity_rbf():
    
    spec1_dtw = [0.47686832740213525, 0.4750889679715303, 0.46441281138790036, 
                 0.4928825622775801, 0.4750889679715303, 0.4608540925266904, 
                 0.47686832740213525, 0.49466192170818507, 0.45729537366548045, 
                 0.4786476868327402, 0.48398576512455516, 0.49466192170818507, 
                 0.46441281138790036, 0.5124555160142349, 0.4928825622775801, 
                 0.49644128113879005]
    
    spec1_haus = [0.4697508896797153, 0.4377224199288256, 0.4128113879003559, 
                  0.40213523131672596, 0.5071174377224199, 0.4288256227758007, 
                  0.4128113879003559, 0.3914590747330961, 0.5071174377224199, 
                  0.42526690391459077, 0.4128113879003559, 0.38434163701067614, 
                  0.5266903914590747, 0.42526690391459077, 0.3896797153024911, 
                  0.38434163701067614]
    # sigma = 1
    spec1_corr = [0.5124555160142349, 0.4822064056939502, 0.4750889679715303, 
                  0.4750889679715303, 0.498220640569395, 0.4893238434163701, 
                  0.47153024911032027, 0.46441281138790036, 0.4893238434163701, 
                  0.4697508896797153, 0.4626334519572954, 0.46619217081850534, 
                  0.4875444839857651, 0.4679715302491103, 0.45907473309608543, 
                  0.4608540925266904]
    
    spec1_eucl = [0.4181494661921708, 0.3683274021352313, 0.3665480427046263, 
                  0.42704626334519574, 0.42170818505338076, 0.3914590747330961, 
                  0.3914590747330961, 0.44483985765124556, 0.42170818505338076, 
                  0.3914590747330961, 0.4092526690391459, 0.45195729537366547, 
                  0.42526690391459077, 0.3861209964412811, 0.4181494661921708, 
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
    fig.savefig(os.path.join('plots', 'hoof_spec_rbf_5_plot.png'), bbox_inches='tight', dpi=300)
    
# cluster_logs/all_hoof/falseDensity/log_standardized_3.log
def all_hoof3_falseDensity():
    ''' 1 - Normed values for spectral clustering
    '''
    # mth=1:4 and nbins = 10:40
    # cluster_logs/all_hoof/falseDensity/   : standardized feats : 
    kmeans = [0.8380782918149466, 0.8185053380782918, 0.8238434163701067, 
              0.8220640569395018, 0.8434163701067615, 0.8274021352313167, 
              0.8238434163701067, 0.8238434163701067, 0.8576512455516014, 
              0.8362989323843416, 0.8327402135231317, 0.8380782918149466, 
              0.8540925266903915, 0.8416370106761566, 0.8434163701067615, 
              0.8451957295373665]
    spec1_dtw = [0.5106761565836299, 0.49466192170818507, 0.49644128113879005, 
                 0.49466192170818507, 0.5106761565836299, 0.498220640569395, 
                 0.49110320284697506, 0.4928825622775801, 0.5213523131672598, 
                 0.5106761565836299, 0.5088967971530249, 0.505338078291815, 
                 0.5302491103202847, 0.5195729537366548, 0.5088967971530249, 
                 0.5071174377224199]
    spec1_haus = [0.6583629893238434, 0.4181494661921708, 0.45195729537366547, 
                  0.4501779359430605, 0.6619217081850534, 0.505338078291815, 
                  0.44661921708185054, 0.45907473309608543, 0.6565836298932385, 
                  0.4893238434163701, 0.4626334519572954, 0.4822064056939502, 
                  0.6601423487544484, 0.4893238434163701, 0.4804270462633452, 
                  0.48576512455516013]
    spec1_corr = [0.6281138790035588, 0.5925266903914591, 0.5871886120996441, 
                  0.5871886120996441, 0.6156583629893239, 0.594306049822064, 
                  0.5889679715302492, 0.5907473309608541, 0.6174377224199288, 
                  0.597864768683274, 0.5907473309608541, 0.5889679715302492, 
                  0.6192170818505338, 0.5925266903914591, 0.5925266903914591, 
                  0.5765124555160143]
    spec1_eucl = [0.6868327402135231, 0.6708185053380783, 0.5604982206405694, 
                  0.505338078291815, 0.6921708185053381, 0.6512455516014235, 
                  0.5355871886120996, 0.5106761565836299, 0.6886120996441281, 
                  0.604982206405694, 0.5409252669039146, 0.5266903914590747, 
                  0.6832740213523132, 0.5693950177935944, 0.5355871886120996, 
                  0.5302491103202847]
    
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
    fig.savefig(os.path.join('plots', 'hoof_spec_accuracy3_plot.png'), bbox_inches='tight', dpi=300)

def all_hoof3_falseDensity_sigma():
    
    spec1_dtw = [0.5640569395017794, 0.5409252669039146, 0.5498220640569395, 
                 0.5444839857651246, 0.5658362989323843, 0.5338078291814946, 
                 0.5284697508896797, 0.5338078291814946, 0.5604982206405694, 
                 0.5302491103202847, 0.5284697508896797, 0.5284697508896797, 
                 0.5711743772241993, 0.5320284697508897, 0.5302491103202847, 
                 0.5373665480427047]
    
    spec1_haus = [0.6138790035587188, 0.5106761565836299, 0.501779359430605, 
                  0.5088967971530249, 0.6281138790035588, 0.5213523131672598, 
                  0.5213523131672598, 0.5355871886120996, 0.5693950177935944, 
                  0.5284697508896797, 0.5320284697508897, 0.5302491103202847, 
                  0.5711743772241993, 0.5498220640569395, 0.5480427046263345, 
                  0.5462633451957295]
    spec1_corr = [0.5569395017793595, 0.5604982206405694, 0.5658362989323843, 
                  0.5622775800711743, 0.5604982206405694, 0.5729537366548043, 
                  0.5604982206405694, 0.5622775800711743, 0.5604982206405694, 
                  0.5711743772241993, 0.5676156583629893, 0.5640569395017794, 
                  0.5640569395017794, 0.5622775800711743, 0.5640569395017794, 
                  0.5604982206405694]
    
    spec1_eucl = [0.6227758007117438, 0.6263345195729537, 0.5249110320284698, 
                  0.505338078291815, 0.6281138790035588, 0.6405693950177936, 
                  0.5355871886120996, 0.5160142348754448, 0.6245551601423488, 
                  0.6174377224199288, 0.5284697508896797, 0.5160142348754448, 
                  0.6192170818505338, 0.599644128113879, 0.5249110320284698, 
                  0.5160142348754448]
    
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
    fig.savefig(os.path.join('plots', 'hoof_spec_dsigma_3_plot.png'), bbox_inches='tight', dpi=300)

def all_hoof3_falseDensity_rbf():
    
    spec1_dtw = [0.5551601423487544, 0.5516014234875445, 0.5373665480427047, 
                 0.5355871886120996, 0.5587188612099644, 0.5266903914590747, 
                 0.5177935943060499, 0.5213523131672598, 0.5658362989323843, 
                 0.5195729537366548, 0.5160142348754448, 0.5231316725978647, 
                 0.5640569395017794, 0.501779359430605, 0.5106761565836299, 
                 0.5124555160142349]
    
    spec1_haus = [0.498220640569395, 0.42170818505338076, 0.41459074733096085, 
                  0.42170818505338076, 0.5231316725978647, 0.3861209964412811, 
                  0.4181494661921708, 0.42704626334519574, 0.5676156583629893, 
                  0.40569395017793597, 0.42704626334519574, 0.4110320284697509, 
                  0.5604982206405694, 0.41637010676156583, 0.4359430604982206, 
                  0.4288256227758007]
    # sigma = 1
    spec1_corr = [0.5622775800711743, 0.4928825622775801, 0.5284697508896797, 
                  0.49466192170818507, 0.5711743772241993, 0.49644128113879005, 
                  0.5249110320284698, 0.49644128113879005, 0.5765124555160143, 
                  0.50355871886121, 0.5409252669039146, 0.5391459074733096, 
                  0.5800711743772242, 0.49644128113879005, 0.5427046263345195, 
                  0.45729537366548045]
    
    spec1_eucl = [0.40213523131672596, 0.4234875444839858, 0.4128113879003559, 
                  0.4377224199288256, 0.40747330960854095, 0.40391459074733094, 
                  0.4430604982206406, 0.4412811387900356, 0.4110320284697509, 
                  0.40391459074733094, 0.4483985765124555, 0.4377224199288256, 
                  0.4092526690391459, 0.40391459074733094, 0.4430604982206406, 
                  0.4608540925266904]
    
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
    fig.savefig(os.path.join('plots', 'hoof_spec_rbf_3_plot.png'), bbox_inches='tight', dpi=300)


if __name__ == '__main__':

    all_hoof_falseDensity()
    all_hoof_falseDensity_sigma()
    all_hoof_falseDensity_rbf()
    all_hoof3_falseDensity()
    all_hoof3_falseDensity_sigma()
    all_hoof3_falseDensity_rbf()
    
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
