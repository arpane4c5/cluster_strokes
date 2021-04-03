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

# cluster_logs/all_hoof/trueBGDensity/log_standardized.log and log_standardized_3.log
def all_hoof_trueBGDensity():
    ''' 1 - Normed values for spectral clustering
    '''
    # mth=1:4 and nbins = 10:40
    # cluster_logs/all_hoof/trueBGDensity/   : standardized feats : 
    kmeans = [0.498220640569395, 0.4483985765124555, 0.4306049822064057, 0.4626334519572954, 0.5142348754448398, 0.43238434163701067, 0.43416370106761565, 0.4412811387900356, 0.5320284697508897, 0.46619217081850534, 0.44661921708185054, 0.47330960854092524, 0.5284697508896797, 0.4412811387900356, 0.4501779359430605, 0.4501779359430605]
    spec1_norm_dtw = [0.5249110320284698, 0.5320284697508897, 0.5106761565836299, 0.48398576512455516, 0.5124555160142349, 0.5444839857651246, 0.5444839857651246, 0.5195729537366548, 0.49644128113879005, 0.49466192170818507, 0.5213523131672598, 0.5106761565836299, 0.5622775800711743, 0.5693950177935944, 0.5320284697508897, 0.5177935943060499]
    spec1_norm_haus = [0.41637010676156583, 0.4288256227758007, 0.4199288256227758, 0.44661921708185054, 0.3790035587188612, 0.42170818505338076, 0.41459074733096085, 0.42170818505338076, 0.3736654804270463, 0.40747330960854095, 0.3807829181494662, 0.40569395017793597, 0.40569395017793597, 0.37544483985765126, 0.3701067615658363, 0.4110320284697509] 
    spec1_norm_corr = [0.5676156583629893, 0.6316725978647687, 0.6298932384341637, 0.6263345195729537, 0.599644128113879, 0.6370106761565836, 0.6334519572953736, 0.6298932384341637, 0.6138790035587188, 0.6263345195729537, 0.6316725978647687, 0.6316725978647687, 0.597864768683274, 0.6263345195729537, 0.6192170818505338, 0.6281138790035588]
    spec1_norm_eucl = [0.5569395017793595, 0.5533807829181495, 0.5676156583629893, 0.5711743772241993, 0.5729537366548043, 0.5551601423487544, 0.5765124555160143, 0.5693950177935944, 0.5622775800711743, 0.5516014234875445, 0.5658362989323843, 0.5676156583629893, 0.5569395017793595, 0.5498220640569395, 0.5480427046263345, 0.5480427046263345]
    
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
    fig.savefig(os.path.join('plots', 'hoofTrueBGDen_spec_accuracy5_plot.png'), bbox_inches='tight', dpi=300)
    

def all_hoof_trueBGDensity_sigma():
    
    spec1_dsigma_dtw = [0.5640569395017794, 0.5088967971530249, 0.5071174377224199, 0.501779359430605, 0.5533807829181495, 0.5142348754448398, 0.5124555160142349, 0.505338078291815, 0.5516014234875445, 0.5071174377224199, 0.50355871886121, 0.5106761565836299, 0.5516014234875445, 0.5160142348754448, 0.5088967971530249, 0.5088967971530249]
    
    spec1_dsigma_haus = [0.4395017793594306, 0.4626334519572954, 0.4608540925266904, 0.47153024911032027, 0.4199288256227758, 0.45195729537366547, 0.4483985765124555, 0.45195729537366547, 0.4234875444839858, 0.40569395017793597, 0.4412811387900356, 0.4359430604982206, 0.3914590747330961, 0.3932384341637011, 0.3505338078291815, 0.3469750889679715]

    spec1_dsigma_corr = [0.5889679715302492, 0.6334519572953736, 0.6245551601423488, 0.5960854092526691, 0.3932384341637011, 0.6352313167259787, 0.6512455516014235, 0.6103202846975089, 0.4679715302491103, 0.6441281138790036, 0.6156583629893239, 0.6120996441281139, 0.6459074733096085, 0.6245551601423488, 0.6138790035587188, 0.604982206405694]
    
    spec1_dsigma_eucl = [0.5604982206405694, 0.5658362989323843, 0.5658362989323843, 0.5640569395017794, 0.5693950177935944, 0.5818505338078291, 0.5836298932384342, 0.5676156583629893, 0.5640569395017794, 0.5693950177935944, 0.5889679715302492, 0.5800711743772242, 0.5551601423487544, 0.5782918149466192, 0.5836298932384342, 0.5818505338078291]
    
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
    fig.savefig(os.path.join('plots', 'hoofTrueBGDen_spec_dsigma_5_plot.png'), bbox_inches='tight', dpi=300)
    
def all_hoof_trueBGDensity_rbf():
    
    spec1_dtw = [0.5871886120996441, 0.5871886120996441, 0.5854092526690391, 0.5854092526690391, 0.5854092526690391, 0.5854092526690391, 0.5836298932384342, 0.5711743772241993, 0.5782918149466192, 0.5871886120996441, 0.49644128113879005, 0.5907473309608541, 0.5551601423487544, 0.39501779359430605, 0.3914590747330961, 0.39501779359430605]
    
    spec1_haus = [0.4359430604982206, 0.44483985765124556, 0.48398576512455516, 0.4412811387900356, 0.3790035587188612, 0.400355871886121, 0.3309608540925267, 0.39679715302491103, 0.3914590747330961, 0.4110320284697509, 0.41459074733096085, 0.3807829181494662, 0.3932384341637011, 0.44661921708185054, 0.3469750889679715, 0.3879003558718861] 
    # sigma = 1
    spec1_corr = [0.5604982206405694, 0.6334519572953736, 0.6227758007117438, 0.5889679715302492, 0.5925266903914591, 0.6245551601423488, 0.6263345195729537, 0.603202846975089, 0.5960854092526691, 0.6263345195729537, 0.6209964412811388, 0.6067615658362989, 0.5960854092526691, 0.6227758007117438, 0.6156583629893239, 0.5729537366548043]
    
    spec1_eucl = [0.6743772241992882, 0.6779359430604982, 0.6725978647686833, 0.6637010676156584, 0.6708185053380783, 0.4608540925266904, 0.6761565836298933, 0.4928825622775801, 0.6690391459074733, 0.35765124555160144, 0.5302491103202847, 0.5142348754448398, 0.6156583629893239, 0.47153024911032027, 0.6565836298932385, 0.5249110320284698]
    
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
    fig.savefig(os.path.join('plots', 'hoofTrueBGDen_spec_rbf_5_plot.png'), bbox_inches='tight', dpi=300)
    
# cluster_logs/all_hoof/falseDensity/log_standardized_3.log
def all_hoof3_trueBGDensity():
    ''' 1 - Normed values for spectral clustering
    '''
    # mth=1:4 and nbins = 10:40
    # cluster_logs/all_hoof/trueBGDensity/   : standardized feats : 
    kmeans = [0.8149466192170819, 0.8167259786476868, 0.8113879003558719, 0.8042704626334519, 0.8113879003558719, 0.798932384341637, 0.791814946619217, 0.7935943060498221, 0.8042704626334519, 0.798932384341637, 0.800711743772242, 0.7953736654804271, 0.806049822064057, 0.791814946619217, 0.798932384341637, 0.7935943060498221]
    spec1_dtw = [0.7811387900355872, 0.7491103202846975, 0.7224199288256228, 0.5693950177935944, 0.802491103202847, 0.791814946619217, 0.7526690391459074, 0.5693950177935944, 0.797153024911032, 0.797153024911032, 0.7935943060498221, 0.5373665480427047, 0.791814946619217, 0.802491103202847, 0.7935943060498221, 0.5782918149466192]
    spec1_haus = [0.5213523131672598, 0.6103202846975089, 0.40391459074733094, 0.40391459074733094, 0.5071174377224199, 0.5373665480427047, 0.5658362989323843, 0.4128113879003559, 0.5604982206405694, 0.5640569395017794, 0.47330960854092524, 0.4359430604982206, 0.5106761565836299, 0.5249110320284698, 0.4608540925266904, 0.44483985765124556] 
    spec1_corr = [0.6779359430604982, 0.6014234875444839, 0.5444839857651246, 0.5516014234875445, 0.6850533807829181, 0.5640569395017794, 0.5444839857651246, 0.5320284697508897, 0.6476868327402135, 0.5693950177935944, 0.5569395017793595, 0.5462633451957295, 0.5284697508896797, 0.5676156583629893, 0.5800711743772242, 0.5729537366548043]
    spec1_eucl = [0.797153024911032, 0.7704626334519573, 0.7597864768683275, 0.7437722419928826, 0.7953736654804271, 0.7704626334519573, 0.7669039145907474, 0.7455516014234875, 0.8078291814946619, 0.7686832740213523, 0.7686832740213523, 0.7562277580071174, 0.8078291814946619, 0.7758007117437722, 0.7669039145907474, 0.7686832740213523]
    
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
    fig.savefig(os.path.join('plots', 'hoofTrueBGDen_spec_accuracy3_plot.png'), bbox_inches='tight', dpi=300)

def all_hoof3_trueBGDensity_sigma():
    
    spec1_dtw = [0.7686832740213523, 0.5427046263345195, 0.5355871886120996, 0.5355871886120996, 0.806049822064057, 0.5338078291814946, 0.5355871886120996, 0.5338078291814946, 0.8078291814946619, 0.5338078291814946, 0.5320284697508897, 0.5320284697508897, 0.8096085409252669, 0.5284697508896797, 0.5266903914590747, 0.5284697508896797]
    
    spec1_haus = [0.5622775800711743, 0.6192170818505338, 0.5444839857651246, 0.5177935943060499, 0.5444839857651246, 0.5836298932384342, 0.5462633451957295, 0.5302491103202847, 0.5516014234875445, 0.5782918149466192, 0.49644128113879005, 0.49110320284697506, 0.5249110320284698, 0.5444839857651246, 0.5462633451957295, 0.46619217081850534] 
    spec1_corr = [0.6459074733096085, 0.6352313167259787, 0.6459074733096085, 0.6352313167259787, 0.6227758007117438, 0.6476868327402135, 0.6441281138790036, 0.6441281138790036, 0.6334519572953736, 0.6565836298932385, 0.6530249110320284, 0.6512455516014235, 0.6298932384341637, 0.6565836298932385, 0.6619217081850534, 0.6548042704626335]
    
    spec1_eucl = [0.8185053380782918, 0.802491103202847, 0.8078291814946619, 0.802491103202847, 0.8238434163701067, 0.8096085409252669, 0.802491103202847, 0.798932384341637, 0.8291814946619217, 0.8096085409252669, 0.802491103202847, 0.797153024911032, 0.8202846975088968, 0.791814946619217, 0.7864768683274022, 0.7704626334519573]
    
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
    fig.savefig(os.path.join('plots', 'hoofTrueBGDen_spec_dsigma_3_plot.png'), bbox_inches='tight', dpi=300)

def all_hoof3_trueBGDensity_rbf():
    
    spec1_dtw = [0.8131672597864769, 0.5427046263345195, 0.5373665480427047, 0.5409252669039146, 0.8149466192170819, 0.5409252669039146, 0.5391459074733096, 0.5355871886120996, 0.8202846975088968, 0.5462633451957295, 0.5462633451957295, 0.5462633451957295, 0.8220640569395018, 0.5427046263345195, 0.5444839857651246, 0.5444839857651246]
    
    spec1_haus = [0.5658362989323843, 0.5818505338078291, 0.5195729537366548, 0.5160142348754448, 0.5409252669039146, 0.5516014234875445, 0.5355871886120996, 0.5177935943060499, 0.5622775800711743, 0.498220640569395, 0.48576512455516013, 0.4875444839857651, 0.5284697508896797, 0.5569395017793595, 0.4679715302491103, 0.4697508896797153] 
    # sigma = 1
    spec1_corr = [0.6886120996441281, 0.6850533807829181, 0.6761565836298933, 0.6743772241992882, 0.6921708185053381, 0.6868327402135231, 0.6761565836298933, 0.6743772241992882, 0.6868327402135231, 0.6814946619217082, 0.6779359430604982, 0.6779359430604982, 0.6690391459074733, 0.6761565836298933, 0.6672597864768683, 0.6601423487544484]
    
    spec1_eucl = [0.400355871886121, 0.398576512455516, 0.398576512455516, 0.398576512455516, 0.398576512455516, 0.3914590747330961, 0.3914590747330961, 0.3914590747330961, 0.398576512455516, 0.3914590747330961, 0.3914590747330961, 0.3914590747330961, 0.40213523131672596, 0.40391459074733094, 0.3896797153024911, 0.3914590747330961]
    
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
    fig.savefig(os.path.join('plots', 'hoofTrueBGDen_spec_rbf_3_plot.png'), bbox_inches='tight', dpi=300)


if __name__ == '__main__':

    all_hoof_trueBGDensity()
    all_hoof_trueBGDensity_sigma()
    all_hoof_trueBGDensity_rbf()
    all_hoof3_trueBGDensity()
    all_hoof3_trueBGDensity_sigma()
    all_hoof3_trueBGDensity_rbf()
    
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
