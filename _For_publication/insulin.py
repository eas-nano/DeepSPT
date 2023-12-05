# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
import sys
sys.path.append('../')
from deepspt_src import *
import matplotlib.pyplot as plt
import random
import pandas as pd
from global_config import globals
import seaborn as sns
from tqdm import tqdm
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import *
from joblib import Parallel, delayed
import os
from sklearn.neural_network import MLPClassifier

#**********************Initiate variables**********************
globals._parse({})

# defie dataset and method that model was trained on to find the model
datasets = ['SimDiff_dim2_ntraces300000_Drandom0.0001-0.5_dt1.0e+00_N5-600_B0.05-0.25_R5-25_subA0-0.7_superA1.3-2_Q1-16']
methods = ['XYZ_SL_DP']
the_data_is = '3D' if 'dim3' in datasets[0] else '2D'

# fingerprint things
fp_datapath = '../_Data/Simulated_diffusion_tracks/'
hmm_filename = 'simulated2D_HMM.json'

# define variables
xy_to_um = 1
z_to_um = 1
dim = 2
dt = 30/1000 # dt of experiment
min_trace_length = 50
min_seglen_for_FP = 5
min_pred_length = 5
num_difftypes =  4
max_change_points = 10
save_note = 'v10'
add_FP = True

hidden_layer_sizes = (100,10,10)
max_iter = 500
early_stopping = True

# get consistent result
seed = globals.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# find the model
modelpath = '../mlruns/'
use_mlflow = False
modeldir = '3'

# find the model
if use_mlflow: # bit troublesome if not on same machine/server
    import mlflow
    mlflow.set_tracking_uri('file:'+join(os.getcwd(), join("", "mlruns")))
    best_models_sorted = find_models_for(datasets, methods)
else:
    # not sorted tho
    if dim==2:
        modeldir = '3'
    elif dim==3:
        modeldir = '36'
    path = '../mlruns/{}'.format(modeldir)
    best_models_sorted = find_models_for_from_path(path)

print('Find models')
best_models_sorted

# %%
# Prep data
figpath = '/deepspt_results/figures/'
datapath = '../_Data/Insulin/data_HeLa/'
save_path = datapath

Monomer = ['Monomer_new/lys/bio_1/640', 'Monomer_new/lys/bio_2/640'] #22

PROJECT_NAMES = Monomer

tTDP_individuals = []
FP_list = []
tTDP_labels = []
ensemble_pred_list = []
ensemble_score_list = []
tracks_list = []

for pn_i, PN in enumerate(PROJECT_NAMES):
    #save_dict_name = PN.replace('/','_')+save_note+'_results_dict_mldir'+best_models_sorted[0].split('/')[0]+'.pkl'
    save_dict_name = PN.replace('/','_')+'_insulin_results_dict.pkl'
    save_PN_name = datapath+'/precomputed_files/'+save_note+PN.replace('/','_')

    PN_data = PN + '/bg_corr_all_tracked.csv'
    print(PN_data, save_PN_name)

    files_dict, X_to_eval, y_to_eval, _, _= load_or_create_resultsdict_for_rawdata(
                                                [PN_data], "not_used_in_2D", "not_used_in_2D", 
                                                globals, datapath, save_dict_name, save_path, 
                                                best_models_sorted,the_data_is, xy_to_um = xy_to_um, 
                                                z_to_um = z_to_um, min_trace_length=min_trace_length, 
                                                device=device, use_mlflow=use_mlflow)
    results_dict = files_dict[PN_data]

    tracks = X_to_eval.copy()
    if 'ensemble' in list(results_dict.keys()):
        ensemble_pred_pre = results_dict['ensemble']
        ensemble_score = results_dict['ensemble_score']
    else:
        ensemble_pred_pre = results_dict[best_models_sorted[0]]['masked_pred']
        ensemble_score = results_dict[best_models_sorted[0]]['masked_score']

    ensemble_pred = Parallel(n_jobs=50)(delayed(postprocess_pred)(ens_pred, ens_score, min_pred_length) 
                        for ens_pred, ens_score in zip(ensemble_pred_pre, ensemble_score))

    if os.path.exists(save_PN_name+'_vanillaFP.pkl'):
        FP = pickle.load(open(save_PN_name+'_vanillaFP.pkl', 'rb'))
    else:
        FP = Parallel(n_jobs=50)(delayed(create_fingerprint_track)(track, fp_datapath, hmm_filename, dim, dt, 'Normal') for track in tracks)
        
        pickle.dump(FP, open(save_PN_name+'_vanillaFP.pkl', 'wb'))

    if os.path.exists(save_PN_name+'_tTDP_FP.pkl'):
        tmp = pickle.load(open(save_PN_name+'_tTDP_FP.pkl', 'rb'))
    else:
        tmp = Parallel(n_jobs=2)(delayed(tTDP_individuals_generator)([pred], [track], max_change_points, num_difftypes, 
                                                                         fp_datapath, hmm_filename, dim=dim, dt=dt, 
                                                                         min_seglen_for_FP=min_seglen_for_FP,
                                                                         add_FP=add_FP) for (pred, track)  in zip(ensemble_pred, tracks))
        pickle.dump(tmp, open(save_PN_name+'_tTDP_FP.pkl', 'wb'))
    for es in ensemble_score:
        ensemble_score_list.append(es)
    for ep in ensemble_pred:
        ensemble_pred_list.append(ep)
    for t in tracks:
        tracks_list.append(t)
  
    tTDP_individuals.append(tmp)
    FP_list.append(FP)
    tTDP_labels.append(np.repeat(pn_i, len(tmp)))

FP_list_raw = np.vstack(FP_list)
tTDP_individuals = np.vstack(tTDP_individuals)                              
tTDP_labels = np.hstack(tTDP_labels)
ensemble_pred = np.array(ensemble_pred_list, dtype=object)
ensemble_score = ensemble_score_list
tracks = np.array(tracks_list, dtype=object)
tracks = np.array([t[:,:2] for t in tracks], dtype=object)
# 1, 4, 13, 10, 19
# %%

tracks_with_dir_idx = [i for i, e in enumerate(ensemble_pred) if 0.3<np.mean(1==e)]

i = 2141 
print (i)
# 2806 1881 2628 2141

plot_diffusion(tracks[i], ensemble_pred[i], savename='../deepspt_results/insulin_SI_figs_track'+str(i))
timepoint_confidence_plot(ensemble_score[i])


def global_difftype_occupancy_piechart(pred_argmax, savename=''):
    difftypes = ['Normal ', 'Directed ', 'Confined ', 'Subdiffusive ']
    color_list = ['blue', 'red', 'green', 'darkorange']
    flat_pred_argmax = [p for subpred in pred_argmax for p in subpred]
    lifetime = np.unique(flat_pred_argmax, return_counts=True)[1]/sum(np.unique(flat_pred_argmax, return_counts=True)[1])
    labels = [difftypes[i]+str(np.round(np.around(lifetime[i], 3)*100, 3))+'%' for i in range(len(difftypes))]
    fig = plt.figure()
    plt.pie(lifetime, labels=labels, colors=color_list, labeldistance=1.15)

    # add a circle at the center to transform it in a donut chart
    my_circle=plt.Circle( (0,0), 0.7, color='white')
    p=plt.gcf()
    p.gca().add_artist(my_circle)

    plt.axis('equal')
    fig.patch.set_facecolor('xkcd:white')
    plt.tight_layout(pad=5)
    if len(savename)>0:
        plt.savefig(savename+'.png')
        plt.savefig(savename+'.pdf')
    plt.show()

""" difftype occupancy pie chart and TDP diffusion"""
m1_norm_trans_dict, _, _ = global_transition_probs(ensemble_pred)
global_difftype_occupancy_piechart(ensemble_pred)
behavior_TDP(ensemble_pred, m1_norm_trans_dict, savename='../deepspt_results/insulin_SI_figs_tTDP_monomer.pdf')
print(PROJECT_NAMES)

# %%
