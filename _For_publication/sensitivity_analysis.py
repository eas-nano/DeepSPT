# %%
import matplotlib.pyplot as plt
import numpy as np
from deepspt_src import *
import matplotlib.pyplot as plt
import random
from glob import glob
import pickle
from global_config import globals
import os
from sensitivity_paths import *

"""
Nrange_paths_X_2D
Nrange_paths_Y_2D
Nrange_paths_X_3D
Nrange_paths_Y_3D

Qrange_paths_X_2D
Qrange_paths_Y_2D
Qrange_paths_X_3D
Qrange_paths_Y_3D

Drange_paths_X_2D
Drange_paths_Y_2D
Drange_paths_X_3D
Drange_paths_Y_3D
"""

# ****************Initiate variables**********************
methods = ['XYZ_SL_DP']
Unet_figpath = 'Unet_paper_figures/'

# global config variables
globals._parse({})

# get consistent result
seed = globals.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
device = globals.device

# data paths
subdir = 'Nrange/'
main_dir = '_Data/Simulated_diffusion_tracks/sensitivity_analysis/'+subdir
dim = 2
if dim==3:
    modeldir = '36'
    datasets = ['SimDiff_dim3_ntraces300000_Drandom0.0001-0.5_dt1.0e+00_N5-600_B0.05-0.25_R5-25_subA0-0.7_superA1.3-2_Q1-16']
    if subdir == 'Nrange/':
        paths_X, paths_Y = Nrange_paths_X_3D, Nrange_paths_Y_3D
    if subdir == 'Qrange/':
        paths_X, paths_Y = Qrange_paths_X_3D, Qrange_paths_Y_3D
    if subdir == 'Drange/':
        paths_X, paths_Y = Drange_paths_X_3D, Drange_paths_Y_3D

elif dim==2:
    modeldir = '3'
    datasets = ['SimDiff_dim2_ntraces300000_Drandom0.0001-0.5_dt1.0e+00_N5-600_B0.05-0.25_R5-25_subA0-0.7_superA1.3-2_Q1-16']
    if subdir == 'Nrange/':
        paths_X, paths_Y = Nrange_paths_X_2D, Nrange_paths_Y_2D
    if subdir == 'Qrange/':
        paths_X, paths_Y = Qrange_paths_X_2D, Qrange_paths_Y_2D
    if subdir == 'Drange/':
        paths_X, paths_Y = Drange_paths_X_2D, Drange_paths_Y_2D

# find the model
use_mlflow = False
min_max_len = 601
modelpath = 'Unet_results/mlruns/'
features = ['XYZ', 'SL', 'DP']

# find the model
if use_mlflow:
    mlflow.set_tracking_uri('file:'+join(os.getcwd(), join("Unet_results", "mlruns")))
    best_models_sorted = find_models_for(datasets, methods)
else:
    # not sorted tho
    path = 'trained_models/{}/'.format(modeldir)
    best_models_sorted = find_models_for_from_path(path)
    print(os.listdir(path))
    print(best_models_sorted)

print('_Data/Simulated_diffusion_tracks/sensitivity_analysis/'+subdir.replace('/','_dim')+str(dim)+'_results_dir.pkl')
if not os.path.exists('_Data/Simulated_diffusion_tracks/sensitivity_analysis/'+subdir.replace('/','_dim')+str(dim)+'_results_dir.pkl'):   
    results_dict = {}
    for path_x, path_y in zip(paths_X, paths_Y):
        X, y_to_eval = pickle.load(open(main_dir+path_x, 'rb')), pickle.load(open(main_dir+path_y, 'rb'))
        X = prep_tracks(X)
        X_to_eval = add_features(X, features)

        files_dict = {}
        for modelname in best_models_sorted[:3]:
            if use_mlflow:
                model = load_UnetModels(modelname, dir=modelpath, device=device)
            else:
                model = load_UnetModels_directly(modelname, device=device, dim=dim)

            tmp_dict = make_preds(model, X_to_eval, y_to_eval, min_max_len=min_max_len,
                                X_padtoken=globals.X_padtoken, y_padtoken=globals.y_padtoken,
                                batch_size=globals.batch_size, device=device)
            files_dict[modelname] = tmp_dict

        if len(list(files_dict.keys()))>=3:
            files_dict['ensemble_score'] = ensemble_scoring(files_dict[list(files_dict.keys())[0]]['masked_score'], 
                                            files_dict[list(files_dict.keys())[1]]['masked_score'], 
                                            files_dict[list(files_dict.keys())[2]]['masked_score'])
            ensemble_pred = [np.argmax(files_dict['ensemble_score'][i], axis=0) for i in range(len(files_dict['ensemble_score']))]
            files_dict['ensemble'] = ensemble_pred
        
        files_dict['y_test'] = y_to_eval
        files_dict['flat_acc'] = np.mean(np.hstack(y_to_eval)==np.hstack(ensemble_pred))
        files_dict['track_acc'] = [np.mean(y_to_eval[i]==ensemble_pred[i]) for i in range(len(ensemble_pred))]
        print(files_dict['flat_acc'], np.median(files_dict['track_acc']))
        results_dict[path_x] = files_dict

    pickle.dump(results_dict, open('_Data/Simulated_diffusion_tracks/sensitivity_analysis/'+subdir.replace('/','_dim')+str(dim)+'_results_dir.pkl', 'wb'))
    print(results_dict.keys())
    print(results_dict[path_x])
    print(results_dict[path_x].keys())
    print(len(results_dict[path_x]['ensemble']))
else:
    results_dict = pickle.load(
                open(
                    '_Data/Simulated_diffusion_tracks/sensitivity_analysis/'+subdir.replace('/','_dim')+str(dim)+'_results_dir.pkl', 'rb'))



# %%
dim = 3 # 2 or 3
subdir = 'Qrange/' # 'Nrange/' 'Drange/' 'Qrange/'
results_dict = pickle.load(
                open(
                    '_Data/Simulated_diffusion_tracks/sensitivity_analysis/'+subdir.replace('/','_dim')+str(dim)+'_results_dir.pkl', 'rb'))
keys = list(results_dict.keys())

medians_list = []
means_list = []
flat_acc_list = []
cf_matrix_list = []
cf_matrix_norm_list = []
for k in keys:
    r = results_dict[k]
    track_acc = r['track_acc']
    flat_acc = r['flat_acc']
    pred = np.hstack(r['ensemble'])
    ytest = np.hstack(r['y_test'])

    cf_matrix = confusion_matrix(ytest, pred)
    cf_matrix_list.append(cf_matrix)

    cf_matrix_norm = confusion_matrix(ytest, pred, normalize='true')
    cf_matrix_norm_list.append(cf_matrix_norm)

    means_list.append(np.mean(track_acc))
    medians_list.append(np.median(track_acc))
    flat_acc_list.append(flat_acc)

if subdir == 'Qrange/':
    xlabel = 'Steplength to localization error ratio'
    xticks = ['0.1', '0.5', '1', '2', '5', '10', '16']

if subdir == 'Nrange/':
    xlabel = 'Number of time points'
    xticks = ['5', '20', '50', '100', '200', '400', '600', '1000']

if subdir == 'Drange/':
    xlabel = 'Diffusion coefficients'
    xticks = ['0.0001-0.001', '0.001-0.01', 
              '0.01-0.1', '0.1-0.5', '0.5-1', 
              '1-2']

x = list(range(len(means_list)))
plt.figure(figsize=(6,5))
plt.scatter(x, means_list, color='navy')
plt.xticks(list(range(len(xticks))), labels=xticks, rotation=30)
plt.ylim(0,1.05)
plt.xlabel(xlabel, size=18)
plt.ylabel('Mean Track Accuracy', size=18)
plt.title('mean acc')
plt.tight_layout()
plt.savefig(Unet_figpath+'/paper_figures/SI/sensitivity_mean_track_acc_'+subdir.replace('/','')+'dim'+str(dim)+'.pdf')

plt.figure(figsize=(6,5))
plt.scatter(x, medians_list, color='navy')
plt.xticks(list(range(len(xticks))), labels=xticks, rotation=30)
plt.ylim(0,1.05)
plt.xlabel(xlabel, size=18)
plt.ylabel('Median Track Accuracy', size=18)
plt.title('median acc')
plt.tight_layout()
plt.savefig(Unet_figpath+'/paper_figures/SI/sensitivity_median_track_acc_'+subdir.replace('/','')+'dim'+str(dim)+'.pdf')

plt.figure(figsize=(6,5))
plt.scatter(x, flat_acc_list, color='navy')
plt.xticks(list(range(len(xticks))), labels=xticks, rotation=30)
plt.ylim(0,1.05)
plt.xlabel(xlabel, size=18)
plt.ylabel('Accuracy', size=18)
plt.title('acc')
plt.tight_layout()
plt.savefig(Unet_figpath+'/paper_figures/SI/sensitivity_acc_'+subdir.replace('/','')+'dim'+str(dim)+'.pdf')

# %%
