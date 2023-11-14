# %%
import matplotlib.pyplot as plt
import numpy as np
from deepspt_src import *
import matplotlib.pyplot as plt
import random
from glob import glob
from torch.utils.data import DataLoader
import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
import pickle
from global_config import globals
import seaborn as sns

#**********************Initiate variables**********************

datasets = ['SimDiff_dim2_ntraces300000_Drandom0.0001-0.5_dt1.0e+00_N5-600_B0.05-0.25_R5-25_subA0-0.7_superA1.3-2_Q1-16']
datasets = ['SimDiff_dim3_ntraces300000_Drandom0.0001-0.5_dt1.0e+00_N5-600_B0.05-0.25_R5-25_subA0-0.7_superA1.3-2_Q1-16']

methods = ['XYZ_SL_DP']

# find models
dim = 3 if 'dim3' in datasets[0] else 2

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

use_mlflow = False
# find the model
if use_mlflow: # bit troublesome if not on same machine/server
    import mlflow
    mlflow.set_tracking_uri('file:'+join(os.getcwd(), join("Unet_results", "mlruns")))
    best_models_sorted = find_models_for(datasets, methods)
else:
    # not sorted tho
    if dim==2:
        modeldir = '3'
    elif dim==3:
        modeldir = '36'
    path = 'mlruns/{}'.format(modeldir)
    best_models_sorted = find_models_for_from_path(path)


"""Evaluate on independent test set"""

datapath = '_Data/Simulated_diffusion_tracks/'

if dim==2:
    # R = 7-25
    filenames_X = ['2022422185_SimDiff_indeptest_dim2_ntraces20000_Drandom0.0001-0.5_dt1.0e+00_N5-600_B0.05-0.25_R5-25_subA0-0.7_superA1.3-2_Q1-16_X.pkl']
    filenames_y = ['2022422185_SimDiff_indeptest_dim2_Drandom0.0001-0.5_dt1.0e+00_N5-600_B0.05-0.25_R5-25_subA0-0.7_superA1.3-2_Q1-16_timeresolved_y.pkl']
elif dim==3:
    # # # # R = 7-25 3 D
    filenames_X = ['2022422216_SimDiff_indeptest_dim3_ntraces20000_Drandom0.0001-0.5_dt1.0e+00_N5-600_B0.05-0.25_R5-25_subA0-0.7_superA1.3-2_Q1-16_X.pkl']
    filenames_y = ['2022422216_SimDiff_indeptest_dim3_Drandom0.0001-0.5_dt1.0e+00_N5-600_B0.05-0.25_R5-25_subA0-0.7_superA1.3-2_Q1-16_timeresolved_y.pkl']

# prep X and y
X_to_eval = np.array(pickle.load(open(datapath+filenames_X[0], 'rb')), dtype=object)
y_to_eval = np.array(pickle.load(open(datapath+filenames_y[0], 'rb')), dtype=object)

features = ['XYZ', 'SL', 'DP']
X_to_eval = add_features(X_to_eval, features)
X_to_eval = [x-x[0] for x in X_to_eval]
max_len = np.max([len(x) for x in X_to_eval])

print(X_to_eval[0].shape)
print(np.array(y_to_eval[0]).shape)

"""Load or pred independent test data"""
savepath = 'Unet_results/predictions/'
savename = 'Sim_data_dim'+str(dim)+'_results_mldir'+best_models_sorted[0].split('/')[0]+'_'+filenames_X[0]+'.pkl'
results_dict = load_or_pred_for_simdata(X_to_eval, y_to_eval, globals, savepath, savename, best_models_sorted,
                                        filenames_X, filenames_y, min_max_len=max_len, dim=dim)

#thresholded_acc_idx = get_ids_by_acc_threshold(acc, acc_threshold = 0, above=True)
ensemble_pred = results_dict['ensemble']
ensemble_score = results_dict['ensemble_score']
print(np.mean(np.hstack(ensemble_pred)==np.hstack(y_to_eval)))

# %%
from scipy.stats import mode
# clean diffusion pred
clean_ensemble_pred = np.hstack([mode(p)[0] for p in ensemble_pred[:4000]])
clean_y_to_eval = np.hstack([np.unique(y) for y in y_to_eval[:4000]])
print(np.mean(clean_ensemble_pred==clean_y_to_eval))

fontsize = 22
flat_test_true = clean_y_to_eval
flat_test_pred = clean_ensemble_pred
cf_matrix = confusion_matrix(flat_test_true, flat_test_pred)
group_counts = ["{0:0.0f}K".format(value/1000) for value in
                cf_matrix.flatten()]

cf_matrix = confusion_matrix(flat_test_true, flat_test_pred, normalize='true')         
group_percentages = ["{0:.0%}".format(value) for value in
                     cf_matrix.flatten()]
labels = [f"{v3}\n{v2}" for v2, v3 in zip(group_counts,group_percentages)]
labels = np.asarray(labels).reshape(4,4)

plt.figure(figsize=(8,6))
ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues', annot_kws={"fontsize":fontsize})
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)
plt.xlabel('Predicted', size=20)
plt.ylabel('True', size=20)
diffs =  ['ND', 'DM', 'CD', 'SD']
plt.xticks(np.linspace(0.5, 3.5, 4), diffs, rotation=45, size=20)
plt.yticks(np.linspace(0.5, 3.5, 4), diffs, rotation=0, size=20)
flat_acc = np.mean(flat_test_pred == flat_test_true)
f1_ = f1_score(flat_test_true, flat_test_pred, average='macro')
plt.title('N: {}, Accuracy: {:.3f}, F1-score: {:.3f}'.format(len(flat_test_true), flat_acc, f1_), size=24)
model_used = 'Ensemble_mldir'+best_models_sorted[0].split('/')[0]
plt.tight_layout()
plt.savefig('deepspt_results/figures/cleandiff_confusion_matrix_'+datasets[0]+'_'+model_used+'.pdf')
plt.show()


# %%
savepath = 'Unet_results/predictions/'
savename_temp = 'Sim_data_dim'+str(dim)+'_results_tempscaled_mldir'+best_models_sorted[0].split('/')[0]+'_'+filenames_X[0]+'.pkl'
temperature = 3.8537957365297553


results_dict_temp = load_or_create_TempPred(savepath, savename_temp, best_models_sorted, temperature,
                                       datapath, filenames_X[0], filenames_y[0], globals,
                                       features=['XYZ','SL','DP'])

ensemble_pred_temp = results_dict_temp['ensemble']
ensemble_score_temp = results_dict_temp['ensemble_score']

# %%
number_quantiles = 20
print('temp', temperature)
print(len(ensemble_score_temp), len(y_to_eval))
savename = 'Unet_results/figures/temp_cali_reliability_mldir'+best_models_sorted[0].split('/')[0]
reliability_plot(ensemble_score_temp, y_to_eval, number_quantiles = 20, savename=savename)
reliability_plot(ensemble_score, y_to_eval, number_quantiles = 20, savename=savename)


# %%
i = 16008
timepoint_confidence_plot(ensemble_score[i])

savename = 'deepspt_results/figures/DeepSPTpred_confidence_{}'.format(i)
timepoint_confidence_plot(ensemble_score_temp[i], savename=savename)

savename = 'deepspt_results/figures/simGT_DeepSPTpred_{}'.format(i)
compare_pred2sim_diffusion(X_to_eval[i][:,1], X_to_eval[i][:,2], 
y_to_eval[i], ensemble_pred[i], savename=savename)

i = 15909

print(i)
timepoint_confidence_plot(ensemble_score[i])
savename = 'deepspt_results/figures/DeepSPTpred_confidence_{}'.format(i)
timepoint_confidence_plot(ensemble_score_temp[i], savename=savename)

savename = 'deepspt_results/figures/simGT_DeepSPTpred_{}'.format(i)
compare_pred2sim_diffusion(X_to_eval[i][:,0], X_to_eval[i][:,2], 
y_to_eval[i], ensemble_pred[i], savename=savename)


# %%
bins = 20

xlen = [7]
ylen = [3.5]
for acci in range(len(ylen)):
    acc = np.zeros([len(y_to_eval)])
    for i in range(len(y_to_eval)):
        acc[i]= np.mean(np.hstack(y_to_eval[i])==np.hstack(ensemble_pred[i]))
    print(np.mean(acc), np.median(acc), np.std(acc))
    plt.figure(figsize=(xlen[acci],ylen[acci]))
    plt.hist(acc, bins=bins, color='steelblue', edgecolor='white', linewidth=1.2)
    plt.xlabel('Accuracy')
    plt.ylabel('Count')
    plt.title('N: {} bins: {} mean+/-std: {:.3f}+/-{:.3f} median: {:.3f}'
              .format(len(acc), bins, np.mean(acc), 
              np.std(acc), np.median(acc)))
    model_used = 'Ensemble_mldir'+best_models_sorted[0].split('/')[0]
    plt.tight_layout()
    plt.savefig('deepspt_results/figures/hist_acc_{}'.format(acci)+datasets[0]+'_'+model_used+'.pdf')


# %%
fontsize = 22
flat_test_true = np.hstack(y_to_eval)
flat_test_pred = np.hstack(ensemble_pred)

cf_matrix = confusion_matrix(flat_test_true, flat_test_pred)
group_counts = ["{0:0.0f}K".format(value/1000) for value in
                cf_matrix.flatten()]

cf_matrix = confusion_matrix(flat_test_true, flat_test_pred, normalize='true')         
group_percentages = ["{0:.0%}".format(value) for value in
                     cf_matrix.flatten()]
labels = [f"{v3}\n{v2}" for v2, v3 in zip(group_counts,group_percentages)]
labels = np.asarray(labels).reshape(4,4)

plt.figure(figsize=(8,6))
ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues', annot_kws={"fontsize":fontsize})
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)
plt.xlabel('Predicted', size=20)
plt.ylabel('True', size=20)
diffs =  ['ND', 'DM', 'CD', 'SD']
plt.xticks(np.linspace(0.5, 3.5, 4), diffs, rotation=45, size=20)
plt.yticks(np.linspace(0.5, 3.5, 4), diffs, rotation=0, size=20)
flat_acc = np.mean(flat_test_pred == flat_test_true)
f1_ = f1_score(flat_test_true, flat_test_pred, average='macro')
plt.title('N: {}, Accuracy: {:.3f}, \nF1-score: {:.3f}'.format(len(flat_test_true), flat_acc, f1_), size=24)
model_used = 'Ensemble_mldir'+best_models_sorted[0].split('/')[0]
plt.tight_layout()
plt.savefig('deepspt_results/figures/confusion_matrix_'+datasets[0]+'_'+model_used+'.pdf')
plt.show()

# %%
flat_test_true_2class = []
flat_test_pred_2class = []
for p,t in zip(ensemble_pred,y_to_eval):
    p,t = np.array(p), np.array(t)

    t[t==1] = 0
    t[t==2] = 3
    p[p==1] = 0
    p[p==2] = 3

    flat_test_true_2class.append(t)
    flat_test_pred_2class.append(p)

flat_test_true_2class = np.hstack(flat_test_true_2class)
flat_test_pred_2class = np.hstack(flat_test_pred_2class)


cf_matrix = confusion_matrix(flat_test_true_2class, flat_test_pred_2class)
group_counts = ["{0:0.0f}K".format(value/1000) for value in
                cf_matrix.flatten()]

cf_matrix = confusion_matrix(flat_test_true_2class, flat_test_pred_2class, normalize='true')         
group_percentages = ["{0:.0%}".format(value) for value in
                     cf_matrix.flatten()]
labels = [f"{v3}\n{v2}" for v2, v3 in zip(group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)

plt.figure(figsize=(8,6))
ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues', annot_kws={"fontsize":fontsize})
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)
plt.xlabel('Predicted', size=20)
plt.ylabel('True', size=20)
diffs =  ['Free', 'Rest.']
plt.xticks(np.linspace(0.5, 1.5, 2), diffs, rotation=45, size=20)
plt.yticks(np.linspace(0.5, 1.5, 2), diffs, rotation=0, size=20)
flat_acc = np.mean(flat_test_true_2class == flat_test_pred_2class)
f1_ = f1_score(flat_test_true_2class, flat_test_pred_2class, average='macro')
plt.title('N: {}, Accuracy: {:.3f}, F1-score: {:.3f}'.format(len(flat_test_true), flat_acc, f1_), size=24)
model_used = 'Ensemble_mldir'+best_models_sorted[0].split('/')[0]
plt.tight_layout()
plt.savefig('deepspt_results/figures/2class_confusion_matrix_'+datasets[0]+'_'+model_used+'.pdf')
plt.show()
print(classification_report(flat_test_true_2class, flat_test_pred_2class, target_names=diffs))
print('Accuracy:', np.mean(np.array(flat_test_pred_2class)==np.array(flat_test_true_2class)))

# %%

flat_test_true_3class = []
flat_test_pred_3class = []
for p,t in zip(ensemble_pred,y_to_eval):
    p,t = np.array(p), np.array(t)

    t[t==2] = 3
    p[p==2] = 3

    flat_test_true_3class.append(t)
    flat_test_pred_3class.append(p)

flat_test_true_3class = np.hstack(flat_test_true_3class)
flat_test_pred_3class = np.hstack(flat_test_pred_3class)


cf_matrix = confusion_matrix(flat_test_true_3class, flat_test_pred_3class)
group_counts = ["{0:0.0f}K".format(value/1000) for value in
                cf_matrix.flatten()]

cf_matrix = confusion_matrix(flat_test_true_3class, flat_test_pred_3class, normalize='true')         
group_percentages = ["{0:.0%}".format(value) for value in
                     cf_matrix.flatten()]
labels = [f"{v3}\n{v2}" for v2, v3 in zip(group_counts,group_percentages)]
labels = np.asarray(labels).reshape(3,3)

plt.figure(figsize=(8,6))
ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues', annot_kws={"fontsize":fontsize})
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)
plt.xlabel('Predicted', size=20)
plt.ylabel('True', size=20)
diffs =  ['ND', 'DM', 'Rest.']
plt.xticks(np.linspace(0.5, 2.5, 3), diffs, rotation=45, size=20)
plt.yticks(np.linspace(0.5, 2.5, 3), diffs, rotation=0, size=20)
flat_acc = np.mean(flat_test_true_3class == flat_test_pred_3class)
f1_ = f1_score(flat_test_true_3class, flat_test_pred_3class, average='macro')
plt.title('N: {}, Accuracy: {:.3f}, F1-score: {:.3f}'.format(len(flat_test_true), flat_acc, f1_), size=24)
model_used = 'Ensemble_mldir'+best_models_sorted[0].split('/')[0]
plt.tight_layout()
plt.savefig('deepspt_results/figures/3class_confusion_matrix_'+datasets[0]+'_'+model_used+'.pdf')
plt.show()
print(classification_report(flat_test_true_3class, flat_test_pred_3class, target_names=diffs))
print('Accuracy:', np.mean(np.array(flat_test_pred_3class)==np.array(flat_test_true_3class)))

# %%
