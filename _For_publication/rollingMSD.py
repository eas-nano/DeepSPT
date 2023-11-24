# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import seaborn as sns
import pickle
from joblib import Parallel, delayed
import time, os, sys
from tqdm import tqdm
from scipy.optimize import curve_fit
from sklearn.metrics import (roc_auc_score, roc_curve, 
                             precision_recall_curve, 
                             classification_report,
                             confusion_matrix)
from deepspt_src import (create_fingerprint_track, 
                  add_features, 
                  find_models_for,
                  find_models_for_from_path,
                  load_UnetModels,
                  load_UnetModels_directly,
                  make_preds,
                  ensemble_scoring,
                  postprocess_pred,
                  plot_diffusion,
                  timepoint_confidence_plot,
                  global_transition_probs,
                  behavior_TDP,
                  find_segments)
import os


def rollingMSD_outputter(t1, window_size=20):
    msds = rollMSD(t1, window=window_size)
    a = []
    for m in msds:
        xdata = np.array(range(len(m)))
        popt, pcov = curve_fit(msd_func, xdata, m, 
                            bounds=[(0,0,0),(100,2,100)], 
                            maxfev=1000)
        a.append(popt[1])

    a = np.hstack(a)
    pred = np.ones(len(msds))*10
    pred[a<=0.7] = 3
    pred[(0.7<a) & (a<1.3)] = 0
    pred[a>=1.3] = 1
    assert 10 not in pred

    return pred 


def SquareDist(x0, x1, y0, y1):
    return (x1 - x0) ** 2 + (y1 - y0) ** 2


def msd(t):
    x, y = t[:,0], t[:,1]
    N = len(x)
    msd = []
    for lag in range(1, N):
        msd.append(
            np.mean(
                [
                    SquareDist(x[j], x[j + lag], y[j], y[j + lag])
                    for j in range(len(x) - lag)
                ]
            )
        )
    return np.array(msd)


def squaredist(x0, x1, y0, y1, z0, z1):
    return np.mean((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2)


def rollMSD(trace, window=5, verbose=False):
    """
    Computes the mean squared displacement (msd) for a trajectory (x,y) up to
    window.
    """
    if trace.shape[1]==2:
        x, y = trace[:,0], trace[:,1]
        z = np.zeros_like(x)
    if trace.shape[1]==3:
        x, y, z = trace[:,0], trace[:,1], trace[:,2]

    rolling_msd = []
    for j in range(len(x)):
        lower = max(0, j-window//2)
        upper = min(len(x)-1, j+window//2)
        if window%2!=0:
            upper = min(len(x)-1, j+window//2+1)
        if window-(upper-lower):
            if lower == 0:
                upper += window-(upper-lower)
            if upper == len(x)-1:
                lower -= window-(upper-lower)
        upper = min(len(x)-1, upper)
        lower = max(0, lower)
        lag_msds = [squaredist(x[lower:upper][:-lag], x[lower:upper][lag:],
                                y[lower:upper][:-lag], y[lower:upper][lag:],
                                z[lower:upper][:-lag], z[lower:upper][lag:])
                                    for lag in range(1, min(len(x)-2, window))]
        rolling_msd.append(lag_msds)
        if verbose==True:
            print(lower, upper, lag_msds, window)
    
    return np.array(rolling_msd)


def msd_func(x, D, alpha, offset):
    return 2 * D * (x) ** alpha + offset


datapath = '../_Data/Simulated_diffusion_tracks/'
test_X_path = '2022422185_SimDiff_indeptest_dim2_ntraces20000_Drandom0.0001-0.5_dt1.0e+00_N5-600_B0.05-0.25_R5-25_subA0-0.7_superA1.3-2_Q1-16_X.pkl'
test_y_path = '2022422185_SimDiff_indeptest_dim2_Drandom0.0001-0.5_dt1.0e+00_N5-600_B0.05-0.25_R5-25_subA0-0.7_superA1.3-2_Q1-16_timeresolved_y.pkl'

tracks = np.array(pickle.load(open(datapath+test_X_path, 'rb')), dtype=object)
testy = np.array(pickle.load(open(datapath+test_y_path, 'rb')), dtype=object)
print(tracks.shape, testy.shape)

# %%
min_len_pred = 10
window_size = 10

rerun_rollingMSD = False
if rerun_rollingMSD:
    pred_list = []
    msd_list = []

    t = time.time()
    pred_list = Parallel(n_jobs=100)(delayed(rollingMSD_outputter)(t1) 
                    for i, t1 in enumerate(tracks))
    print(len(pred_list))
    print('time', time.time()-t)

    acc = np.mean(np.hstack(testy)==np.hstack(pred_list))
    track_acc = [np.mean(testy[i]==pred_list[i]) for i in range(len(pred_list))]

    plt.figure()
    plt.hist(track_acc)

    results_dict = {'pred_list':pred_list, 'track_acc':track_acc, 'acc':acc}
    pickle.dump(results_dict, open('baseline_methods/rollingMSD/rollingMSD_results.pkl', 'wb'))

# %%


results_dict = pickle.load(open('baseline_methods/rollingMSD/rollingMSD_results.pkl', 'rb'))
results_dict.keys()

pred_list = results_dict['pred_list']
track_acc = results_dict['track_acc']
acc = results_dict['acc']
plt.figure()
plt.hist(track_acc)
print(acc)
print(np.median(track_acc))

testy_3classes = []
for t in testy.copy():
    t = np.array(t)
    t[t==2] = 3
    testy_3classes.append(t)
testy_3classes = np.array(testy_3classes, dtype=object)

pred_list_2classes = []
testy_2classes = []
for t,p in zip(testy.copy(), pred_list):
    t = np.array(t)
    t[t==2] = 3
    t[t==1] = 0
    testy_2classes.append(t)

    p = np.array(p)
    p[p==1] = 0
    pred_list_2classes.append(p)
testy_2classes = np.array(testy_2classes, dtype=object)
pred_list_2classes = np.array(pred_list_2classes, dtype=object)

acc_3classes = np.mean(np.hstack(testy_3classes)==np.hstack(pred_list))
track_acc_3classes = [np.mean(testy_3classes[i]==pred_list[i]) for i in range(len(pred_list))]
acc_2classes = np.mean(np.hstack(testy_2classes)==np.hstack(pred_list_2classes))
track_acc_2lasses = [np.mean(testy_2classes[i]==pred_list_2classes[i]) for i in range(len(pred_list_2classes))]


plt.figure()
plt.hist(track_acc_3classes)
print(acc_3classes)
print(np.median(track_acc_3classes))

plt.figure()
plt.hist(track_acc_2lasses)
print(acc_2classes)
print(np.median(track_acc_2lasses))

flat_test_true = np.hstack(pred_list)
flat_test_pred = np.hstack(testy)

cf_matrix = confusion_matrix(flat_test_true, flat_test_pred)
group_counts = ["{0:0.0f}K".format(value/1000) for value in
                cf_matrix.flatten()]

cf_matrix = confusion_matrix(flat_test_true, flat_test_pred, normalize='true')         
group_percentages = ["{0:.0%}".format(value) for value in
                     cf_matrix.flatten()]
labels = [f"{v3}\n{v2}" for v2, v3 in zip(group_counts,group_percentages)]
labels = np.asarray(labels).reshape(4,4)

fontsize = 22

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
from sklearn.metrics import f1_score
f1_ = f1_score(flat_test_true, flat_test_pred, average='macro')
plt.title('N: {}, Accuracy: {:.3f}, F1: {:.3f}'.format(len(flat_test_true), flat_acc, f1_), size=24)
plt.tight_layout()
#plt.savefig('/paper_figures/4class_rollMSD_confusion_matrix.pdf')
plt.show()
print(classification_report(flat_test_true, flat_test_pred, target_names=diffs))
print('Accuracy:', np.mean(np.array(flat_test_pred)==np.array(flat_test_true)))


flat_test_true = np.hstack(pred_list)
flat_test_pred = np.hstack(testy_3classes)

cf_matrix = confusion_matrix(flat_test_true, flat_test_pred)
group_counts = ["{0:0.0f}K".format(value/1000) for value in
                cf_matrix.flatten()]

cf_matrix = confusion_matrix(flat_test_true, flat_test_pred, normalize='true')         
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
flat_acc = np.mean(flat_test_pred == flat_test_true)
from sklearn.metrics import f1_score
f1_ = f1_score(flat_test_true, flat_test_pred, average='macro')
plt.title('N: {}, Accuracy: {:.3f}, F1: {:.3f}'.format(len(flat_test_true), flat_acc, f1_), size=24)
plt.tight_layout()
#plt.savefig('/paper_figures/3class_rollMSD_confusion_matrix.pdf')
plt.show()
print(classification_report(flat_test_true, flat_test_pred, target_names=diffs))
print('Accuracy:', np.mean(np.array(flat_test_pred)==np.array(flat_test_true)))


flat_test_true = np.hstack(testy_2classes)
flat_test_pred = np.hstack(pred_list_2classes)

cf_matrix = confusion_matrix(flat_test_true, flat_test_pred)
group_counts = ["{0:0.0f}K".format(value/1000) for value in
                cf_matrix.flatten()]

cf_matrix = confusion_matrix(flat_test_true, flat_test_pred, normalize='true')         
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
flat_acc = np.mean(flat_test_pred == flat_test_true)
from sklearn.metrics import f1_score
f1_ = f1_score(flat_test_true, flat_test_pred, average='macro')
plt.title('N: {}, Accuracy: {:.3f}, F1: {:.3f}'.format(len(flat_test_true), flat_acc, f1_), size=24)
plt.tight_layout()
#plt.savefig('/paper_figures/2class_rollMSD_confusion_matrix.pdf')
plt.show()
print(classification_report(flat_test_true, flat_test_pred, target_names=diffs))
print('Accuracy:', np.mean(np.array(flat_test_pred)==np.array(flat_test_true)))

# %%
