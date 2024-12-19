# %%
import numpy as np
import pickle
import matplotlib.pyplot as plt
import datetime
import sys
sys.path.append('../')
from deepspt_src import *
from global_config import globals
import warnings
from Andi_Challenge_Code.andi_datasets.datasets_theory import datasets_theory
from Andi_Challenge_Code.andi_datasets.datasets_challenge import challenge_theory_dataset
import sys
warnings.filterwarnings("ignore")

def make_andi_challenge_y_temporal(y, t, divider=1):
    y_temporal = []
    for i in range(len(y)):
        y_temp = np.ones(len(t[i])//divider)*-1
        cp = int(y[i][0])
        first_val = y[i][1]
        second_val = y[i][3]
        y_temp[:cp] = first_val
        y_temp[cp:] = second_val
        y_temporal.append(y_temp)

    return np.array(y_temporal, dtype=int)

dim = 2
datapath = '../Andi_Challenge_Code/ANDI_challenge_testsets/'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

path = '../Andi_Challenge_Code/ANDI_challenge_testsets/'
taskfile = 'X3_3D_20000_testtask3.txt'
reffile = 'X3_3D_20000_testref3.txt'

N_save = 20000
save_dataset = True
path_datasets = path+'X3_3D_{}_test'.format(N_save)
_, _, _, _, X3_, Y3_ = challenge_theory_dataset(
                                N = N_save, 
                                tasks = 3, 
                                dimensions = [2,3], 
                                min_T = 200,
                                max_T = 200,
                                N_save = N_save,
                                t_save = N_save,
                                save_dataset = save_dataset,
                                path_datasets = path_datasets,
                                load_dataset = True,)

print(path_datasets)
print(path+taskfile.split('task3.txt')[0])
assert path_datasets == path+taskfile.split('task3.txt')[0]

X3_2D = np.array(X3_[1])
Y3_2D = np.array(Y3_[1])
X3_3D = np.array(X3_[2])
Y3_3D = np.array(Y3_[2])

X3_2D_test2 = np.array([np.column_stack(
    [track[:200],track[200:]]).astype(float) for track in X3_2D])
Y3_2D_test2 = np.vstack(Y3_2D)

X3_3D_test2 = np.array([np.column_stack(
    [track[:200],track[200:400], track[400:]]).astype(float) for track in X3_3D])
Y3_3D_test2 = np.vstack(Y3_3D)

# prep data
if dim == 3:
    tracks = X3_3D_test2
    Y = Y3_3D_test2
if dim == 2:
    tracks = X3_2D_test2
    Y = Y3_2D_test2

X = [x-x[0] for x in tracks]
print(len(X), 'len X')
features = ['XYZ', 'SL', 'DP']
methods = ['XYZ_SL_DP']
X_to_eval = add_features(X, features)
y_to_eval = [np.ones(len(x))*0.5 for x in X_to_eval]

# define dataset and method that model was trained on to find the model
if dim == 2:
    modeldir = '47'
if dim == 3:
    modeldir = '46'

# find the model
dir_name = ''
modelpath = 'mlruns'

use_mlflow = False # troublesome if not on same machine as trained (mlflow) thus False

def find_models_from_path_and_name(path, modelname):
    # load from path
    files = sorted(glob(path+'*/*{}'.format(modelname), recursive=True))
    return files

modelname = '_DEEPSPT_ANDI_model.torch'
path = modelpath+'/{}/'.format(modeldir)
print(path)
best_models_sorted = find_models_from_path_and_name(path, modelname)
print(best_models_sorted) # ordered as found

if dim == 2:

    model_names = ['0049bd708cef4b6f93c46af062d66d4b',
                    '5f50dc8013ab4ffb873801015db84f52',
                    '78dff113d28a492ab4bd66f9b1737942']
    
elif dim == 3:
    model_names = ['3f0b9ad0ad4c4f4e9b57ca174b73e525',
                   '4e696d5f195a4d129edb7532f368dd23',
                   '72e3f5c463ef474a90d836d7b45608a9']

# filter best models to only include the ones that are in model_names
best_models_sorted = [best_models_sorted[i] for i in range(len(best_models_sorted)) if best_models_sorted[i].split(path)[1].split('/')[0] in model_names]
best_models_sorted

# model/data params
min_max_len = 200 # min and max length of tracks model used during training
X_padtoken = 0 # pre-pad tracks to get them equal length
y_padtoken = 10 # pad y for same reason
batch_size = 32 # batch size for evaluation
use_temperature = True # use temperature scaling for softmax

# save paths
savename_score = '../deepspt_results/analytics/ANDIdeepspt_ensemble_score.pkl'
savename_pred = '../deepspt_results/analytics/ANDIdeepspt_ensemble_pred.pkl'
rerun_segmentaion = True # Set false to load previous results

print(len(X_to_eval))
# run temporal segmentation module of DeepSPT
ensemble_score, ensemble_pred = run_temporalsegmentation_ANDI(
                                best_models_sorted, 
                                X_to_eval, y_to_eval,
                                dir_name=dir_name, 
                                device=device, 
                                dim=dim, 
                                min_max_len=min_max_len, 
                                X_padtoken=X_padtoken, 
                                y_padtoken=y_padtoken,
                                batch_size=batch_size,
                                rerun_segmentaion=rerun_segmentaion,
                                savename_score=savename_score,
                                savename_pred=savename_pred,
                                use_temperature=use_temperature)


change_points = []
for i in range(len(ensemble_pred)):
    segl, cp, val = find_segments(ensemble_pred[i])
    change_points.append(cp[-2])
change_points = np.array(change_points)

plt.figure()
plt.scatter(change_points, Y[:,0])
plt.show()
plt.close()

import matplotlib.pyplot as plt
print(Y.shape)

pred_cp = change_points
true_cp = Y[:,0]

print(pred_cp.shape, true_cp.shape, pred_cp[0], true_cp[0])
plt.figure()
plt.scatter(true_cp, pred_cp)
plt.show()
plt.close()

plt.figure()
plt.hist(true_cp/np.array([len(t) for t in X]), alpha=0.5)
plt.show()
plt.close()

plt.figure()
plt.hist(pred_cp/np.array([len(t) for t in X]), alpha=0.5)
plt.show()
plt.close()

print('RMSE: ', np.sqrt(np.mean((pred_cp-true_cp)**2)))


# divider accounts for andi has row stacked xy for len is double of true len
temporal_true = make_andi_challenge_y_temporal(Y, X, divider=1)

acc = []
for i in range(len(ensemble_pred)):
    assert len(ensemble_pred[i]) == len(temporal_true[i])
    acc.append(np.mean(ensemble_pred[i]==temporal_true[i]))

plt.figure()
plt.hist(acc, bins=25)
plt.annotate('mean acc: %.3f' % np.mean(acc), xy=(0.03, 0.94), 
             xycoords='axes fraction', ha='left', va='center',
             fontsize=16)
plt.annotate('Flat acc: %.3f' % np.mean(np.hstack(temporal_true)==np.hstack(ensemble_pred)), xy=(0.03, 0.88), 
             xycoords='axes fraction', ha='left', va='center',
             fontsize=16)
plt.annotate('median acc: %.3f' % np.median(acc), xy=(0.03, 0.82), 
             xycoords='axes fraction', ha='left', va='center',
             fontsize=16)
plt.xlim(0,1)
plt.show()
plt.close()

p, t = np.hstack(ensemble_pred), np.hstack(temporal_true)

# confusion matrix

from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(t, p)
cm = cm / cm.sum(axis=1)[:, np.newaxis]
print(cm.shape)
plt.figure()
sns.heatmap(cm, annot=True, cmap='Blues')
plt.show()
plt.close()

import pickle

pickle.dump(temporal_true, open('../Andi_Challenge_Code/analytics/ANDIdeepspt_{}D_ensemble_temporal_true.pkl'.format(dim), 'wb'))
pickle.dump(ensemble_pred, open('../Andi_Challenge_Code/analytics/ANDIdeepspt_{}D_temporal_pred.pkl'.format(dim), 'wb'))

# %%
