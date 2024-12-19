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

# saving
features = globals.features 
method = "_".join(features,)

# choose dim to train for
dim = 3

modelname = 'DeepSPT_ANDI_heterotracks'
datapath = '../Andi_Challenge_Code/ANDI_challenge_testsets/'
print(os.listdir(datapath))

note = 'test1'
N_save = 20000 # number of tracks 
max_T = 200 # maximum track length
min_number_of_segments = 3
max_number_of_segments = 6 # maximum number of segments -1 => change points
diff_to_loc_ratio = 0.5 # ratio of diffusion to localization error
filename_X = 'ANDI_{}_hetero_2D3Dtracks_N{}_maxlen{}_D2noise{}_maxsegm{}.pkl'.format(note, N_save, max_T, diff_to_loc_ratio, max_number_of_segments)

path_datasets = datapath+filename_X
N_save_train = N_save

save_dataset = False

print('Data is dim: ', dim)
print('Data is from: ', datapath)
X, y = pickle.load(open(datapath+filename_X, 'rb'))
X = [x-x[0] for x in X]
X = X if dim == 3 else [x[:,:2] for x in X]

print('shapes', len(X), len(y), X[0].shape, y[0].shape)
features = ['XYZ', 'SL', 'DP']
methods = ['XYZ_SL_DP']
X_to_eval = add_features(X, features)
y_to_eval = [np.ones(len(x))*0.5 for x in X_to_eval]

print('shapes', len(X_to_eval), X_to_eval[0].shape,)
# define dataset and method that model was trained on to find the model
if dim == 2:
    modeldir = '55'
if dim == 3:
    modeldir = '56'

# find the model
dir_name = ''
modelpath = 'mlruns'

use_mlflow = False # troublesome if not on same machine as trained (mlflow) thus False

def find_models_from_path_and_name(path, modelname):
    # load from path
    files = sorted(glob(path+'*/*{}'.format(modelname), recursive=True))
    return files

print(os.listdir(modelpath))

modelname = '.torch'
path = modelpath+'/{}/'.format(modeldir)
print(path)
print(sorted(glob(path, recursive=True)))
best_models_sorted = find_models_from_path_and_name(path, modelname)[:3]
print(best_models_sorted) # ordered as found

# filter best models to only include the ones that are in model_names
best_models_sorted

# %%
# model/data params
min_max_len = 599 # min and max length of tracks model used during training
X_padtoken = 0 # pre-pad tracks to get them equal length
y_padtoken = 10 # pad y for same reason
batch_size = 32 # batch size for evaluation
use_temperature = True # use temperature scaling for softmax
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# save paths
savename_score = '../deepspt_results/analytics/ANDIdeepsptHetero_ensemble_score.pkl'
savename_pred = '../deepspt_results/analytics/ANDIdeepsptHetero_ensemble_pred.pkl'
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

# divider accounts for andi has row stacked xy for len is double of true len
temporal_true = y

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
plt.title('Accuracy of DeepSPT on ANDI hetero tracks dim {}'.format(dim))
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
plt.title('CM of DeepSPT on ANDI hetero tracks dim {}'.format(dim))
plt.show()
plt.close()

import pickle

pickle.dump(temporal_true, open('../Andi_Challenge_Code/analytics/ANDIdeepspt_hetero_{}D_ensemble_temporal_true_hetero.pkl'.format(dim), 'wb'))
pickle.dump(ensemble_pred, open('../Andi_Challenge_Code/analytics/ANDIdeepspt_hetero_{}D_temporal_pred_hetero.pkl'.format(dim), 'wb'))


# %%
