# %%
import numpy as np
import pickle
import matplotlib.pyplot as plt
import datetime
from deepspt_src import *
from global_config import globals
import warnings
warnings.filterwarnings("ignore")

"""
Simulate diffusion of two classes
Run temporal segmentation module of DeepSPT on simulated data
Create fingerprints of simulated data
Predict the two classes based on their diffusional features
"""

# get consistent result
seed = globals._parse({})

# device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""Generate a simulated data """
# variables
n_per_clean_diff = 1 # gets multiplied by n_classes=4
n_classes = 4 # number og diffusion types
n_changing_traces = 50 # number of tracks
random_D = True

# Two populations of tracks with Ds due to stochasticity tracks may return lower D if computed 
Drandomranges_pairs = [[2*10**-3,   6*10**-3],   [5*10**-3,   9*10**-3]]
Nrange = [150,200] # length of tracks
Brange = [0.05,0.25] # boundary geometry
Rranges = [[5,12],[8,15]] # relative active diffusion
subalpharanges = [[0.3,0.6], [0.4, 0.7]] # subdiffusion exponent
superalpharange = [1.3, 2] # superdiffusion exponent (not used for dir_motion='active')
Qrange = [6,16] # steps from diffusion to localization error ratio
Dfixed = 0.1 # fixed diffusion coefficient (not used for random_D=True)
dir_motion = 'active'

dim = 3 # 2D or 3D
dt = 1 # frame rate in seconds
max_changepoints = 4 # number of times changing diffusion traces can change
min_parent_len = 5 # minimum length of subtrace
total_parents_len = Nrange[1] # max length of changing diffusion (heterogeneous) tracks
path = '_Data/Simulated_diffusion_tracks/' # path to save and load
output_name = 'tester_set'+str(dim) # name of output file - change to get new tracks if already run
print(path+output_name)

# Generate data
if not os.path.exists(path+output_name+'.pkl'): # dont generate if already exists
    changing_diffusion_list_all = []
    changing_label_list_all = []
    print(n_per_clean_diff, n_changing_traces)
    for i in range(2):
        print("Generating data")
        subalpharange = subalpharanges[i]
        Rrange = Rranges[i]
        Drandomrange = Drandomranges_pairs[i]
        params_matrix = Get_params(n_per_clean_diff, dt, random_D, False,
                                Nrange = Nrange, Brange = Brange, 
                                Rrange = Rrange, 
                                subalpharange = subalpharange,
                                superalpharange = superalpharange, 
                                Qrange = Qrange, 
                                Drandomrange = Drandomrange,
                                Dfixed = Dfixed)
        NsND, NsAD, NsCD, NsDM, NstD = [params_matrix[i] for i in range(5)]
        Ds, r_cs, ellipse_dims, angles, vs, wiggle, r_stuck, subalphas, superalphas, sigmaND, sigmaAD, sigmaCD, sigmaDM, sigmaStD = params_matrix[7:]

        # Changing diffusion types
        s = datetime.datetime.now()
        changing_diffusion_list, changing_label_list = Gen_changing_diff(n_changing_traces, 
                                                                        max_changepoints, 
                                                                        min_parent_len, 
                                                                        total_parents_len, 
                                                                        dt, random_D=random_D, 
                                                                        n_classes=n_classes, dim=dim,
                                                                        Nrange = Nrange, Brange = Brange, 
                                                                        Rrange = Rrange, 
                                                                        subalpharange = subalpharange,
                                                                        superalpharange = superalpharange, 
                                                                        Qrange = Qrange, 
                                                                        Drandomrange = Drandomrange,
                                                                        Dfixed = Dfixed,
                                                                        DMtype=dir_motion)
        for cdl,cll in zip(changing_diffusion_list, changing_label_list):
            changing_diffusion_list_all.append(cdl)
            changing_label_list_all.append(cll)
    pickle.dump(changing_diffusion_list_all, open(path+output_name+'.pkl', 'wb'))
    pickle.dump(changing_label_list_all, open(path+output_name+'_labels.pkl', 'wb'))

else:
    changing_diffusion_list_all = pickle.load(open(path+output_name+'.pkl', 'rb'))
    changing_label_list_all = pickle.load(open(path+output_name+'_labels.pkl', 'rb'))

# prep data
tracks = changing_diffusion_list_all
X = [x-x[0] for x in tracks]
print(len(X), 'len X')
features = ['XYZ', 'SL', 'DP']
X_to_eval = add_features(X, features)
y_to_eval = [np.ones(len(x))*0.5 for x in X_to_eval]

# define dataset and method that model was trained on to find the model
if dim == 3:
    datasets = ['SimDiff_dim3_ntraces300000_Drandom0.0001-0.5_dt1.0e+00_N5-600_B0.05-0.25_R5-25_subA0-0.7_superA1.3-2_Q1-16']
    modeldir = '36'
if dim == 2:
    datasets = ['SimDiff_dim2_ntraces300000_Drandom0.0001-0.5_dt1.0e+00_N5-600_B0.05-0.25_R5-25_subA0-0.7_superA1.3-2_Q1-16']
    modeldir = '3'
methods = ['XYZ_SL_DP']

# find the model
dir_name = ''
modelpath = 'mlruns/'

use_mlflow = False # troublesome if not on same machine as trained (mlflow) thus False
if use_mlflow:
    import mlflow
    mlflow.set_tracking_uri('file:'+join(os.getcwd(), "Unet_results"))
    best_models_sorted = find_models_for(datasets, methods)
else:
    path = 'mlruns/{}'.format(modeldir)
    assert os.path.exists(path), 'Need to download models - see Deep learning assisted analysis of single particle tracking for automated correlation between diffusion and function Kæstel-Hansen et al.'
    best_models_sorted = find_models_for_from_path(path)
    print(best_models_sorted) # ordered as found

# model/data params
min_max_len = 601 # min and max length of tracks model used during training
X_padtoken = 0 # pre-pad tracks to get them equal length
y_padtoken = 10 # pad y for same reason
batch_size = 32 # batch size for evaluation
use_temperature = True # use temperature scaling for softmax

# save paths
savename_score = 'deepspt_results/analytics/testdeepspt_ensemble_score.pkl'
savename_pred = 'deepspt_results/analytics/testdeepspt_ensemble_pred.pkl'
rerun_segmentaion = True # Set false to load previous results

print(len(X_to_eval))
# run temporal segmentation module of DeepSPT
ensemble_score, ensemble_pred = run_temporalsegmentation(
                                best_models_sorted, 
                                X_to_eval, y_to_eval,
                                use_mlflow=use_mlflow,  
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

# pretrained HMM for fingerprints
fp_datapath = '_Data/Simulated_diffusion_tracks/'
hmm_filename = 'simulated2D_HMM.json'

# dont use features that include length of track (one can if it makes sense for data) else dont use in below functions
selected_features = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,
                            19,20,21,23,24,25,27,28,29,30,31,
                            32,33,34,35,36,37,38,39,40,41,42])

# featurize using diffusional fingerprinting module of DeepSPT
FP_1 = create_temporalfingerprint(changing_diffusion_list_all[:n_changing_traces], 
                                    ensemble_pred[:n_changing_traces], fp_datapath, hmm_filename, dim, dt,
                                    selected_features=selected_features)

FP_2 = create_temporalfingerprint(changing_diffusion_list_all[n_changing_traces:], 
                                    ensemble_pred[n_changing_traces:], fp_datapath, hmm_filename, dim, dt,
                                    selected_features=selected_features)
FP_2.shape
# %%
i = 0

timepoint_confidence_plot(ensemble_score[i])
savename = ''

savename = ''
compare_pred2sim_diffusion(changing_diffusion_list_all[i][:,0], 
                           changing_diffusion_list_all[i][:,2], 
                           changing_label_list_all[i], 
                           ensemble_pred[i], 
                           savename=savename)

if dim==3:
    plot_3Ddiffusion([changing_diffusion_list_all[i]], [changing_label_list_all[i]])
elif dim==2:
    plot_diffusion(changing_diffusion_list_all[i], changing_diffusion_list_all[i])

# %%

# confusion matrix of ensemble_pred vs changing_label_list_all
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns

cm = confusion_matrix(np.hstack(changing_label_list_all), np.hstack(ensemble_pred), normalize='true')
plt.figure(figsize=(6,5))
labels = ['ND', 'DM', 'CD', 'SD']
sns.heatmap(cm*100,
            annot=True, 
            annot_kws={"size": 30}, 
            fmt='.0f', cmap='Blues', 
            xticklabels=labels, 
            yticklabels=labels) # font size
# annotate %
plt.title('Confusion matrix (%)', fontsize=24)
plt.ylabel('True label', fontsize=18)
plt.xlabel('Predicted label', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()

print('Accuracy: ', np.round(accuracy_score(np.hstack(changing_label_list_all), np.hstack(ensemble_pred)),3))

acc = []
for i in range(len(changing_label_list_all)):
    acc.append(accuracy_score(changing_label_list_all[i], ensemble_pred[i]))

print('Accuracy: ', np.round(np.mean(acc),3), '+/-', np.round(np.std(acc),3))
print('median Accuracy: ', np.round(np.median(acc),3), '+/-', np.round(np.std(acc),3))

plt.figure(figsize=(5,4))
plt.hist(acc, bins=50)


# %%

fp_names = np.array(['Alpha', 'D', 'extra', 'pval', 'Efficiency', 'logEfficiency', 'FractalDim', 
                    'Gaussianity', 'Kurtosis', 'MSDratio', 
                    'Trappedness', 't0', 't1', 't2', 't3', 'lifetime', 
                    'avgSL', 'avgMSD', 'AvgDP', 'corrDP',
                    'signDP', 'minSL', 'maxSL',
                    'BroadnessSL', 'CoV', 'FractionSlow', 
                    'FractionFast', 'Volume', 'perc_ND', 'perc_DM', 
                    'perc_CD', 'perc_SD', 'num_changepoints', 'inst_msd_D',
                    'meanSequence', 'medianSequence', 'maxSequence', 
                    'minSequence', 'stdSequence', 'simSeq'])

featnum = 33 # feature to plot
bins = 10
fig, ax = plt.subplots(1,1, figsize=(5,5))
plt.title(fp_names[featnum])
ax.hist(FP_1[:,featnum], bins=bins, 
        range=(np.min([FP_1[:,featnum], FP_2[:,featnum]]),np.max([FP_1[:,featnum], FP_2[:,featnum]])),
        alpha=0.5)
ax.hist(FP_2[:,featnum], bins=bins, 
        range=(np.min([FP_1[:,featnum], FP_2[:,featnum]]),np.max([FP_1[:,featnum], FP_2[:,featnum]])),
        alpha=0.5)


# %%
# acc of linear classifier on FP1 and FP2
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

scaler = StandardScaler()

FP_pred = np.vstack([FP_1, FP_2])

y_before_after = np.hstack([np.zeros(FP_1.shape[0]), np.ones(FP_2.shape[0])])

kf = StratifiedShuffleSplit(n_splits=5, random_state=0, test_size=0.1)
kf.get_n_splits(FP_pred)

accuracy = []
precision = []
recall = []
f1 = []
TP1 = []
TP2 = []
TP3 = []
FP1 = []
FP2 = []
FP3 = []
print(FP_pred.shape, y_before_after.shape)
for train_index, test_index in kf.split(FP_pred, y_before_after):
    X_train, X_test = FP_pred[train_index], FP_pred[test_index]
    y_train, y_test = y_before_after[train_index], y_before_after[test_index]

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train[np.isnan(X_train)] = 0
    X_test[np.isnan(X_test)] = 0

    # random oversampling
    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler(random_state=0)
    X_train, y_train = ros.fit_resample(X_train, y_train)

    clf = LogisticRegression(random_state=0, solver='lbfgs', max_iter=10000,
                            multi_class='multinomial').fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy.append(accuracy_score(y_test, y_pred))
    precision.append(precision_score(y_test, y_pred, average='micro'))
    recall.append(recall_score(y_test, y_pred, average='macro'))
    f1.append(f1_score(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred, normalize='true')
    TP1.append(cm[0,0])
    FP1.append(cm[0,1])
    TP2.append(cm[1,1])
    FP2.append(cm[1,0])
    #TP3.append(cm[2,2])
    #FP3.append(cm[2,0])

print('TP1: ', np.round(np.mean(TP1),3), '+/-', np.round(np.std(TP1),3))
print('TP2: ', np.round(np.mean(TP2),3), '+/-', np.round(np.std(TP2),3))
print('TP3: ', np.round(np.mean(TP3),3), '+/-', np.round(np.std(TP3),3))
print('FP1: ', np.round(np.mean(FP1),3), '+/-', np.round(np.std(FP1),3))
print('FP2: ', np.round(np.mean(FP2),3), '+/-', np.round(np.std(FP2),3))
print('FP3: ', np.round(np.mean(FP3),3), '+/-', np.round(np.std(FP3),3))
print('Accuracy: ', np.round(np.mean(accuracy),3), '+/-', np.round(np.std(accuracy),3))
print('Precision: ', np.round(np.mean(precision),3), '+/-', np.round(np.std(precision),3))
print('Recall: ', np.round(np.mean(recall),3), '+/-', np.round(np.std(recall),3))
print('F1: ', np.round(np.mean(f1),3), '+/-', np.round(np.std(f1),3))

#plot confusion matrix of tp1 tp2 fp1 fp2
plt.figure(figsize=(6,5))
sns.heatmap(
    np.array([[np.mean(TP1), np.mean(FP1)],
              [np.mean(FP2), np.mean(TP2)]])*100, 
              annot=True, 
              annot_kws={"size": 30}, 
              fmt='.0f', cmap='Blues') # font size
# annotate %
plt.title('Confusion matrix (%)', fontsize=24)
plt.ylabel('True label', fontsize=18)
plt.xlabel('Predicted label', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()


# %%

