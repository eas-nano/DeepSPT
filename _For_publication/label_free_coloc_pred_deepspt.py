# %%
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from glob import glob
import os
from deepspt_src import *
    

# Define the MLP architecture
class MLP(nn.Module):
    def __init__(self, layers):
        super(MLP, self).__init__()

        architecture = []
        for i in range(len(layers)-1):
            architecture.append(nn.Linear(layers[i], layers[i+1]))
            if i != len(layers)-2:
                architecture.append(nn.ReLU())
            else:
                architecture.append(nn.Softmax(dim=1))
        self.model = nn.Sequential(*architecture)

    def forward(self, x):
        x = self.model(x)
        return x
    

class EndosomeDataset(Dataset):
    def __init__(self, X, Y, device='cpu'):
        self.X = torch.from_numpy(X).to(device).float()
        self.Y = torch.from_numpy(Y).to(device).long()
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    

# Load the data
coloc_experiment = 'rotavirus' # rotavirus, dextran
EEA1val,NPC1val = 0,1
if coloc_experiment == 'rotavirus':
    loadpath = 'EEA1_NPC1_results/precomputed_files/coloc_results/coloc_rota/'

    tracks_coloc_all = []
    track_coloc_info_all = []
    timepoint_annotation_all = []
    tracks_coloc_EEA1_NPC1_labels = []
    tracks_coloc_info_EEA1_NPC1_labels = []
    timepoint_annotation_EEA1_NPC1_labels = []
   
    tracks_coloc_full_all = []
    tracks_coloc_full_EEA1_NPC1_labels = []
    
    experiment_name_coloc = []
    experiment_name_timepoint = []
    experiment_name_coloc_info = []
    experiment_name_coloc_full = []

    experiments = []
    for file in np.sort(glob(loadpath+'*', recursive=True)):
        if 'v1ROTAcoloc_EEA1' in file:
            tracks_coloc = pickle.load(open(file, 'rb'))
            for t in tracks_coloc:
                experiment_name_coloc.append(file.split('ROTA')[0]+'_EEA1')
                tracks_coloc_all.append(t)
                tracks_coloc_EEA1_NPC1_labels.append(EEA1val)
        if 'v1ROTAcoloc_NPC1' in file:
            tracks_coloc = pickle.load(open(file, 'rb'))
            for t in tracks_coloc:
                experiment_name_coloc.append(file.split('ROTA')[0]+'_NPC1')
                tracks_coloc_all.append(t)
                tracks_coloc_EEA1_NPC1_labels.append(NPC1val)
        if 'v1ROTAcoloc_info_EEA1' in file:
            track_coloc_info = pickle.load(open(file, 'rb'))
            for t in track_coloc_info:
                experiment_name_coloc_info.append(file.split('ROTA')[0]+'_EEA1')
                track_coloc_info_all.append(t)
                tracks_coloc_info_EEA1_NPC1_labels.append(EEA1val)
        if 'v1ROTAcoloc_info_NPC1' in file:
            track_coloc_info = pickle.load(open(file, 'rb'))
            for t in track_coloc_info:
                experiment_name_coloc_info.append(file.split('ROTA')[0]+'_NPC1')
                track_coloc_info_all.append(t)
                tracks_coloc_info_EEA1_NPC1_labels.append(NPC1val)
        if 'v1ROTAtimepoints_annotated_by_colocEEA1' in file:
            timepoint_annotation = pickle.load(open(file, 'rb'))
            for t in timepoint_annotation:
                experiment_name_timepoint.append(file.split('ROTA')[0]+'_EEA1')
                timepoint_annotation_all.append(t)
                timepoint_annotation_EEA1_NPC1_labels.append(EEA1val)
        if 'v1ROTAtimepoints_annotated_by_colocNPC1' in file:
            timepoint_annotation = pickle.load(open(file, 'rb'))
            for t in timepoint_annotation:
                experiment_name_timepoint.append(file.split('ROTA')[0]+'_NPC1')
                timepoint_annotation_all.append(t)
                timepoint_annotation_EEA1_NPC1_labels.append(NPC1val)  
        if 'v1ROTAcoloc_tracks_fullEEA1' in file:
            tracks_coloc_full = pickle.load(open(file, 'rb'))
            for t in tracks_coloc_full:
                experiment_name_coloc_full.append(file.split('ROTA')[0]+'_EEA1')
                tracks_coloc_full_all.append(t)
                tracks_coloc_full_EEA1_NPC1_labels.append(EEA1val)
        if 'v1ROTAcoloc_tracks_fullNPC1' in file:
            tracks_coloc_full = pickle.load(open(file, 'rb'))
            for t in tracks_coloc_full:
                experiment_name_coloc_full.append(file.split('ROTA')[0]+'_NPC1')
                tracks_coloc_full_all.append(t)
                tracks_coloc_full_EEA1_NPC1_labels.append(NPC1val)

    has_ones = []
    for l in timepoint_annotation_all:
        if 1. in np.array(l[0]):
            has_ones.append(1)

    print(len(tracks_coloc_all), len(track_coloc_info_all), len(has_ones), len(tracks_coloc_full_all))
    assert len(tracks_coloc_all) == len(track_coloc_info_all) == len(has_ones)

elif coloc_experiment == 'dextran':
    loadpath = 'EEA1_NPC1_results/precomputed_files/coloc_results/coloc_dextran/'
    tracks_coloc_all = []
    track_coloc_info_all = []
    timepoint_annotation_all = []
    tracks_coloc_EEA1_NPC1_labels = []
    tracks_coloc_info_EEA1_NPC1_labels = []
    timepoint_annotation_EEA1_NPC1_labels = []
    tracks_coloc_full_all = []
    tracks_coloc_full_EEA1_NPC1_labels = []

    experiment_name_coloc = []
    experiment_name_coloc_full = []
    experiment_name_timepoint = []
    experiment_name_coloc_info = []

    for file in np.sort(glob(loadpath+'*', recursive=True)):
        if 'DEXTRANcoloc_EEA1' in file:
            tracks_coloc = pickle.load(open(file, 'rb'))
            for t in tracks_coloc:
                experiment_name_coloc.append(file.split('DEXTRAN')[0]+'_EEA1')
                tracks_coloc_all.append(t)
                tracks_coloc_EEA1_NPC1_labels.append(EEA1val)
        if 'DEXTRANcoloc_NPC1' in file:
            tracks_coloc = pickle.load(open(file, 'rb'))
            for t in tracks_coloc:
                experiment_name_coloc.append(file.split('DEXTRAN')[0]+'_NPC1')
                tracks_coloc_all.append(t)
                tracks_coloc_EEA1_NPC1_labels.append(NPC1val)
        if 'DEXTRANcoloc_info_EEA1' in file:
            track_coloc_info = pickle.load(open(file, 'rb'))
            for t in track_coloc_info:
                experiment_name_coloc_info.append(file.split('DEXTRAN')[0]+'_EEA1')
                track_coloc_info_all.append(t)
                tracks_coloc_info_EEA1_NPC1_labels.append(EEA1val)
        if 'DEXTRANcoloc_info_NPC1' in file:
            track_coloc_info = pickle.load(open(file, 'rb'))
            for t in track_coloc_info:
                experiment_name_coloc_info.append(file.split('DEXTRAN')[0]+'_NPC1')
                track_coloc_info_all.append(t)
                tracks_coloc_info_EEA1_NPC1_labels.append(NPC1val)
        if 'DEXTRANtimepoints_annotated_by_colocEEA1' in file:
            timepoint_annotation = pickle.load(open(file, 'rb'))
            for t in timepoint_annotation:
                experiment_name_timepoint.append(file.split('DEXTRAN')[0]+'_EEA1')
                timepoint_annotation_all.append(t)
                timepoint_annotation_EEA1_NPC1_labels.append(EEA1val)
        if 'DEXTRANtimepoints_annotated_by_colocNPC1' in file:
            timepoint_annotation = pickle.load(open(file, 'rb'))
            for t in timepoint_annotation:
                experiment_name_timepoint.append(file.split('DEXTRAN')[0]+'_NPC1')
                timepoint_annotation_all.append(t)
                timepoint_annotation_EEA1_NPC1_labels.append(NPC1val)
        if 'DEXTRANcoloc_tracks_fullEEA1' in file:
            tracks_coloc_full = pickle.load(open(file, 'rb'))
            for t in tracks_coloc_full:
                experiment_name_coloc_full.append(file.split('DEXTRAN')[0]+'_EEA1')
                tracks_coloc_full_all.append(t)
                tracks_coloc_full_EEA1_NPC1_labels.append(EEA1val)
        if 'DEXTRANcoloc_tracks_fullNPC1' in file:
            tracks_coloc_full = pickle.load(open(file, 'rb'))
            for t in tracks_coloc_full:
                experiment_name_coloc_full.append(file.split('DEXTRAN')[0]+'_NPC1')
                tracks_coloc_full_all.append(t)
                tracks_coloc_full_EEA1_NPC1_labels.append(NPC1val)

    has_ones = []
    for l in timepoint_annotation_all:
        if 1. in np.array(l[0]):
            
            has_ones.append(1)
    
    print(len(tracks_coloc_all), len(track_coloc_info_all), len(has_ones), len(tracks_coloc_full_all))
    assert len(tracks_coloc_all) == len(track_coloc_info_all) == len(has_ones)

# %%
experiment_name_coloc = np.array(experiment_name_coloc)
experiment_name_timepoint = np.array(experiment_name_timepoint)
experiment_name_coloc_info = np.array(experiment_name_coloc_info)
experiment_name_coloc_full = np.array(experiment_name_coloc_full)

argsorter_expname_coloc = np.argsort(experiment_name_coloc)
argsorter_expname_timepoint = np.argsort(experiment_name_timepoint)
argsorter_expname_coloc_info = np.argsort(experiment_name_coloc_info)
argsorter_expname_coloc_full = np.argsort(experiment_name_coloc_full)

tracks_coloc_all_sorted = np.array(tracks_coloc_all)[argsorter_expname_coloc]
track_coloc_info_all_sorted = np.array(track_coloc_info_all)[argsorter_expname_coloc_info]
timepoint_annotation_all_sorted = np.array(timepoint_annotation_all)[argsorter_expname_timepoint]
tracks_coloc_full_all_sorted = np.array(tracks_coloc_full_all)[argsorter_expname_coloc_full]

tracks_coloc_EEA1_NPC1_labels_sorted = np.array(tracks_coloc_EEA1_NPC1_labels)[argsorter_expname_coloc]
tracks_coloc_info_EEA1_NPC1_labels_sorted = np.array(tracks_coloc_info_EEA1_NPC1_labels)[argsorter_expname_coloc_info]
timepoint_annotation_EEA1_NPC1_labels_sorted = np.array(timepoint_annotation_EEA1_NPC1_labels)[argsorter_expname_timepoint]
tracks_coloc_full_EEA1_NPC1_labels_sorted = np.array(tracks_coloc_full_EEA1_NPC1_labels)[argsorter_expname_coloc_full]

experiment_name_coloc_sorted = np.array(experiment_name_coloc)[argsorter_expname_coloc]
experiment_name_coloc_info_sorted = np.array(experiment_name_coloc_info)[argsorter_expname_coloc_info]
experiment_name_timepoint_sorted = np.array(experiment_name_timepoint)[argsorter_expname_timepoint]
experiment_name_coloc_full_sorted = np.array(experiment_name_coloc_full)[argsorter_expname_coloc_full]

print(len(timepoint_annotation_all_sorted))
assert np.mean(tracks_coloc_EEA1_NPC1_labels_sorted == tracks_coloc_info_EEA1_NPC1_labels_sorted) == 1
assert np.mean(experiment_name_coloc_sorted == experiment_name_coloc_info_sorted) == 1
assert np.mean(experiment_name_coloc_sorted == experiment_name_coloc_full_sorted) == 1
len(tracks_coloc_full_all_sorted)

len(tracks_coloc_full_all)

# %%

# check any duplicate duplicate has eea1 and npc1
tracks_coloc_EEA1_NPC1_labels_sorted_dupli = []
dupfilter = np.unique(track_coloc_info_all_sorted[:,0], return_counts=True)[1]>1
dup_coloc_idx = np.unique(track_coloc_info_all_sorted[:,0], return_counts=True)[0][dupfilter]
for dup_colidx in dup_coloc_idx:
    tracks_coloc_EEA1_NPC1_labels_sorted_dupli.append(tracks_coloc_EEA1_NPC1_labels_sorted[track_coloc_info_all_sorted[:,0]==dup_colidx])

dupli_count=0
for cpair in tracks_coloc_EEA1_NPC1_labels_sorted_dupli:
    if len(np.unique(cpair))!=1:
        dupli_count+=1

print(dupli_count)

if coloc_experiment == 'dextran' or coloc_experiment == 'rotavirus':
    X_tracks = []
    y = []
    has_dup = []
    for t_idx in track_coloc_info_all_sorted[:,0]:
        endo_label = np.unique(tracks_coloc_EEA1_NPC1_labels_sorted[track_coloc_info_all_sorted[:,0]==t_idx])

        if len(endo_label)==1:
            X_tracks.append(tracks_coloc_full_all_sorted[track_coloc_info_all_sorted[:,0]==t_idx][0,0])
            y.append(np.max(endo_label))
            has_dup.append(0)
        else:  
            print(1)
            # continue
            X_tracks.append(tracks_coloc_full_all_sorted[track_coloc_info_all_sorted[:,0]==t_idx][0,0])
            y.append(1)
            has_dup.append(1)  

X_tracks = np.array(X_tracks, dtype=object)
y = np.hstack(y)
endo_coloclabel = y.copy()
pickle.dump(y, open('deepspt_results/analytics/rota_coloc_y.pkl', 'wb'))
has_dup = np.hstack(has_dup)
len(y), len(endo_coloclabel), np.mean(endo_coloclabel==y),

# %%

X = [x-x[0] for x in X_tracks]

print(len(X), 'len X')
features = ['XYZ', 'SL', 'DP']
X_to_eval = add_features(X, features)
y_to_eval = [np.ones(len(x))*0.5 for x in X_to_eval]

# define dataset and method that model was trained on to find the model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
datasets = ['SimDiff_dim3_ntraces300000_Drandom0.0001-0.5_dt1.0e+00_N5-600_B0.05-0.25_R5-25_subA0-0.7_superA1.3-2_Q1-16']
methods = ['XYZ_SL_DP']
dim = 3 if 'dim3' in datasets[0] else 2
# find the model
dir_name = ''
modelpath = 'Unet_results/mlruns/'
modeldir = '36'
use_mlflow = False
if use_mlflow:
    import mlflow
    mlflow.set_tracking_uri('file:'+join(os.getcwd(), join("Unet_results", "mlruns")))
    best_models_sorted = find_models_for(datasets, methods)
else:
    # not sorted tho
    path = '/nfs/datasync4/jacobkh/SPT/mlruns/{}'.format(modeldir)
    best_models_sorted = find_models_for_from_path(path)
    print(best_models_sorted)

# model params
min_max_len = 601
X_padtoken = 0
y_padtoken = 10
batch_size = 32

savename_score = 'deepspt_results/analytics/coloc_'+coloc_experiment+'_ensemble_score.pkl'
savename_pred = 'deepspt_results/analytics/coloc_'+coloc_experiment+'_ensemble_pred.pkl'
rerun_segmentaion = True
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
                                savename_pred=savename_pred)

# %%
from tqdm import tqdm


tTDP_individuals = []
tTDP_labels = []
FP_list = []
ensemble_pred_list = []
ensemble_score_list = []

fp_datapath = '_Data/Simulated_diffusion_tracks/'
hmm_filename = 'simulated2D_HMM.json'
dim = 3
dt = 2.7
min_seglen_for_FP = 5
min_pred_length = 5
num_difftypes = 4
max_change_points = 10
add_FP = True
save_PN_name = 'EEA1_NPC1_results/precomputed_files/coloc_results/coloc_rota/Coloc_'+coloc_experiment

fp_names = np.array(['Alpha', 'D', 'extra', 'pval', 'Efficiency', 'logEfficiency', 'FractalDim', 
                     'Gaussianity', 'Kurtosis', 'MSDratio', 
                     'Trappedness', 't0', 't1', 't2', 't3', 'lifetime', 
                     'length', 'avgSL', 'avgMSD', 'AvgDP', 'corrDP',
                     'signDP', 'sumSL', 'minSL', 'maxSL',
                     'BroadnessSL', 'Speed', 'CoV', 'FractionSlow', 
                     'FractionFast', 'Volume', 'perc_ND', 'perc_DM', 
                     'perc_CD', 'perc_SD', 'num_changepoints', 'inst_msd_D',
                      'meanSequence', 'medianSequence', 'maxSequence', 
                      'minSequence', 'stdSequence', 'simSeq'])

# dont use features that include length
selected_features = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,
                              19,20,21,23,24,25,27,28,29,30,31,
                              32,33,34,35,36,37,38,39,40,41,42])

FP_all = create_temporalfingerprint(X_tracks, 
                                  ensemble_pred, fp_datapath, hmm_filename, dim, dt,
                                  selected_features=selected_features)

pickle.dump(FP_all, open(loadpath+'coloc_'+coloc_experiment+'_FP_all.pkl', 'wb'))
pickle.dump(y, open(loadpath+'coloc_'+coloc_experiment+'_y.pkl', 'wb'))

# %%

import joblib

modelsavepath = 'EEA1_NPC1_results/precomputed_files/eea1npc1_classifier/'
scaler = joblib.load(modelsavepath+'scaler.pkl')

X_scaled = FP_all
X_scaled = scaler.transform(X_scaled)  #[y==0]
y_test = y#[y==0]

conf_threshold = 0.6

print(np.unique(y, return_counts=True))
conditions_to_pred = [0,1] # 
diffs =  np.array(['EEA1', 'NPC1'])

# Define your model
print(X_scaled.shape[1])

layers = [X_scaled.shape[1], 2]
print(layers, 'layers')
model = MLP(layers)

print(X_scaled.shape, y_test.shape)
test_dataset = EndosomeDataset(X_scaled, y_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

y_true = []
y_pred = []
y_pred_pre_unsure = []

use_pytorch = True
if use_pytorch:
    model.load_state_dict(torch.load(modelsavepath+'final_best_model.pt'))
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            proba, predicted = torch.max(outputs.data, 1)
            y_pred_pre_unsure.append(predicted.cpu().numpy().tolist())
            predicted[proba<conf_threshold] = -1
            y_true.append(labels.cpu().numpy().tolist())
            y_pred.append(predicted.cpu().numpy().tolist())
else:
    model = joblib.load(modelsavepath+'final_best_model_v2.pkl')
    proba = model.predict_proba(X_scaled)
    predicted = np.argmax(proba, axis=1)
    y_pred_pre_unsure = predicted
    proba_max = np.max(proba, axis=1)
    predicted[proba_max<conf_threshold] = -1
    y_true = y_test
    y_pred = predicted

y_true = np.hstack(y_true).astype(int)
y_pred = np.hstack(y_pred).astype(int)
y_pred_pre_unsure = np.hstack(y_pred_pre_unsure).astype(int)

test_acc = accuracy_score(y_true[y_pred!=-1], y_pred[y_pred!=-1])
print(np.mean(y_pred[y_pred!=-1][y_true[y_pred!=-1]==0]==0))
print(np.mean(y_pred[y_pred!=-1][y_pred[y_pred!=-1]==0]==y_true[y_pred!=-1][y_pred[y_pred!=-1]==0]))
print(np.mean(y_pred[y_pred!=-1][y_true[y_pred!=-1]==1]==1))
print(np.mean(y_pred[y_pred!=-1][y_pred[y_pred!=-1]==1]==y_true[y_pred!=-1][y_pred[y_pred!=-1]==1]))

cm = confusion_matrix(y_true[y_pred!=-1], y_pred[y_pred!=-1], labels=[0,1], normalize='true')
pred_TP = cm[0,0]
pred_FP = cm[0,1]
pred_TN = cm[1,1]
pred_FN = cm[1,0]
y_pred_all = y_pred[y_pred!=-1]
y_test_all = y_test[y_pred!=-1]
N_test = len(y_test_all)

print(np.unique(y, return_counts=True)[1]/len(y))
print(np.unique(y_pred[y_pred!=-1], return_counts=True)[1]/len(y_pred[y_pred!=-1]))

print(f'Test Acc = {test_acc:.4f}')
print(f'Test Acc = {np.mean(test_acc):.4f}')

cms = np.array([[np.mean(pred_TP),np.mean(pred_FP)],
                [np.mean(pred_FN), np.mean(pred_TN)]])
cms_stds = np.array([[np.std(pred_TP, ddof=1),np.std(pred_FP, ddof=1)],
                     [np.std(pred_FN, ddof=1), np.std(pred_TN, ddof=1)]])

fontsize = 22
group_counts = "{}".format(N_test)

group_percentages = ["{:0.0f}".format(mu*100) for mu in cms.flatten()]

labels = np.asarray(group_percentages).reshape(2,2)

plt.figure(figsize=(6,5))
ax = sns.heatmap(cms, annot=labels, fmt='', cmap='Blues', annot_kws={"fontsize":fontsize},
                vmin=0, vmax=1)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)
plt.xlabel('Predicted', size=20)
plt.ylabel('True', size=20)

plt.xticks(np.linspace(0.5, 1.5, 2), diffs[conditions_to_pred], rotation=0, size=14)
plt.yticks(np.linspace(0.5, 1.5, 2), diffs[conditions_to_pred], rotation=45, size=14)
flat_acc = np.mean(np.hstack(y_pred_all) == np.hstack(y_test_all))
f1_ = f1_score(np.hstack(y_pred_all), np.hstack(y_test_all), average='macro')
plt.title(
    'N: {} predictions, {} tracks, \nAccuracy: {:.3f}, F1-score: {:.3f}'.format(
        len(np.hstack(y_pred_all)), N_test, flat_acc, f1_), size=16)

plt.tight_layout()
plt.savefig('deepspt_results/figures/coloc_to_EEA1NPC1_'+coloc_experiment+'conf_threshold'+str(conf_threshold)+'_confusion_matrix.pdf',
            pad_inches=0.2, bbox_inches='tight')
plt.show()
print(classification_report(np.hstack(y_test_all), np.hstack(y_pred_all), target_names=diffs[conditions_to_pred]))
print('Accuracy:', np.mean(np.array(np.hstack(y_pred_all))==np.array(np.hstack(y_test_all))))

if conf_threshold==0.6:
    norm1 = .72
    norm2 = .72

cms = np.array([[np.mean(pred_TP)/norm1,1-np.mean(pred_TP)/norm1],
                [1-np.mean(pred_TN)/norm2, np.mean(pred_TN)/norm2]])
cms_stds = np.array([[np.std(pred_TP, ddof=1),np.std(pred_FP, ddof=1)],
                     [np.std(pred_FN, ddof=1), np.std(pred_TN, ddof=1)]])

fontsize = 22
group_counts = "{}".format(N_test)

group_percentages = ["{:0.0f}%".format(mu*100) for mu in cms.flatten()]

labels = np.asarray(group_percentages).reshape(2,2)


plt.figure(figsize=(6,5))
ax = sns.heatmap(cms*100, annot=labels, fmt='', cmap='Blues', annot_kws={"fontsize":fontsize},
                vmin=0, vmax=100)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)
plt.xlabel('Predicted', size=20)
plt.ylabel('True', size=20)

plt.xticks(np.linspace(0.5, 1.5, 2), diffs[conditions_to_pred], rotation=0, size=14)
plt.yticks(np.linspace(0.5, 1.5, 2), diffs[conditions_to_pred], rotation=45, size=14)
flat_acc = np.mean(np.hstack(y_pred_all) == np.hstack(y_test_all))
f1_ = f1_score(np.hstack(y_pred_all), np.hstack(y_test_all), average='macro')
plt.title(
    'N: {} predictions, {} tracks, \nAccuracy: {:.3f}, F1-score: {:.3f}'.format(
        len(np.hstack(y_pred_all)), N_test, flat_acc, f1_), size=16)

plt.tight_layout()
plt.savefig('deepspt_results/figures/coloc_to_EEA1NPC1_'+coloc_experiment+'conf_threshold'+str(conf_threshold)+'_Normalizedconfusion_matrix.pdf',
            pad_inches=0.2, bbox_inches='tight')
plt.show()
print(classification_report(np.hstack(y_test_all), np.hstack(y_pred_all), target_names=diffs[conditions_to_pred]))
print('Accuracy:', np.mean(np.array(np.hstack(y_pred_all))==np.array(np.hstack(y_test_all))))

