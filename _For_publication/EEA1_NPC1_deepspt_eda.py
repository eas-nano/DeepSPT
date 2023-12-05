# %%
import matplotlib.pyplot as plt
import pickle 
import numpy as np
import torch
import sys
import os 
from sklearn.neural_network import MLPClassifier
import sys
sys.path.append('../')
from deepspt_src import *
import random
from global_config import globals
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import *
from scipy.stats import mode

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from glob import glob
from joblib import Parallel, delayed
import time


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


def random_oversampler(X_train_ee, X_train_le, y_train_ee, y_train_le):
    np.random.seed(seed)
    max_num_ = np.max([len(X_train_ee), len(X_train_le)])
    if np.max([0,max_num_-X_train_ee.shape[0]])>0:
        oversampling_ee_idx = np.random.choice(list(range(len(X_train_ee))), size=(max_num_-X_train_ee.shape[0]))
        oversampling_ee = X_train_ee[oversampling_ee_idx]
        oversampling_y_ee = y_train_ee[oversampling_ee_idx]
        X_train_ee = np.vstack([X_train_ee, oversampling_ee])
        y_train_ee = np.hstack([y_train_ee, oversampling_y_ee])

    if np.max([0,max_num_-X_train_le.shape[0]])>0:
        oversampling_le_idx = np.random.choice(list(range(len(X_train_le))), size=(max_num_-X_train_le.shape[0]))
        oversampling_le = X_train_le[oversampling_le_idx]
        oversampling_y_le = y_train_le[oversampling_le_idx]
        X_train_le = np.vstack([X_train_le, oversampling_le])
        y_train_le = np.hstack([y_train_le, oversampling_y_le])
    return X_train_ee, X_train_le,  y_train_ee, y_train_le


def label_translator(i):
    if i == 0:
        return 'EEA1'
    elif i == 1:
        return 'NPC1'

# global config variables
globals._parse({})

# set seed
seed = globals.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# ****************Initiate variables**********************
# define dataset and method that model was trained on to find the model
datasets = ['SimDiff_dim3_ntraces300000_Drandom0.0001-0.5_dt1.0e+00_N5-600_B0.05-0.25_R5-25_subA0-0.7_superA1.3-2_Q1-16']
methods = ['XYZ_SL_DP']
the_data_is = '3D' if 'dim3' in datasets[0] else '2D'
deepspt_figpath = '../deepspt_results/'
device = globals.device

# find the model
use_mlflow = False # appears to require the device/computer mlflow ran on
modeldir = '36'

# find the model
if use_mlflow:
    mlflow.set_tracking_uri('file:'+join(os.getcwd(), join("", "mlruns")))
    best_models_sorted = find_models_for(datasets, methods)
else:
    # not sorted tho
    path = '../mlruns/{}/'.format(modeldir)
    best_models_sorted = find_models_for_from_path(path)
    print(os.listdir(path))
    print(best_models_sorted)

modelpath = '../mlruns/'

# Prep data
# Define where to find experiments
main_dir = '../_Data/LLSM_data/EEA1_NPC1/'
sub_dir1 = '20230406_p55_p5_sCMOS_Jacob_Anand_Ricky/' # eea1 npc1
sub_dir1_CS2 = 'CS2_SVGA_EEA1mscarlett_NPC1_JFX646/'
sub_dir1_CS3 = 'CS3_SVGA_EEA1mscarlett_NPC1_JFX646/'
sub_dir3 = '20230411_p5_p55_sCMOS_Jacob_Anand_Ricky/' # eea1 npc1 and eea1 npc1 + dex
sub_dir3_CS1 = 'CS1_SVGA_EEA1_mscarlett_NPC1_Halo_JF646X/'
sub_dir3_CS2 = 'CS2_SVGA_EEA1_mscarlett_NPC1_Halo_JFX646_PhrodogreenDex/' # +dex

EEA1_paths = [main_dir+sub_dir1+sub_dir1_CS2, main_dir+sub_dir1+sub_dir1_CS3,
              main_dir+sub_dir3+sub_dir3_CS1, main_dir+sub_dir3+sub_dir3_CS2]
NPC1_paths = [main_dir+sub_dir1+sub_dir1_CS2, main_dir+sub_dir1+sub_dir1_CS3,
              main_dir+sub_dir3+sub_dir3_CS1, main_dir+sub_dir3+sub_dir3_CS2]
EEA1_paths
# %%

xy_to_um, z_to_um = 0.1, 0.25
min_trace_length = 20
min_brightness = 0
max_brightness = np.inf
do_fuse_ = False

# temperature and tracking problems
dont_use_experiments = ['Ex11_488_100mW_560_100mW_642_200mW_z0p5', 
                        'Ex09_488_100mW_560_100mW_642_200mW_z0p5']

no_tracks = []
num_t_eea1_exp = []

# EEA1 data
no_tracks = []
num_t_eea1_exp = []

def use_file_eea1(file):
    
    if 'ch560nmCamA' not in file:
        return False
    if 'EX_xyProj' in file or 'EX_yzProj' in file or 'EX_xzProj' in file:
        return False
    if '/Nalysis' in file or '/nalysis' in file:
        return False
    
    return True


def parallel_loader_eea1(file):
    
    EEA1_df, seqOfEvents_list, keep_idx_len,\
        keep_idx_dim, keep_idx_bri,\
        original_idx = load_3D_data(
                        file.split('ProcessedTracks.mat')[0], 
                        '{}ProcessedTracks.mat', 
                        OUTPUT_NAME,
                        min_trace_length,
                        min_brightness,
                        max_brightness,
                        dont_use=dont_use_experiments)
    seqOfEvents_list_curated = []
    for i, idx in enumerate(original_idx):
        if idx not in keep_idx_len:
            continue
        if idx not in keep_idx_dim:
            continue
        if idx not in keep_idx_bri:
            continue
        seqOfEvents_list_curated.append(seqOfEvents_list[i])

    if len(EEA1_df)>=1:
        
        EEA1_tracks, EEA1_timepoints, EEA1_frames,\
        EEA1_track_ids, EEA1_amplitudes,\
        EEA1_amplitudes_S, EEA1_amplitudes_SM,\
        EEA1_amplitudes_sig, EEA1_amplitudes_bg,\
        EEA1_catIdx = curate_3D_data_to_tracks(EEA1_df, xy_to_um, z_to_um)
        EEA1_compound_idx = np.array(list(range(len(EEA1_tracks))))[EEA1_catIdx>4]
        num_t_eea1_exp.append(len(EEA1_tracks))
        handle_compound_tracks(EEA1_compound_idx, seqOfEvents_list_curated, EEA1_tracks, EEA1_timepoints, EEA1_frames,
                            EEA1_track_ids, EEA1_amplitudes, EEA1_amplitudes_S,
                            EEA1_amplitudes_SM, EEA1_amplitudes_sig, EEA1_amplitudes_bg,
                            min_trace_length, min_brightness)

        if do_fuse_:
            EEA1_tracks, EEA1_timepoints, EEA1_frames,\
            EEA1_track_ids, EEA1_amplitudes,\
            EEA1_amplitudes_S, EEA1_amplitudes_SM,\
            EEA1_amplitudes_sig, EEA1_amplitudes_bg, = fuse_tracks(
                                            EEA1_compound_idx, EEA1_tracks, 
                                            EEA1_timepoints, EEA1_frames,
                                            EEA1_track_ids, EEA1_amplitudes, 
                                            EEA1_amplitudes_S, EEA1_amplitudes_SM, 
                                            EEA1_amplitudes_sig, EEA1_amplitudes_bg,
                                            min_trace_length, min_brightness,
                                            blinking_forgiveness=1)
            
        files_to_save = [file.split('ProcessedTracks.mat')[0] for i in range(len(EEA1_tracks))] 
        exp_to_save = [file.split('ch560nmCamA')[0] for i in range(len(EEA1_tracks))]
        return files_to_save, exp_to_save,\
            EEA1_tracks, EEA1_timepoints, EEA1_frames,\
            EEA1_track_ids, EEA1_amplitudes,\
            EEA1_amplitudes_S, EEA1_amplitudes_SM,\
            EEA1_amplitudes_sig, EEA1_amplitudes_bg
    else:
        return [file.split('ch560nmCamA')[0]], [file.split('ch560nmCamA')[0]], [np.ones((1,3))*-1],\
                [np.ones((1,3))*-1], [np.ones((1,3))*-1], [np.ones((1,3))*-1],\
                    [np.ones((1,3))*-1], [np.ones((1,3))*-1], [np.ones((1,3))*-1]

files_curated_eea1 = []
for pn_i in range(len(EEA1_paths)):
    SEARCH_PATTERN = "{}/**/ch560nmCamA/**/ProcessedTracks.mat"
    OUTPUT_NAME = "eea1"
    _input = SEARCH_PATTERN.format(EEA1_paths[pn_i])
    files = sorted(glob(_input, recursive=True))
    for f in files:
        if use_file_eea1(f):
            files_curated_eea1.append(f)

t = time.time()
t1 = datetime.datetime.now()
print(t1)
results = Parallel(n_jobs=30)(
                delayed(parallel_loader_eea1)(file) for file in files_curated_eea1
                if use_file_eea1(file))
print(datetime.datetime.now(), time.time()-t)
print('load')

EEA1_filenames_all = np.array(flatten_list([r[0] for r in results]))
EEA1_expname_all = np.array(flatten_list([r[1] for r in results]))
EEA1_tracks_all = np.array(flatten_list([r[2] for r in results]), dtype=object)
EEA1_timepoints_all = np.array(flatten_list([r[3] for r in results]), dtype=object)
EEA1_frames_all = np.array(flatten_list([r[4] for r in results]), dtype=object)
EEA1_track_ids_all = np.array(flatten_list([r[5] for r in results]), dtype=object)
EEA1_amplitudes_all = np.array(flatten_list([r[6] for r in results]), dtype=object)
EEA1_amplitudes_sig_all = np.array(flatten_list([r[7] for r in results]), dtype=object)
EEA1_amplitudes_bg_all = np.array(flatten_list([r[8] for r in results]), dtype=object)

len(EEA1_filenames_all), len(EEA1_tracks_all)

# %%
# NPC1 data


def use_file_npc1(file): 
    if 'ch642nmCamB' not in file:
        return False
    if 'EX_xyProj' in file or 'EX_yzProj' in file or 'EX_xzProj' in file:
        return False
    if '/Nalysis' in file or '/nalysis' in file:
        return False
    return True

num_t_npc1_exp = []

def parallel_loader_npc1(file):

    NPC1_df, seqOfEvents_list, keep_idx_len,\
        keep_idx_dim, keep_idx_bri,\
        original_idx = load_3D_data(
                        file.split('ProcessedTracks.mat')[0], 
                        '{}ProcessedTracks.mat', 
                        OUTPUT_NAME,
                        min_trace_length,
                        min_brightness,
                        max_brightness,
                        dont_use=dont_use_experiments)
    seqOfEvents_list_curated = []
    for i, idx in enumerate(original_idx):
        if idx not in keep_idx_len:
            continue
        if idx not in keep_idx_dim:
            continue
        if idx not in keep_idx_bri:
            continue
        seqOfEvents_list_curated.append(seqOfEvents_list[i])

    NPC1_tracks, NPC1_timepoints, NPC1_frames,\
    NPC1_track_ids, NPC1_amplitudes,\
    NPC1_amplitudes_S, NPC1_amplitudes_SM,\
    NPC1_amplitudes_sig, NPC1_amplitudes_bg,\
    NPC1_catIdx = curate_3D_data_to_tracks(NPC1_df, xy_to_um, z_to_um)
    NPC1_compound_idx = np.array(list(range(len(NPC1_tracks))))[NPC1_catIdx>4]
    num_t_npc1_exp.append(len(NPC1_tracks))
    handle_compound_tracks(NPC1_compound_idx, seqOfEvents_list_curated, NPC1_tracks, NPC1_timepoints, NPC1_frames,
                            NPC1_track_ids, NPC1_amplitudes, NPC1_amplitudes_S,
                                NPC1_amplitudes_SM, NPC1_amplitudes_sig, NPC1_amplitudes_bg,
                                min_trace_length, min_brightness)

    
    if do_fuse_:
        NPC1_tracks, NPC1_timepoints, NPC1_frames,\
        NPC1_track_ids, NPC1_amplitudes,\
        NPC1_amplitudes_S, NPC1_amplitudes_SM,\
        NPC1_amplitudes_sig, NPC1_amplitudes_bg, = fuse_tracks(
                                        NPC1_compound_idx, NPC1_tracks, 
                                        NPC1_timepoints, NPC1_frames,
                                        NPC1_track_ids, NPC1_amplitudes, 
                                        NPC1_amplitudes_S, NPC1_amplitudes_SM, 
                                        NPC1_amplitudes_sig, NPC1_amplitudes_bg,
                                        min_trace_length, min_brightness,
                                        blinking_forgiveness=1)
    
    files_to_save = [file.split('ProcessedTracks.mat')[0] for i in range(len(NPC1_tracks))] 
    exp_to_save = [file.split('ch642nmCamB')[0] for i in range(len(NPC1_tracks))]
    return files_to_save, exp_to_save,\
            NPC1_tracks, NPC1_timepoints, NPC1_frames,\
            NPC1_track_ids, NPC1_amplitudes,\
            NPC1_amplitudes_S, NPC1_amplitudes_SM,\
            NPC1_amplitudes_sig, NPC1_amplitudes_bg

files_curated_npc1 = []
for pn_i in range(len(NPC1_paths)):
    SEARCH_PATTERN = "{}/**/ch642nmCamB/**/ProcessedTracks.mat"
    OUTPUT_NAME = "npc1"
    _input = SEARCH_PATTERN.format(EEA1_paths[pn_i])
    files = sorted(glob(_input, recursive=True))
    for f in files:
        if use_file_npc1(f):
            files_curated_npc1.append(f)

t = time.time()
t1 = datetime.datetime.now()
print(t1)
results = Parallel(n_jobs=30)(
                delayed(parallel_loader_npc1)(file) for file in files_curated_npc1
                if use_file_npc1(file))
print(datetime.datetime.now(), time.time()-t)
print('load')

NPC1_filenames_all = np.array(flatten_list([r[0] for r in results]))
NPC1_expname_all = np.array(flatten_list([r[1] for r in results]))
NPC1_tracks_all = np.array(flatten_list([r[2] for r in results]), dtype=object)
NPC1_timepoints_all = np.array(flatten_list([r[3] for r in results]), dtype=object)
NPC1_frames_all = np.array(flatten_list([r[4] for r in results]), dtype=object)
NPC1_track_ids_all = np.array(flatten_list([r[5] for r in results]), dtype=object)
NPC1_amplitudes_all = np.array(flatten_list([r[6] for r in results]), dtype=object)
NPC1_amplitudes_sig_all = np.array(flatten_list([r[7] for r in results]), dtype=object)
NPC1_amplitudes_bg_all = np.array(flatten_list([r[8] for r in results]), dtype=object)

len(NPC1_filenames_all), len(NPC1_tracks_all)

# %%
EEA1val = 0
NPC1val = 1
tracks = list(EEA1_tracks_all) + list(NPC1_tracks_all)
tracks = np.array(tracks, dtype=object)
track_labels = np.concatenate([np.ones(len(EEA1_tracks_all))*EEA1val,
                               np.ones(len(NPC1_tracks_all))*NPC1val])


print(len(EEA1_tracks_all), len(NPC1_tracks_all))
print(len(EEA1_filenames_all), len(NPC1_filenames_all))
track_labels.shape


# %%
X = [x-x[0] for x in tracks]
print(len(X), 'len X')
features = ['XYZ', 'SL', 'DP']
X_to_eval = add_features(X, features)
y_to_eval = [np.ones(len(x))*0.5 for x in X_to_eval]
# defie dataset and method that model was trained on to find the model
datasets = ['SimDiff_dim3_ntraces300000_Drandom0.0001-0.5_dt1.0e+00_N5-600_B0.05-0.25_R5-25_subA0-0.7_superA1.3-2_Q1-16']
methods = ['XYZ_SL_DP']
dim = 3 if 'dim3' in datasets[0] else 2

min_max_len = 601
X_padtoken = 0
y_padtoken = 10
batch_size = 32

# get consistent result
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = 'cpu'

# find the model
dir_name = ''
modelpath = '../mlruns/'
modeldir = '36'
use_mlflow = False

# find the model
if use_mlflow:
    import mlflow
    mlflow.set_tracking_uri('file:'+join(os.getcwd(), join("", "mlruns")))
    best_models_sorted = find_models_for(datasets, methods)
else:

    # not sorted tho
    path = '../mlruns/{}'.format(modeldir)
    best_models_sorted = find_models_for_from_path(path)
    print(best_models_sorted)

if not os.path.exists('../deepspt_results/analytics/EEA1NPC1_llsm_ensemble_score.pkl'):
    files_dict = {}
    for modelname in best_models_sorted:
        if use_mlflow:
            model = load_UnetModels(modelname, dir=modelpath, device=device, dim=dim)
        else:
            model = load_UnetModels_directly(modelname, device=device, dim=dim)

        tmp_dict = make_preds(model, X_to_eval, y_to_eval, min_max_len=min_max_len,
                            X_padtoken=X_padtoken, y_padtoken=y_padtoken,
                            batch_size=batch_size, device=device)
        files_dict[modelname] = tmp_dict

    if len(list(files_dict.keys()))>1:
        files_dict['ensemble_score'] = ensemble_scoring(files_dict[list(files_dict.keys())[0]]['masked_score'], 
                                    files_dict[list(files_dict.keys())[1]]['masked_score'], 
                                    files_dict[list(files_dict.keys())[2]]['masked_score'])
        ensemble_score = files_dict['ensemble_score']
        ensemble_pred = [np.argmax(files_dict['ensemble_score'][i], axis=0) for i in range(len(files_dict['ensemble_score']))]

    pickle.dump(ensemble_score, open(dir_name+'../deepspt_results/analytics/EEA1NPC1_llsm_ensemble_score.pkl', 'wb'))
    pickle.dump(ensemble_pred, open(dir_name+'../deepspt_results/analytics/EEA1NPC1_llsm_ensemble_pred.pkl', 'wb'))
else:
    ensemble_score = pickle.load(open(dir_name+'../deepspt_results/analytics/EEA1NPC1_llsm_ensemble_score.pkl', 'rb'))
    ensemble_pred = pickle.load(open(dir_name+'../deepspt_results/analytics/EEA1NPC1_llsm_ensemble_pred.pkl', 'rb'))

# %%

EEA1_flat_pred = np.hstack(np.array(ensemble_pred)[track_labels==EEA1val])
NPC1_flat_pred = np.hstack(np.array(ensemble_pred)[track_labels==NPC1val])

plt.figure(figsize=(12,2))
plt.bar(np.unique(EEA1_flat_pred, return_counts=True)[0], 
        np.unique(EEA1_flat_pred, return_counts=True)[1]/np.sum(np.unique(EEA1_flat_pred, return_counts=True)[1]),
        color='darkred', label='EEA1', alpha=.75,
        align='edge', width=-0.4)
plt.bar(np.unique(NPC1_flat_pred, return_counts=True)[0], 
        np.unique(NPC1_flat_pred, return_counts=True)[1]/np.sum(np.unique(NPC1_flat_pred, return_counts=True)[1]),
        color='dimgrey', label='NPC1', alpha=.75,
        width=0.4, align='edge', 
        )

plt.legend(fontsize=16)
plt.xticks(np.arange(4), ['Normal', 'Directed', 'Confined', 'Subdiffusive'])
plt.ylabel('Time (%track)')
plt.ylim(0,.6)
plt.savefig('../deepspt_results/figures/EEA1vsNPC1_diffusion_barplot.pdf', 
            bbox_inches='tight', pad_inches=0.5)
plt.show()


# %%
endosomal_tracks = tracks
endosomal_pred = np.array(ensemble_pred)
expname_all = np.hstack([EEA1_expname_all, NPC1_expname_all])

fp_datapath = '../_Data/Simulated_diffusion_tracks/'
hmm_filename = 'simulated2D_HMM.json'
dim = 3
dt = 2.7

selected_features = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,
                                19,20,21,23,24,25,27,28,29,30,31,
                                32,33,34,35,36,37,38,39,40,41,42])
window_size = 20
results = Parallel(n_jobs=100)(
        delayed(make_tracks_into_FP_timeseries)(
            track, pred_track, window_size=window_size, selected_features=selected_features,
            fp_datapath=fp_datapath, hmm_filename=hmm_filename, dim=dim, dt=dt)
        for track, pred_track in zip(endosomal_tracks, endosomal_pred))
timeseries_clean = np.array([r[0] for r in results])

length_track = np.hstack([len(t) for t in endosomal_tracks])
pickle.dump(timeseries_clean, open('../deepspt_results/analytics/timeseries_EEA1NPC1.pkl', 'wb'))
pickle.dump(track_labels, open('../deepspt_results/analytics/track_labels_rolling_EEA1NPC1.pkl', 'wb'))
pickle.dump(length_track, open('../deepspt_results/analytics/length_track_EEA1NPC1.pkl', 'wb'))
pickle.dump(endosomal_tracks, open('../deepspt_results/analytics/endosomal_tracks_EEA1NPC1.pkl', 'wb'))
pickle.dump(expname_all, open('../deepspt_results/analytics/expname_all_EEA1NPC1.pkl', 'wb'))
timeseries_clean.shape, timeseries_clean[0].shape, endosomal_tracks.shape, track_labels.shape


# %%
from tqdm import tqdm


tTDP_individuals = []
tTDP_labels = []
FP_list = []
ensemble_pred_list = []
ensemble_score_list = []

fp_datapath = '../_Data/Simulated_diffusion_tracks/'
hmm_filename = 'simulated2D_HMM.json'
dim = 3
dt = 2.7
min_seglen_for_FP = 5
min_pred_length = 5
num_difftypes = 4
max_change_points = 10
add_FP = True
save_PN_name = '../deepspt_results/analytics/20230406_p55_p5_sCMOS_Jacob_Anand_Ricky'
if False:#os.path.exists(save_PN_name+'_vanillaFP.pkl'):
        FP = pickle.load(open(save_PN_name+'_vanillaFP.pkl', 'rb'))
else:
    FP = []
    for track in tqdm(tracks):
        FP.append(create_fingerprint_track(track, fp_datapath, hmm_filename, dim, dt, 'Normal'))
    pickle.dump(FP, open(save_PN_name+'_vanillaFP.pkl', 'wb'))
FP = np.vstack(FP)

new_feature1, new_feature2,new_feature3,\
new_feature4, new_feature5,new_feature8 = gen_temporal_features(ensemble_pred)

inst_msds_D_all = get_inst_msd(tracks, dim, dt)
perc_ND, perc_DM, perc_CD,\
perc_SD, num_cp = get_perc_per_diff(ensemble_pred)
FP_all = np.column_stack([FP, perc_ND.reshape(-1,1), perc_DM.reshape(-1,1), 
                          perc_CD.reshape(-1,1), perc_SD.reshape(-1,1), 
                          num_cp.reshape(-1,1), inst_msds_D_all.reshape(-1,1), 
                          new_feature1.reshape(-1,1), new_feature2.reshape(-1,1), 
                          new_feature3.reshape(-1,1), new_feature4.reshape(-1,1), 
                          new_feature5.reshape(-1,1), new_feature8.reshape(-1,1)])
print(FP_all.shape, FP_all.shape)

# %%

loadpath = '../deepspt_results/EEA1_NPC1_results/precomputed_files/coloc_results/coloc_endo/'
tracks_coloc_all = []
track_coloc_info_all = []

experiment_name_coloc = []
experiment_name_coloc_info = []

for file in np.sort(glob(loadpath+'*', recursive=True)):
    if 'ENDOpotential_coloc_.pkl' in file:
        tracks_coloc = pickle.load(open(file, 'rb'))
        for t in tracks_coloc:
            experiment_name_coloc.append(file.split('_ENDO')[0])
            tracks_coloc_all.append(t)

    if 'ENDOpotential_coloc_info' in file:
        track_coloc_info = pickle.load(open(file, 'rb'))
        for t in track_coloc_info:
            experiment_name_coloc_info.append(file.split('_ENDO')[0])
            track_coloc_info_all.append(t)

print(len(tracks_coloc_all), len(track_coloc_info_all))

assert len(tracks_coloc_all) == len(track_coloc_info_all)

experiment_name_coloc = np.array(experiment_name_coloc)
experiment_name_coloc_info = np.array(experiment_name_coloc_info)

argsorter_expname_coloc = np.argsort(experiment_name_coloc)
argsorter_expname_coloc_info = np.argsort(experiment_name_coloc_info)

tracks_coloc_all_sorted = np.array(tracks_coloc_all)[argsorter_expname_coloc]
track_coloc_info_all_sorted = np.array(track_coloc_info_all)[argsorter_expname_coloc_info]

experiment_name_coloc_sorted = np.array(experiment_name_coloc)[argsorter_expname_coloc]
experiment_name_coloc_info_sorted = np.array(experiment_name_coloc_info)[argsorter_expname_coloc_info]

assert np.mean(experiment_name_coloc_sorted == experiment_name_coloc_info_sorted) == 1

tracks_eea1npc1 = list(EEA1_tracks_all) + list(NPC1_tracks_all)
track_labels_eea1npc1 = np.concatenate([np.ones(len(EEA1_tracks_all))*EEA1val,
                                        np.ones(len(NPC1_tracks_all))*NPC1val])

has_coloc_or_dup = []
eea1_with_coloc = []
npc1_with_coloc = []
for t_idx in track_coloc_info_all_sorted[:,0]:
    eea1_label = track_coloc_info_all_sorted[:,0][track_coloc_info_all_sorted[:,0]==t_idx]
    npc1_label = track_coloc_info_all_sorted[:,1][track_coloc_info_all_sorted[:,0]==t_idx]
    print(eea1_label, npc1_label)
    eea1_with_coloc.append(eea1_label)
    npc1_with_coloc.append(npc1_label)
    
eea1_with_coloc = np.unique(np.hstack(eea1_with_coloc))
npc1_with_coloc = np.unique(np.hstack(npc1_with_coloc))

# %%

EEA1val = 0
NPC1val = 1
tracks = list(EEA1_tracks_all) + list(NPC1_tracks_all)
track_labels = np.concatenate([np.ones(len(EEA1_tracks_all))*EEA1val,
                               np.ones(len(NPC1_tracks_all))*NPC1val])

selected_features = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,
                              19,20,21,23,24,25,27,28,29,30,31,
                              32,33,34,35,36,37,38,39,40,41,42])
X = FP_all.copy()[:,selected_features]
print(len(selected_features))

fp_starts_here = max_change_points*num_difftypes+num_difftypes**2*max_change_points
conditions_to_pred = [0,1]
print('conditions_to_pred', conditions_to_pred)


y = track_labels.copy()
y = y[np.isin(track_labels, conditions_to_pred)]//np.max(track_labels[np.isin(track_labels, conditions_to_pred)])
X = X[np.isin(track_labels, conditions_to_pred)]

pickle.dump(X, open('../deepspt_results/analytics/EEA1_NPC1_only_FPX.pkl', 'wb'))
pickle.dump(y, open('../deepspt_results/analytics/EEA1_NPC1_only_FPy.pkl', 'wb'))

# %%


# %%

EEA1_flat_pred = np.hstack(np.array(ensemble_pred)[track_labels==EEA1val])
NPC1_flat_pred = np.hstack(np.array(ensemble_pred)[track_labels==NPC1val])

plt.figure(figsize=(12,2))
plt.bar(np.unique(EEA1_flat_pred, return_counts=True)[0], 
        np.unique(EEA1_flat_pred, return_counts=True)[1]/np.sum(np.unique(EEA1_flat_pred, return_counts=True)[1]),
        color='darkred', label='EEA1', alpha=.75,
        align='edge', width=-0.4)
plt.bar(np.unique(NPC1_flat_pred, return_counts=True)[0], 
        np.unique(NPC1_flat_pred, return_counts=True)[1]/np.sum(np.unique(NPC1_flat_pred, return_counts=True)[1]),
        color='dimgrey', label='NPC1', alpha=.75,
        width=0.4, align='edge', 
        )

plt.legend(fontsize=16)
plt.xticks(np.arange(4), ['Normal', 'Directed', 'Confined', 'Subdiffusive'])
plt.ylabel('Time (%track)')
plt.ylim(0,.6)
plt.savefig('../deepspt_results/figures/EEA1vsNPC1_diffusion_barplot.pdf', 
            bbox_inches='tight', pad_inches=0.5)
plt.show()

# %%

seed = 42
print('seed', seed)
np.random.seed(seed)
random_idx = np.random.randint(0,len(EEA1_tracks_all)-1, size=400)

EEA1_tracks_all_random = [EEA1_tracks_all[i] for i in random_idx]
NPC1_tracks_all_random = [NPC1_tracks_all[i] for i in random_idx]

plt.figure(figsize=(10,5))
xs = np.tile(np.array(list(range(0,80,4))), 20)
xs2 = np.tile(np.array(list(range(0,80,4))), 20)+100
ys = np.repeat(np.array(list(range(0,80,4))), 20)

for i in range(len(EEA1_tracks_all_random)):
    plt.plot(EEA1_tracks_all_random[i][:,0]-EEA1_tracks_all_random[i][0,0]+xs[i],
             EEA1_tracks_all_random[i][:,1]-EEA1_tracks_all_random[i][0,1]+ys[i], 
             color='darkred', lw=1)
    x = np.random.randint(57,110)
    y = np.random.randint(0,50)
    plt.plot(NPC1_tracks_all_random[i][:,0]-NPC1_tracks_all_random[i][0,0]+xs2[i],
             NPC1_tracks_all_random[i][:,1]-NPC1_tracks_all_random[i][0,1]+ys[i],
             color='dimgrey', lw=1)
plt.aspect_ratio = 1
plt.savefig('../deepspt_results/figures/EEA1vsNPC1_random_tracks.pdf')


# %%
plt.figure(figsize=(5,5))

colors_list = ['darkred', 'dimgrey']
labels_list = ['EEA1 (N:'+str(np.sum(track_labels==0))+')', 
               'NPC1 (N:'+str(np.sum(track_labels==1))+')']


bins = len(np.linspace(0,2.5,75))
fig, ax = plt.subplots(2,1,figsize=(3,4))
for i in np.unique(track_labels)[::-1]:
    ax[0].hist(FP_all[:,0][track_labels==i], 
             density=True,
             color=colors_list[int(i)],
             alpha=0.5, range=(0,2.5),
             bins=bins,
             label=labels_list[int(i)])
    
bins = len(np.linspace(0,0.1,75))
for i in np.unique(track_labels)[::-1]:
    ax[1].hist(FP_all[:,1][track_labels==i], 
             bins=bins, 
             density=True,
             color=colors_list[int(i)],
             alpha=0.5,
             range=(0,.1),
             label=labels_list[int(i)])

# adjust spacing
plt.subplots_adjust(wspace=0.5, hspace=.5)
ax[0].set_xlabel('Alpha')
ax[0].set_ylabel('Density')
ax[1].set_xlabel('D (\u03BCm\u00b2/s)')
ax[1].set_ylabel('Density')
plt.savefig('../deepspt_results/figures/EEA1vsNPC1_FP_alphaD.pdf', bbox_inches='tight', pad_inches=0.5)


# %%
