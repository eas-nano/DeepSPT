# %%
import matplotlib.pyplot as plt
import pickle 
import numpy as np
import torch
import sys
import os 
from sklearn.neural_network import MLPClassifier
from deepspt_src import *
import random
from global_config import globals
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import *
from scipy.stats import mode
from imblearn.over_sampling import RandomOverSampler
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from glob import glob
import time
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# global config variables
globals._parse({})

# set seed
seed = globals.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Prep data
savepath = 'EEA1_NPC1_results/precomputed_files/coloc_results/coloc_rota'

# Define where to find experiments
main_dir = '/scratch/Marilina/20220810_p5_p55_sCMOS_Mari_Rota'
SEARCH_PATTERN = '{}/**/ProcessedTracks.mat'
OUTPUT_NAME = 'rotavirus'
_input = SEARCH_PATTERN.format(main_dir)
files = sorted(glob(_input, recursive=True))
dont_use_experiments = []

files_curated = []
for file in files:
    if 'ch488nmCamA' in file:
        continue
    if 'EX_xyProj' in file:
        continue
    if 'EX_xzProj' in file:
        continue
    if 'EX_yzProj' in file:
        continue
    if 'Ex05' in file:
        files_curated.append(file)
    if 'Ex06' in file:
        files_curated.append(file)
    if 'Ex07' in file:
        files_curated.append(file)
    if 'Ex08' in file:
        files_curated.append(file)
    if 'Ex10' in file:
        files_curated.append(file)
    if 'Ex12' in file:
        files_curated.append(file)
    if 'Ex13' in file:
        files_curated.append(file)
    if 'Ex14' in file:
        files_curated.append(file)
    if 'Ex15' in file:
        files_curated.append(file)
    if 'Ex16' in file:
        files_curated.append(file)
    if 'Ex17' in file:
        files_curated.append(file)
    if 'Ex18' in file:
        files_curated.append(file)
    if 'Ex19' in file:
        files_curated.append(file)

xy_to_um, z_to_um = 0.1, 0.25
min_trace_length = 20
min_brightness = 0
max_brightness = np.inf
do_fuse_ = False
dont_use_high_power = True # exclude high laser power
use_20230415_high_power = False # if false these are not included, if true these are included
use_20230419_high_power = False

# True, True, False: 0.6-1.2mW + 2mW
# True, False, False: 0.6-1.2mW
# False, False, False: all: <= 4mW
files_curated

# %%
# Rota data
def filter_tracks(tracks_filter, Rota_tracks,
                    Rota_timepoints, Rota_frames,
                    Rota_track_ids, Rota_amplitudes,
                    Rota_amplitudes_S, Rota_amplitudes_SM,
                    Rota_amplitudes_sig, Rota_amplitudes_bg):
    Rota_tracks = [Rota_tracks[i] for i in tracks_filter]
    Rota_timepoints = [Rota_timepoints[i] for i in tracks_filter]
    Rota_frames = [Rota_frames[i] for i in tracks_filter]
    Rota_track_ids = [Rota_track_ids[i] for i in tracks_filter]
    Rota_amplitudes = [Rota_amplitudes[i] for i in tracks_filter]
    Rota_amplitudes_S = [Rota_amplitudes_S[i] for i in tracks_filter]
    Rota_amplitudes_SM = [Rota_amplitudes_SM[i] for i in tracks_filter]
    Rota_amplitudes_sig = [Rota_amplitudes_sig[i] for i in tracks_filter]
    Rota_amplitudes_bg = [Rota_amplitudes_bg[i] for i in tracks_filter]
    return Rota_tracks, Rota_timepoints, Rota_frames,\
            Rota_track_ids, Rota_amplitudes,\
            Rota_amplitudes_sig, Rota_amplitudes_bg

multiplier_nominator = (1.2*10)
num_t_rota_exp = []
from joblib import Parallel, delayed
import time

def use_file_rota(file, ch_cam_name='ch560nmCamA'):
    if ch_cam_name not in file:
        return False
    return True


def parallel_loader_rota(file):

    Rota_df, seqOfEvents_list, keep_idx_len,\
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
    
    Rota_tracks, Rota_timepoints, Rota_frames,\
    Rota_track_ids, Rota_amplitudes,\
    Rota_amplitudes_S, Rota_amplitudes_SM,\
    Rota_amplitudes_sig, Rota_amplitudes_bg,\
    Rota_catIdx = curate_3D_data_to_tracks(Rota_df, xy_to_um, z_to_um)
    Rota_compound_idx = np.array(list(range(len(Rota_tracks))))[Rota_catIdx>4]
    handle_compound_tracks(Rota_compound_idx, seqOfEvents_list_curated, Rota_tracks, Rota_timepoints, Rota_frames,
                            Rota_track_ids, Rota_amplitudes, Rota_amplitudes_S,
                                Rota_amplitudes_SM, Rota_amplitudes_sig, Rota_amplitudes_bg,
                                min_trace_length, min_brightness)
    num_t_rota_exp.append(len(Rota_tracks))
    if do_fuse_:
        Rota_tracks, Rota_timepoints, Rota_frames,\
        Rota_track_ids, Rota_amplitudes,\
        Rota_amplitudes_S, Rota_amplitudes_SM,\
        Rota_amplitudes_sig, Rota_amplitudes_bg, = fuse_tracks(
                                        Rota_compound_idx, Rota_tracks, 
                                        Rota_timepoints, Rota_frames,
                                        Rota_track_ids, Rota_amplitudes, 
                                        Rota_amplitudes_S, Rota_amplitudes_SM, 
                                        Rota_amplitudes_sig, Rota_amplitudes_bg,
                                        min_trace_length, min_brightness,
                                        blinking_forgiveness=1)

    files_to_save = [file.split('ProcessedTracks.mat')[0] for i in range(len(Rota_tracks))] 
    exp_to_save = [file.split(ch_cam_name)[0] for i in range(len(Rota_tracks))]
    return files_to_save, exp_to_save,\
            Rota_tracks ,Rota_timepoints ,Rota_track_ids,\
            Rota_frames, Rota_amplitudes,\
            Rota_amplitudes_sig, Rota_amplitudes_bg

t = time.time()
t1 = datetime.datetime.now()

ch_cam_name = 'ch560nmCamB'
results = Parallel(n_jobs=30)(
                delayed(parallel_loader_rota)(file) for file in files_curated
                if use_file_rota(file, ch_cam_name=ch_cam_name))
print(datetime.datetime.now(), time.time()-t)
print('load')

Rota_filenames_all = np.array(flatten_list([r[0] for r in results]))
Rota_expname_all = np.array(flatten_list([r[1] for r in results]))
Rota_tracks_all = np.array(flatten_list([r[2] for r in results]), dtype=object)
Rota_timepoints_all = np.array(flatten_list([r[3] for r in results]), dtype=object)
Rota_track_ids_all = np.array(flatten_list([r[4] for r in results]), dtype=object)
Rota_frames_all = np.array(flatten_list([r[5] for r in results]), dtype=object)
Rota_amplitudes_all = np.array(flatten_list([r[6] for r in results]), dtype=object)
Rota_amplitudes_sig_all = np.array(flatten_list([r[7] for r in results]), dtype=object)
Rota_amplitudes_bg_all = np.array(flatten_list([r[8] for r in results]), dtype=object)
len(Rota_filenames_all), Rota_tracks_all.shape

dist_to_ap2hull_all = pickle.load(open('deepspt_results/analytics/Rota560nm_dist_to_ap2hull_all.pkl', 'rb'))
dist_to_ap2hull_all = np.array(dist_to_ap2hull_all)

Rota_expname_all

# %%

print(np.unique([e.split('/')[5] for e in Rota_expname_all]))
print(np.unique([e.split('/')[4] for e in Rota_expname_all]))

print(np.unique([e.split('/')[5] for e in Rota_expname_all]).shape)
print(np.unique([e.split('/')[4] for e in Rota_expname_all]).shape)

# %%

ex05 = [20,23,25,30,35] 
ex06 = []
ex07 = [44]
ex08 = [13,23,26,28,34,42,54,67,69]
ex10 = [2,21]
ex12 = [37,60,115,119] 
ex13 = [1,2,3,7] 
ex14 = [78,86,108,132,144] 
ex15 = [11,26,84,100,101,103,121,148,159] 
ex16 = [11,45,57,68,84,198] 
ex17 = [2,17,24,25,26,34] 
ex18 = [8,13,16,21,28,41,42,68,110] 
ex19 = [8,13,17,71,79,81]
mega_list_manual_annot = ex05+ex06+ex07+ex08+ex10+ex12+ex13+ex14+ex15+ex16+ex17+ex18+ex19

link_error_dict = {}

print()
manually_annotated_idx = []
manually_annotated_names = []
for i in range(len(Rota_filenames_all)):
    if 'Ex05' in Rota_filenames_all[i]:
        for ui in ex05:
            if 'p'+str(ui)+'_' in Rota_track_ids_all[i]:
                if ui==30:
                    #maybe first few or last few frames?
                    print(ui, i)
                manually_annotated_idx.append(i)
                manually_annotated_names.append(Rota_filenames_all[i]+Rota_track_ids_all[i])
    if 'Ex06' in Rota_filenames_all[i]:
        for ui in ex06:
            if 'p'+str(ui)+'_' in Rota_track_ids_all[i]:
                manually_annotated_idx.append(i)
                manually_annotated_names.append(Rota_filenames_all[i]+Rota_track_ids_all[i])
    if 'Ex07' in Rota_filenames_all[i]:
        for ui in ex07:
            if 'p'+str(ui)+'_' in Rota_track_ids_all[i]:
                manually_annotated_idx.append(i)
                manually_annotated_names.append(Rota_filenames_all[i]+Rota_track_ids_all[i])
    if 'Ex08' in Rota_filenames_all[i]:
        for ui in ex08:
            if 'p'+str(ui)+'_' in Rota_track_ids_all[i]:
                manually_annotated_idx.append(i)
                manually_annotated_names.append(Rota_filenames_all[i]+Rota_track_ids_all[i])
    if 'Ex10' in Rota_filenames_all[i]:
        for ui in ex10:
            if 'p'+str(ui)+'_' in Rota_track_ids_all[i]:
                manually_annotated_idx.append(i)
                manually_annotated_names.append(Rota_filenames_all[i]+Rota_track_ids_all[i])
    if 'Ex12' in Rota_filenames_all[i]:
        for ui in ex12:
            if 'p'+str(ui)+'_' in Rota_track_ids_all[i]:
                manually_annotated_idx.append(i)
                manually_annotated_names.append(Rota_filenames_all[i]+Rota_track_ids_all[i])
    if 'Ex13' in Rota_filenames_all[i]:
        for ui in ex13:
            if 'p'+str(ui)+'_' in Rota_track_ids_all[i]:
                if ui==7:
                    link_error_dict[i] = 20
                manually_annotated_idx.append(i)
                manually_annotated_names.append(Rota_filenames_all[i]+Rota_track_ids_all[i])
    if 'Ex14' in Rota_filenames_all[i]:
        for ui in ex14:
            if 'p'+str(ui)+'_' in Rota_track_ids_all[i]:
                manually_annotated_idx.append(i)
                manually_annotated_names.append(Rota_filenames_all[i]+Rota_track_ids_all[i])
    if 'Ex15' in Rota_filenames_all[i]:
        for ui in ex15:
            if 'p'+str(ui)+'_' in Rota_track_ids_all[i]:
                if ui==100:
                    link_error_dict[i] = 3
                manually_annotated_idx.append(i)
                manually_annotated_names.append(Rota_filenames_all[i]+Rota_track_ids_all[i])
    if 'Ex16' in Rota_filenames_all[i]:
        for ui in ex16:
            if 'p'+str(ui)+'_' in Rota_track_ids_all[i]:
                if ui==11:
                    if len(Rota_tracks_all[i])==133:
                        link_error_dict[i] = 115-len(Rota_tracks_all[i])
                manually_annotated_idx.append(i)
                manually_annotated_names.append(Rota_filenames_all[i]+Rota_track_ids_all[i])
    if 'Ex17' in Rota_filenames_all[i]:
        for ui in ex17:
            if 'p'+str(ui)+'_' in Rota_track_ids_all[i]:
                if ui==17:
                    link_error_dict[i] = 9
                manually_annotated_idx.append(i)
                manually_annotated_names.append(Rota_filenames_all[i]+Rota_track_ids_all[i])
    if 'Ex18' in Rota_filenames_all[i]:
        for ui in ex18:
            if 'p'+str(ui)+'_' in Rota_track_ids_all[i]:
                if ui==16:
                    link_error_dict[i] = 75-len(Rota_tracks_all[i])
                manually_annotated_idx.append(i)
                manually_annotated_names.append(Rota_filenames_all[i]+Rota_track_ids_all[i])
    if 'Ex19' in Rota_filenames_all[i]:
        for ui in ex19:
            if 'p'+str(ui)+'_' in Rota_track_ids_all[i]:
                manually_annotated_idx.append(i)
                manually_annotated_names.append(Rota_filenames_all[i]+Rota_track_ids_all[i])

manually_annotated_idx = np.array(manually_annotated_idx)
len(manually_annotated_names)
manually_annotated_idx, link_error_dict

print('correct link error')
FP_pred_manual = []
manual_tracks_all = []
manual_pred_all = []
manual_idx_t_all = []

eot = 1000
frames_unoat_start_manual = {'0':10, '1':3, '1_2':30, '2':35, '2_2':45, '3':20,
                             '5':30,
                             '7':13, '8':100, '9':13, '9_2':62, '10':35,
                             '11':52, '12':0 ,'13':7,'14':18,
                             '15':11, '15_2':40 ,'16':15, '16_2':46, '17':5,'18':4,
                             '19':60, '20':35, '21':10, '21_2':35, '22':40,
                             '23':15, '24':20, '25':0,
                             '28':16, '29':29, '30':33,
                             '31':17, '32':20, '33':25, '34':5,
                             '35':18, '35_2':33, '36':0, '37':34, '38':10,
                             '39':21, '40':6, '40_2':12, '41':10, 
                             '43':35, '44':10, '45':15, '46':15,
                             '49':10, '50':10, 
                             '51':17, '53':15, '53_2':40, '54':67, 
                             '55':5, '56':10, '57':0, '58':15,
                             '59':10, '60':15, '61':55, '62':70,
                             '63':10, '64':40, '65':15, '66':13,
                             '68':35, '69':25, '70':24, 
                             '71':60, '71_2':105, '72':10, '72_2':50,
                             '73':5, '73_2':30, '74':35
                             }

frames_unoat_end_manual = {'0':40,'1':50,'2':eot, '3':35,
                           '5':50,
                           '7':18, '8':110, '9':90, '10':50,
                           '11':90 ,'12':20, '13':25, '14':40,
                           '15':70,'16':50, '17':70, '18':15,
                           '19':75, '20':55, '21':60, '22':55,
                           '23':70, '24':eot, '25':20,
                           '28':30, '29':60, '30':43,
                           '31':30, '32':30, '33':45, '34':60,
                           '35':40, '36':40, '37':45, '38':50,
                           '39':32, '40':40, '41':30, 
                           '43':eot,'44':20, '45':25, '46':20,
                           '49':30, '50':35, 
                           '51':40, '53':105, '54':74, 
                           '55':20, '56':50, '57':30, '58':22,
                           '59':25, '60':15, '61':95, '62':107,
                           '63':50, '64':70, '65':57, '66':40,
                           '68':75, '69':80, '70':85, 
                           '71':eot,'72':eot,
                           '73':eot,'74':eot}

# fix linking errors
Rota_tracks_all_v2 = []
for i in range(len(Rota_tracks_all)):
    
    if str(i) in np.array(list(frames_unoat_start_manual.keys())):
        idx = str(i)
        if '_' in idx:
            continue
        manual_idx_t = manually_annotated_idx[int(idx)]

        if manual_idx_t in np.array(list(link_error_dict.keys())):
            up_to_frame = link_error_dict[manual_idx_t]
            if up_to_frame<0:
                RT = Rota_tracks_all[manual_idx_t][:up_to_frame]
            else:
                RT = Rota_tracks_all[manual_idx_t][up_to_frame:]
        else:
            RT = Rota_tracks_all[manual_idx_t]
        if len(RT)<5:
            continue
        
        manual_idx_t_all.append(manual_idx_t)
        
        manual_track = RT
        Rota_tracks_all_v2.append(RT)
    else:
        Rota_tracks_all_v2.append(Rota_tracks_all[i])

Rota_tracks_all = np.array(Rota_tracks_all_v2, dtype=object)


# %%
ni = 9
track_idx = manual_idx_t_all[ni]
plt.figure()
plt.plot(Rota_amplitudes_all[track_idx][:,0], 
         color='#0091ad', alpha=0.85, zorder=0,
         label='Raw intensity')
plt.scatter(90, Rota_amplitudes_all[track_idx][90,0], 
            color='k', alpha=1, zorder=1, label='Manual annotation')
plt.legend()
plt.xlabel('Time (frames)')
plt.ylabel('Intensity (a.u.)')
plt.savefig('deepspt_results/figures/Rota560nm_manualannot_track_example.pdf')

# %%

# Rota_tracks_all = np.array(Rota_tracks_all, dtype=object)
tracks = Rota_tracks_all

X = [x-x[0] for x in tracks]
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

savename_score = 'deepspt_results/analytics/Rota560_ensemble_score.pkl'
savename_pred = 'deepspt_results/analytics/Rota560_ensemble_pred.pkl'
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

selected_features = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,
                              19,20,21,23,24,25,27,28,29,30,31,
                              32,33,34,35,36,37,38,39,40,41,42])

frame_uncoat_end_to_use = frames_unoat_end_manual 

FP_pred_manual = []
manual_tracks_all = []
manual_pred_all = []
manual_idx_t_all = []
uncoat_start_all = []
uncoat_end_all = []
for i,idx in tqdm(enumerate(frames_unoat_start_manual.keys())):
    if '_' in idx:
        continue
    manual_idx_t = manually_annotated_idx[int(idx)]

    if manual_idx_t in np.array(list(link_error_dict.keys())):
        up_to_frame = link_error_dict[manual_idx_t]
        if up_to_frame<0:
            RT = Rota_tracks_all[manual_idx_t]
            EP = ensemble_pred[manual_idx_t]
            uncoat_start = frames_unoat_start_manual[idx]
            uncoat_end = frame_uncoat_end_to_use[idx]
        else:
            RT = Rota_tracks_all[manual_idx_t]
            EP = ensemble_pred[manual_idx_t]
            uncoat_start = frames_unoat_start_manual[idx]-up_to_frame
            uncoat_end = frame_uncoat_end_to_use[idx]-up_to_frame
    else:
        RT = Rota_tracks_all[manual_idx_t]
        EP = ensemble_pred[manual_idx_t]
        uncoat_start = frames_unoat_start_manual[idx]
        uncoat_end = frame_uncoat_end_to_use[idx]
    if len(RT)<5:
        continue
    
    manual_idx_t_all.append(manual_idx_t)

    uncoat_start_all.append(uncoat_start)
    uncoat_end_all.append(uncoat_end)

    manual_track = RT
    manual_pred = EP
    manual_tracks_all.append(manual_track)
    manual_pred_all.append(manual_pred)

fp_datapath = '_Data/Simulated_diffusion_tracks/'
hmm_filename = 'simulated2D_HMM.json'
dt = 4
min_seglen_for_FP = 5
min_pred_length = 5
num_difftypes = 4
max_change_points = 10
add_FP = True
save_PN_name = 'deepspt_results/analytics/20220810_p5_p55_sCMOS_Mari_Rota'
window_size = 30
results = Parallel(n_jobs=100)(
        delayed(make_tracks_into_FP_timeseries)(
            track, pred_track, dt=dt, window_size=window_size, selected_features=selected_features,
            fp_datapath=fp_datapath, hmm_filename=hmm_filename, dim=dim)
        for track, pred_track in zip(manual_tracks_all, manual_pred_all))
timeseries_clean = np.array([r[0] for r in results])

length_track = np.hstack([len(t) for t in manual_tracks_all])

uncoat_end_all_clean = np.array(uncoat_end_all).copy()

# ensure there is atleast > 1 point in each end of tracks for training

keep_idx = []
min_frames_from_edge = 2
for f,t in zip(uncoat_end_all_clean, manual_tracks_all):
    if len(t[:f])>=min_frames_from_edge and len(t[f:])>=min_frames_from_edge:
        keep_idx.append(True)
    else:
        keep_idx.append(False)
keep_idx = np.array(keep_idx)

timeseries_clean = timeseries_clean[keep_idx]
length_track = length_track[keep_idx]
uncoat_end_all_clean = uncoat_end_all_clean[keep_idx]
uncoat_start_all = np.array(uncoat_start_all)[keep_idx]
manual_tracks_all = np.array(manual_tracks_all)[keep_idx]

pickle.dump(timeseries_clean, open('deepspt_results/analytics/timeseries_clean560nm.pkl', 'wb'))
pickle.dump(length_track, open('deepspt_results/analytics/length_track560nm.pkl', 'wb'))
pickle.dump(uncoat_end_all_clean, open('deepspt_results/analytics/frame_change_pruned560nm.pkl', 'wb'))
pickle.dump(uncoat_start_all, open('deepspt_results/analytics/frame_change_pruned560nm_v2.pkl', 'wb'))
pickle.dump(manual_tracks_all, open('deepspt_results/analytics/escape_tracks_all560nm.pkl', 'wb'))

# %%
