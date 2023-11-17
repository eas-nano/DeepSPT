# %%
import numpy as np
import pickle
import matplotlib.pyplot as plt
import datetime
from deepspt_src import *
from global_config import globals
import warnings
from joblib import Parallel, delayed
warnings.filterwarnings("ignore")

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
Drandomranges_pairs = [[2*10**-3,   5*10**-3],   [5*10**-3,   9*10**-3]]
Nrange = [5,200] # length of tracks
Branges = [[0.05,0.25],[0.15,0.35]] # boundary geometry
Rranges = [[5,10],[8,17]] # relative active diffusion
subalpharanges = [[0.3,0.5], [0.5, 0.7]] # subdiffusion exponent
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
output_name = 'tester_set2'+str(dim) # name of output file - change to get new tracks if already run
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
        Brange = Branges[i]
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

# fuse 
# changing_diffusion_list_all[:n_changing_traces] 
# with changing_diffusion_list_all[n_changing_traces:]
# to get tracks that at a random time switches class
# return list with label per time step
print(np.mean(Drandomranges_pairs))
glued_tracks = []
glued_labels = []
for i in range(len(changing_diffusion_list_all[:n_changing_traces])):
    first = changing_diffusion_list_all[i]
    second = changing_diffusion_list_all[i+n_changing_traces]
    # move second trace to end of first trace and add noise
    second[:,0] += np.random.normal(first[-1,0], 
                    np.sqrt(dim*dt*np.mean(Drandomranges_pairs)))
    second[:,1] += np.random.normal(first[-1,1],
                    np.sqrt(dim*dt*np.mean(Drandomranges_pairs)))
    glued_tracks.append(
        np.concatenate((first, 
                        second)))
    glued_labels.append(
        np.concatenate((np.zeros(len(changing_diffusion_list_all[i])),
                        np.ones(len(changing_diffusion_list_all[i+n_changing_traces])))))
# %%
i = np.random.randint(len(glued_tracks))
plt.plot(glued_tracks[i][:,0], glued_tracks[i][:,1],
         c='k')
plt.scatter(glued_tracks[i][:,0], glued_tracks[i][:,1],
            c=glued_labels[i], zorder=10, s=10)

# %%

# prep data
tracks = glued_tracks
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
                                savename_score='ensemble_score.pkl',
                                savename_pred='ensemble_pred.pkl',
                                use_temperature=use_temperature)

# pretrained HMM for fingerprints
fp_datapath = '_Data/Simulated_diffusion_tracks/'
hmm_filename = 'simulated2D_HMM.json'

window_size = 20
selected_features = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,
                            19,20,21,23,24,25,27,28,29,30,31,
                            32,33,34,35,36,37,38,39,40,41,42])

results2 = Parallel(n_jobs=2)(
        delayed(make_tracks_into_FP_timeseries)(
            track, pred_track, window_size=window_size, selected_features=selected_features,
            fp_datapath=fp_datapath, hmm_filename=hmm_filename, dim=dim, dt=dt)
        for track, pred_track in zip(glued_tracks, ensemble_pred))
timeseries_clean = np.array([r[0] for r in results2])

# %%

