# %%
import plotly.graph_objects as go
import numpy as np
from scipy.spatial import ConvexHull
import pickle 
import numpy as np
import sys
sys.path.append('../')
from deepspt_src import *
from global_config import globals
import matplotlib.pyplot as plt
from sklearn.decomposition import *
import tifffile as tiff
import glob
from scipy.ndimage import gaussian_filter, label
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pickle
import numpy as np
from scipy.spatial import ConvexHull
from pygel3d import hmesh
import datetime
#import ipyvolume as ipv
from scipy.ndimage import label, generate_binary_structure
from skimage.segmentation import find_boundaries
from glob import glob
import os
from joblib import Parallel, delayed
import time
import random
import pickle 
import os 
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from global_config import globals
from glob import glob
import alphashape
from scipy.optimize import minimize
import trimesh
import time
from joblib import Parallel, delayed

def use_file(file, ch_cam_name='ch488nmCamB'):
    
    if ch_cam_name not in file:
        return False
    if 'EX_xyProj' in file or 'EX_yzProj' in file or 'EX_xzProj' in file:
        return False
    
    return True


def parallel_loader(file, ch_cam_name='ch488nmCamB'):
    _df, seqOfEvents_list, keep_idx_len,\
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

    if len(_df)>=1:
        
        tracks, timepoints, frames,\
        track_ids, amplitudes,\
        amplitudes_S, amplitudes_SM,\
        amplitudes_sig, amplitudes_bg,\
        catIdx = curate_3D_data_to_tracks(_df, xy_to_um, z_to_um)
        compound_idx = np.array(list(range(len(tracks))))[catIdx>4]
        handle_compound_tracks(compound_idx, seqOfEvents_list_curated, 
                                tracks, timepoints, frames,
                            track_ids, amplitudes, amplitudes_S,
                            amplitudes_SM, amplitudes_sig, amplitudes_bg,
                            min_trace_length, min_brightness)

        if do_fuse_:
            tracks, timepoints, frames,\
            track_ids, amplitudes,\
            amplitudes_S, amplitudes_SM,\
            amplitudes_sig, amplitudes_bg, = fuse_tracks(
                                            compound_idx, tracks, 
                                            timepoints, frames,
                                            track_ids, amplitudes, 
                                            amplitudes_S, amplitudes_SM, 
                                            amplitudes_sig, amplitudes_bg,
                                            min_trace_length, min_brightness,
                                            blinking_forgiveness=1)
            
        files_to_save = [file.split('ProcessedTracks.mat')[0] for i in range(len(tracks))] 
        exp_to_save = [file.split(ch_cam_name)[0] for i in range(len(tracks))]
        return files_to_save, exp_to_save,\
            tracks, timepoints, frames,\
            track_ids, amplitudes,\
            amplitudes_S, amplitudes_SM,\
            amplitudes_sig, amplitudes_bg
    else:
        return [file.split(ch_cam_name)[0]], [file.split(ch_cam_name)[0]], [np.ones((1,3))*-1],\
                [np.ones((1,3))*-1], [np.ones((1,3))*-1], [np.ones((1,3))*-1],\
                    [np.ones((1,3))*-1], [np.ones((1,3))*-1], [np.ones((1,3))*-1]

def dist(hull, points):
    # Construct PyGEL Manifold from the convex hull
    m = hmesh.Manifold()
    for s in hull.simplices:
        m.add_face(hull.points[s])

    dist = hmesh.MeshDistance(m)
    res = []
    for p in points:
        # Get the distance to the point
        # But don't trust its sign, because of possible
        # wrong orientation of mesh face
        d = dist.signed_distance(p)

        # Correct the sign with ray inside test
        if dist.ray_inside_test(p):
            if d > 0:
                d *= -1
        else:
            if d < 0:
                d *= -1
        res.append(d)
    return np.array(res)


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
# Define where to find experiments
main_dir = '../_Data/LLSM_data/Singlelabeled_Rotavirus_AP2/20220810_p5_p55_sCMOS_Mari_Rota'
SEARCH_PATTERN = '{}/**/ProcessedTracks.mat'
OUTPUT_NAME = 'rotavirus'
_input = SEARCH_PATTERN.format(main_dir)
files = sorted(glob(_input, recursive=True))
dont_use_experiments = []

files_curated = []
for file in files:
    if 'ch488nmCamA' not in file:
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
files_curated

# %%
print(len(files_curated))
print(len(np.unique([e.split('/Ex')[0] for e in files_curated])))
print(np.unique([e.split('/Ex')[0] for e in files_curated]))

print(len(files_curated))
print(len(np.unique([e.split('/')[4] for e in files_curated])))
print(len(np.unique([e.split('/')[5] for e in files_curated])))

# %%

t = time.time()
t1 = datetime.datetime.now()
print(t1)

ch_cam_name = 'ch488nmCamA'
results = Parallel(n_jobs=5)(
                delayed(parallel_loader)(file, ch_cam_name=ch_cam_name) 
                                         for file in files_curated
                if use_file(file, ch_cam_name=ch_cam_name))
print(datetime.datetime.now(), time.time()-t)
print('load')

filenames_all = np.array(flatten_list([r[0] for r in results]))
expname_all = np.array(flatten_list([r[1] for r in results]))
tracks_all = np.array(flatten_list([r[2] for r in results]), dtype=object)
timepoints_all = np.array(flatten_list([r[3] for r in results]), dtype=object)
frames_all = np.array(flatten_list([r[4] for r in results]), dtype=object)
track_ids_all = np.array(flatten_list([r[5] for r in results]), dtype=object)

# %%
print('idx 5,6,7,8,9,10 har mange tracks')

alpha = 0.2
alpha_vals = [0.2, 0.3, 0.2, 0.3, 0.15, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.2]

print(alpha, 'alpha')
idx = 7
expidx = idx
experiments_unique = np.unique(expname_all)
print(experiments_unique[idx])

tracks_exp = tracks_all[expname_all==experiments_unique[idx]]
print(len(tracks_exp),)
# Flatten the trajectories into a single array of points
points = np.hstack(
    [[traj[:,0][[np.argmin(traj[:,0]), np.argmax(traj[:,0])]], 
      traj[:,1][[np.argmin(traj[:,1]), np.argmax(traj[:,1])]], 
      traj[:,2][[np.argmin(traj[:,2]), np.argmax(traj[:,2])]]] for traj in tracks_exp]
).T

points_all = np.array(
    [[x, y, z] for traj in tracks_exp for x, y, z 
     in zip(traj[::2,0], traj[::2,1], traj[::2,2])]
)

save_path = experiments_unique[idx]

angle_degrees = 30

tracks_exp_rotated = []
for t in tracks_exp:
    tmp = []
    for p in t:
        tmp.append(rotate_around_y(p, angle_degrees))
    tracks_exp_rotated.append(np.vstack(tmp))

points_all_rotated = np.array(
    [[x, y, z] for traj in tracks_exp_rotated for x, y, z 
     in zip(traj[:,0], traj[:,1], traj[:,2])]
)


# Replace this with your point cloud data
point_cloud = points_all_rotated.copy()

# Grid size
grid_size = 5

# Find min and max values for x and y
x_min, x_max = np.min(point_cloud[:, 0]), np.max(point_cloud[:, 0])
y_min, y_max = np.min(point_cloud[:, 1]), np.max(point_cloud[:, 1])

# Compute the number of bins for x and y
x_bins = int((x_max - x_min) / grid_size) + 1
y_bins = int((y_max - y_min) / grid_size) + 1

# Create a pandas DataFrame from the point cloud
df = pd.DataFrame(point_cloud, columns=['x', 'y', 'z'])

# Bin the x and y coordinates
df['x_bin'] = pd.cut(df['x'], bins=np.linspace(x_min, x_max, x_bins+1), include_lowest=True)
df['y_bin'] = pd.cut(df['y'], bins=np.linspace(y_min, y_max, y_bins+1), include_lowest=True)

# Group the DataFrame by the binned x and y coordinates and find the coordinates with the largest z values
result = df.groupby(['x_bin', 'y_bin']).apply(lambda group: group.loc[group['z'].idxmin()])

max_z_coord = result[['x', 'y', 'z']].dropna().values
print(max_z_coord.shape)

# Generate random 3D points
points = max_z_coord

# Find the mean and covariance matrix of the points
mean = np.mean(points, axis=0)
cov = np.cov(points.T)
maha_distance = mahalanobis_distance(points, mean, cov)
points = points[maha_distance < 1.8]

# Find the optimal parameters for the plane
initial_guess = [-1, 1, 1, 1]
result = minimize(sum_squared_distances, initial_guess, args=(points,))
optimal_params = result.x

# Print the optimal parameters
print(f"Optimal parameters: a = {optimal_params[0]}, b = {optimal_params[1]}, c = {optimal_params[2]}, d = {optimal_params[3]}")

# distance to plane
distances_to_coverslip = distance_to_plane(optimal_params, points_all_rotated)

plt.hist(distances_to_coverslip, bins=100)
plt.vlines(0.4, 0, 1000, color='red')

# Create the meshgrid for the plane
x_range = np.linspace(np.min(points_all_rotated[:, 0]), np.max(points_all_rotated[:, 0]), 10)
y_range = np.linspace(np.min(points_all_rotated[:, 1]), np.max(points_all_rotated[:, 1]), 10)
x_grid, y_grid = np.meshgrid(x_range, y_range)
z_grid = (-optimal_params[0] * x_grid - optimal_params[1] * y_grid - optimal_params[3]) / optimal_params[2]

# Calculate the 3D alpha shape
alpha_shape = alphashape.alphashape(points_all_rotated, alpha)

# Get the vertices and faces of the Trimesh object
vertices = alpha_shape.vertices
faces = alpha_shape.faces

dist_to_hull = trimesh.proximity.signed_distance(alpha_shape, points_all_rotated)
cell_points_rota = points_all_rotated#[distances_to_coverslip > 0.4]
filtering = (dist_to_hull > -3) & (distances_to_coverslip > 0.)
print(cell_points_rota[filtering, 0].shape, cell_points_rota[:, 0].shape)


tracks_filtering = []
dist_to_cs_tracks = []
dist_to_tracks = []
for i, t in enumerate(tracks_exp_rotated):
    dist_to = trimesh.proximity.signed_distance(alpha_shape, t)
    dist_to_thresholded = dist_to > -3
    dist_to_cs = distance_to_plane(optimal_params, t)
    dist_to_cs_thresholded = dist_to_cs > 0.05

    dist_to_cs_tracks.append(dist_to_cs)
    dist_to_tracks.append(dist_to)
    
    if np.mean(dist_to_cs_thresholded) > 0.5 and np.mean(dist_to_thresholded) > 0.5:
        tracks_filtering.append(i)
    # if np.mean(dist_to_thresholded) > 0.5:
    #     tracks_filtering.append(i)

tracks_filtering = np.array(tracks_filtering)
dist_to_cs_tracks = np.array(dist_to_cs_tracks, dtype=object)
dist_to_tracks = np.array(dist_to_tracks, dtype=object)
tracks_exp_rotated = np.array(tracks_exp_rotated, dtype=object)


# %%

tracks = tracks_all
print(len(tracks))
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
    def find_models_for_from_path(path):
        # load from path
        files = sorted(glob(path+'/*/*_UNETmodel.torch', recursive=True))
        return files

    # not sorted tho
    path = '../mlruns/{}'.format(modeldir)
    best_models_sorted = find_models_for_from_path(path)
    print(best_models_sorted)

if not os.path.exists(dir_name+'../deepspt_results/analytics/RotaEEA1NPC1_ensemble_score.pkl'):
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

    pickle.dump(ensemble_score, open(dir_name+'../deepspt_results/analytics/AP2_ensemble_score.pkl', 'wb'))
    pickle.dump(ensemble_pred, open(dir_name+'../deepspt_results/analytics/AP2_ensemble_pred.pkl', 'wb'))
else:
    ensemble_score = pickle.load(open(dir_name+'../deepspt_results/analytics/AP2_ensemble_score.pkl', 'rb'))
    ensemble_pred = pickle.load(open(dir_name+'../deepspt_results/analytics/AP2_ensemble_pred.pkl', 'rb'))
ensemble_pred = np.array(ensemble_pred)

ensemble_pred.shape

# %%

tracks_filtering_all = []
dist_to_cs_tracks_all = []
dist_to_tracks_all = []
tracks_exp_rotated_all = []
dist_to_ap2hull_all = []
alpha_vals = [0.2, 0.3, 0.2, 0.3, 0.15, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.2]
spatial_label_all_exp = []
for exp_i,exp in enumerate(np.unique(expname_all)):
    print(type(tracks_all))
    tracks_exp = tracks_all[expname_all==exp]

    # Flatten the trajectories into a single array of points
    points = np.hstack(
        [[traj[:,0][[np.argmin(traj[:,0]), np.argmax(traj[:,0])]], 
        traj[:,1][[np.argmin(traj[:,1]), np.argmax(traj[:,1])]], 
        traj[:,2][[np.argmin(traj[:,2]), np.argmax(traj[:,2])]]] for traj in tracks_exp]
    ).T

    points_all = np.array(
        [[x, y, z] for traj in tracks_exp for x, y, z 
        in zip(traj[::2,0], traj[::2,1], traj[::2,2])]
    )

    save_path = exp_i
    angle_degrees = 60
    tracks_exp_rotated = []
    for t in tracks_exp:
        tmp = []
        for p in t:
            tmp.append(rotate_around_y(p, angle_degrees))
        tracks_exp_rotated.append(np.vstack(tmp))

    points_all_rotated = np.array(
        [[x, y, z] for traj in tracks_exp_rotated for x, y, z 
        in zip(traj[:,0], traj[:,1], traj[:,2])]
    )

    # Replace this with your point cloud data
    point_cloud = points_all_rotated.copy()

    # Grid size
    grid_size = 5

    # Find min and max values for x and y
    x_min, x_max = np.min(point_cloud[:, 0]), np.max(point_cloud[:, 0])
    y_min, y_max = np.min(point_cloud[:, 1]), np.max(point_cloud[:, 1])

    # Compute the number of bins for x and y
    x_bins = int((x_max - x_min) / grid_size) + 1
    y_bins = int((y_max - y_min) / grid_size) + 1

    # Create a pandas DataFrame from the point cloud
    df = pd.DataFrame(point_cloud, columns=['x', 'y', 'z'])

    # Bin the x and y coordinates
    df['x_bin'] = pd.cut(df['x'], bins=np.linspace(x_min, x_max, x_bins+1), include_lowest=True)
    df['y_bin'] = pd.cut(df['y'], bins=np.linspace(y_min, y_max, y_bins+1), include_lowest=True)

    # Group the DataFrame by the binned x and y coordinates and find the coordinates with the largest z values
    result = df.groupby(['x_bin', 'y_bin']).apply(lambda group: group.loc[group['z'].idxmin()])
    max_z_coord = result[['x', 'y', 'z']].dropna().values

    # Generate random 3D points
    points = max_z_coord

    # Find the mean and covariance matrix of the points
    mean = np.mean(points, axis=0)
    cov = np.cov(points.T)
    maha_distance = mahalanobis_distance(points, mean, cov)
    points = points[maha_distance < 1.8]

    # Find the optimal parameters for the plane
    initial_guess = [-1, 1, 1, 1]
    result = minimize(sum_squared_distances, initial_guess, args=(points,))
    optimal_params = result.x

    # Print the optimal parameters
    print(f"Optimal parameters: a = {optimal_params[0]}, b = {optimal_params[1]}, c = {optimal_params[2]}, d = {optimal_params[3]}")

    # distance to plane
    distances_to_coverslip = distance_to_plane(optimal_params, points_all_rotated)

    plt.hist(distances_to_coverslip, bins=100)
    plt.vlines(0.4, 0, 1000, color='red')


    # Create the meshgrid for the plane
    x_range = np.linspace(np.min(points_all_rotated[:, 0]), np.max(points_all_rotated[:, 0]), 10)
    y_range = np.linspace(np.min(points_all_rotated[:, 1]), np.max(points_all_rotated[:, 1]), 10)
    x_grid, y_grid = np.meshgrid(x_range, y_range)
    z_grid = (-optimal_params[0] * x_grid - optimal_params[1] * y_grid - optimal_params[3]) / optimal_params[2]

    # Calculate the 3D alpha shape
    alpha = alpha_vals[exp_i]
    alpha_shape = alphashape.alphashape(points_all_rotated, alpha)

    # Get the vertices and faces of the Trimesh object
    vertices = alpha_shape.vertices
    faces = alpha_shape.faces

    dist_to_hull = trimesh.proximity.signed_distance(alpha_shape, points_all_rotated)


    filtering = (dist_to_hull > -3) & (distances_to_coverslip > 0.)

    tracks_filtering = []
    dist_to_cs_tracks = []
    dist_to_tracks = []
    for i, t in enumerate(tracks_exp_rotated):
        dist_to = trimesh.proximity.signed_distance(alpha_shape, t)
        dist_to_thresholded = dist_to > -3
        dist_to_cs = distance_to_plane(optimal_params, t)
        dist_to_cs_thresholded = dist_to_cs > 0.05

        dist_to_cs_tracks.append(dist_to_cs)
        dist_to_tracks.append(dist_to)
        
        if np.mean(dist_to_cs_thresholded) > 0.5 and np.mean(dist_to_thresholded) > 0.5:
            tracks_filtering.append(True)
        else:
            tracks_filtering.append(False)

    tracks_filtering = np.array(tracks_filtering)
    dist_to_cs_tracks = np.array(dist_to_cs_tracks, dtype=object)
    dist_to_tracks = np.array(dist_to_tracks, dtype=object)
    tracks_exp_rotated = np.array(tracks_exp_rotated, dtype=object)

    for i in range(len(tracks_exp_rotated)):
        tracks_filtering_all.append(tracks_filtering[i])
        dist_to_cs_tracks_all.append(dist_to_cs_tracks[i])
        dist_to_tracks_all.append(dist_to_tracks[i])
        tracks_exp_rotated_all.append(tracks_exp_rotated[i])

    for i in range(len(dist_to_cs_tracks)):
        d = np.mean(dist_to_cs_tracks[i]>.5)
        if d>0.2:
            spatial_label_all_exp.append(1)
        else:
            spatial_label_all_exp.append(0)

# %%

above_glass = []
on_glass = []
spatial_label = []
print(dist_to_cs_tracks_all)
for i in range(len(dist_to_cs_tracks_all)):
    d = np.mean(dist_to_cs_tracks_all[i]>.5)
    if d>0.2:
        spatial_label.append(1)
        above_glass.append(ensemble_pred[i])
    else:
        spatial_label.append(0)
        on_glass.append(ensemble_pred[i])
spatial_label = np.array(spatial_label)

above_glass_flat = np.hstack(above_glass)
on_glass_flat = np.hstack(on_glass)
print(np.unique(above_glass_flat, return_counts=True)[1]/np.sum(np.unique(above_glass_flat, return_counts=True)[1]))
print(np.unique(on_glass_flat, return_counts=True)[1]/np.sum(np.unique(on_glass_flat, return_counts=True)[1]))

# shift to left
plt.figure(figsize=(12,2))
plt.bar(np.unique(above_glass_flat, return_counts=True)[0], 
        np.unique(above_glass_flat, return_counts=True)[1]/np.sum(np.unique(above_glass_flat, return_counts=True)[1]),
        color='green', label='Dorsal', alpha=1,
        align='edge', width=-0.4
        )
plt.bar(np.unique(on_glass_flat, return_counts=True)[0], 
        np.unique(on_glass_flat, return_counts=True)[1]/np.sum(np.unique(on_glass_flat, return_counts=True)[1]),
        color='steelblue', label='Ventral', alpha=1,
        align='edge', width=0.4)
plt.legend(loc='upper left')
plt.xticks(np.arange(4), ['Normal', 'Directed', 'Confined', 'Subdiffusive'])
plt.ylabel('Time (%track)')

plt.savefig('../deepspt_results/figures/AP2_Ventral_vs_Dorsal_diffusion_barplot.pdf',
            dpi=300, bbox_inches='tight', pad_inches=0.5)   

# %%

fp_datapath = '../_Data/Simulated_diffusion_tracks/'
hmm_filename = 'simulated2D_HMM.json'
dim = 3
dt = 4

# dont use features with length
selected_features = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,
                              19,20,21,23,24,25,27,28,29,30,31,
                              32,33,34,35,36,37,38,39,40,41,42])
print(len(tracks_all), len(ensemble_pred))
FP_all = create_temporalfingerprint(tracks_all, ensemble_pred, fp_datapath, hmm_filename, dim, dt,
                               selected_features=selected_features)


# %%
exp_idx = 1
experiments = np.unique(expname_all)
spatial_label_exp = []
for d in np.array(dist_to_cs_tracks_all)[expname_all==experiments[exp_idx]]:
    d_check = np.mean(d>.5)
    if d_check>0.2:
        spatial_label_exp.append(1)
    else:
        spatial_label_exp.append(0)
spatial_label_exp = np.array(spatial_label_exp)
tracks_exp = np.array(tracks_all)[expname_all==experiments[exp_idx]]
ensemble_pred_exp = np.array(ensemble_pred)[expname_all==experiments[exp_idx]]
spatial_label_exp.shape, tracks_exp.shape, ensemble_pred_exp.shape

# plot in 2D tracks colored by diff type

fig, ax = plt.subplots(1,2,figsize=(8,4))
for idx,t in enumerate(tracks_exp[spatial_label_exp==0]):
    x = t[:,0]
    y = t[:,1]
    z = t[:,2]
    color_dict = {'0':'tab:blue', '1':'red', '2':'green', '3':'Darkorange'}
    c = [colors.to_rgba(color_dict[str(label)]) for label in ensemble_pred_exp[spatial_label_exp==0][idx]]
    lines = [((x0,y0), (x1,y1)) for x0, y0, x1, y1 in zip(x[:-1], y[:-1], x[1:], y[1:])]
    colored_lines = LineCollection(lines, colors=c, linewidths=(1.5,))
    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') 
                for color in color_dict.values()]
    diff_types = ['ND', 'DM', 'CD', 'SD']
    ax[1].add_collection(colored_lines)
    ax[1].autoscale_view()
    ax[1].set_title('Ventral')
    ax[1].set_xlabel('x (\u03BCm)')
    ax[1].set_ylabel('y (\u03BCm)')

for idx,t in enumerate(tracks_exp[spatial_label_exp==1]):
    x = t[:,0]
    y = t[:,1]
    z = t[:,2]
    color_dict = {'0':'tab:blue', '1':'red', '2':'green', '3':'Darkorange'}
    c = [colors.to_rgba(color_dict[str(label)]) for label in ensemble_pred_exp[spatial_label_exp==1][idx]]
    lines = [((x0,y0), (x1,y1)) for x0, y0, x1, y1 in zip(x[:-1], y[:-1], x[1:], y[1:])]
    colored_lines = LineCollection(lines, colors=c, linewidths=(1.5,))
    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') 
                for color in color_dict.values()]
    diff_types = ['ND', 'DM', 'CD', 'SD']
    ax[0].add_collection(colored_lines)
    ax[0].autoscale_view()
    ax[0].set_title('Dorsal')
    ax[0].set_xlabel('x (\u03BCm)')
    ax[0].set_ylabel('y (\u03BCm)')

plt.tight_layout()
plt.savefig('../deepspt_results/figures/AP2_2Dtracks_diffcoloured_exp{}.pdf'.format(exp_idx),
            pad_inches=0.5, bbox_inches='tight')

print(FP_all.shape, len(spatial_label_all_exp), expname_all.shape, len(tracks_all))

pickle.dump(FP_all, open('../deepspt_results/analytics/AP2_FPX.pkl', 'wb'))
pickle.dump(spatial_label_all_exp, open('../deepspt_results/analytics/AP2_FPy.pkl', 'wb'))
pickle.dump(tracks_all, open('../deepspt_results/analytics/AP2_tracks.pkl', 'wb'))
pickle.dump(expname_all, open('../deepspt_results/analytics/AP2_expname.pkl', 'wb'))
# %%
