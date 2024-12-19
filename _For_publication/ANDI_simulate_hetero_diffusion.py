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


warnings.filterwarnings("ignore")

datapath = '../Andi_Challenge_Code/ANDI_challenge_testsets/'
filename = 'task3.txt'


N_models = 20 # number of tracks
note = 'test2' # train test or HypoptVal + integer

stochastic_length = False # if True, T is random between 10 and max_T, else T = max_T
max_T = 200 # maximum track length
models = [0,1,2,3,4] # choose all
min_number_of_segments = 3
max_number_of_segments = 6 # maximum number of segments -1 => change points
dimension = [2,3]
N_save = N_models
load_trajectories = False
path_datasets = datapath+'X3_2D3D_{}_hetero_train'.format(N_save)
diff_to_loc_ratio = 0.5 # ratio of diffusion to localization error

hetero_tracks = []
hetero_labels = []
AD = datasets_theory()
print(AD.avail_models_name)
segment_lengths_all = []
number_of_segments_all = []

"""
tracks are generated as follows (see: ANDI Nat. Comm. paper):
1. choose number_of_segments between 2 and 5
2. choose segment_lengths between 5 and T//number_of_segments
3. make sure that the sum of segment_lengths equals T
4. choose diffusion model for each segment
5. generate xyz for each segment
6. The xyz were then standardized to have a unitary 
  standard deviation sigmaD of the distribution of displacements over unit time
7. Segments are then scaled by a random number sampled from a normal distribution 
   to include the effect of an effective diffusion coefficient 
    D = sigmaD * N(0,1) for each segment, xyz
8. The tracks were then translated by a random number sampled from a normal
    distribution to include the effect of localization error, at a
    diffusion-to-localization ratio of 0.5 or whats given
9. The subtracks were then concatenated to form a single trajectory
10. rinse and repeat
"""

mSL_model0 = []
mSL_model1 = []
mSL_model2 = []
mSL_model3 = []
mSL_model4 = []
for i in range(N_models):
    if stochastic_length:
        T = np.random.randint(10, max_T)
    else:
        T = max_T
    number_of_segments = np.random.randint(min_number_of_segments, max_number_of_segments)
    # divide T into number_of_segments of minimum length 5
    # while error: ValueError: low >= high
    while True:
        try:
            segment_lengths = np.random.randint(5, T//number_of_segments, number_of_segments)
            break
        except ValueError:
            number_of_segments -= 1
    segment_lengths[-1] = T - np.sum(segment_lengths[:-1])
    segment_lengths_all.append(segment_lengths)
    number_of_segments_all.append(number_of_segments)
    assert np.sum(segment_lengths) == T
    assert len(segment_lengths) == number_of_segments
    assert np.min(segment_lengths) >= 5
    # print(sadfs)
    individual_track = []
    individual_labels = []
    x0 = np.zeros(1)
    y0 = np.zeros(1)
    z0 = np.zeros(1)

    models_chosen = []
    for i in range(number_of_segments):
        rc = np.random.choice(models)
        if i > 0:
            while rc == models_chosen[-1]:
                rc = np.random.choice(models)
        models_chosen.append(rc)

    for seglength, model in zip(segment_lengths,models_chosen):
        if model == 0: # ATTM
            exponents = [0.05, 0.99]
        elif model == 1: # CTRW
            exponents = [0.05, .99]
        elif model == 2: # FBM
            exponents = [0.5, 1.9]
        elif model == 3: # LW
            exponents = [1, 2]
        elif model == 4: # SBM
            exponents = [0.05, 2]
        exponent = np.random.uniform(exponents[0], exponents[1])
        x = AD.create_dataset(
                    seglength, 1, exponent, [model], 
                    path=path_datasets,
                    load_trajectories=load_trajectories,
                    save_trajectories=False)
        y = AD.create_dataset(
                    seglength, 1, exponent, [model], 
                    path=path_datasets,
                    load_trajectories=load_trajectories,
                    save_trajectories=False)
        z = AD.create_dataset(
                    seglength, 1, exponent, [model], 
                    path=path_datasets,
                    load_trajectories=load_trajectories,
                    save_trajectories=False)
       
        x = np.array(x[0][2:])+np.random.normal(0,10**-4,len(x[0][2:]))
        y = np.array(y[0][2:])+np.random.normal(0,10**-4,len(y[0][2:]))
        z = np.array(z[0][2:])+np.random.normal(0,10**-4,len(z[0][2:]))

        x = (x-np.mean(x))/np.std(x)
        y = (y-np.mean(y))/np.std(y)
        z = (z-np.mean(z))/np.std(z)
        D_for_segment = np.random.normal(0, 1)
        x = (x-x[0])*D_for_segment
        y = (y-y[0])*D_for_segment
        z = (z-z[0])*D_for_segment
        mSLx = np.mean(np.sqrt((x.T[1:]-x.T[:-1])**2))
        mSLy = np.mean(np.sqrt((y.T[1:]-y.T[:-1])**2))
        mSLz = np.mean(np.sqrt((z.T[1:]-z.T[:-1])**2))
        mSL = np.mean([mSLx, mSLy, mSLz])
        
        if model == 0:
            mSL_model0.append(mSL)
        elif model == 1:
            mSL_model1.append(mSL)
        elif model == 2:
            mSL_model2.append(mSL)
        elif model == 3:
            mSL_model3.append(mSL)
        elif model == 4:
            mSL_model4.append(mSL)

        x = x + x0 + np.random.normal(0, diff_to_loc_ratio*mSL, x.shape)
        y = y + y0 + np.random.normal(0, diff_to_loc_ratio*mSL, y.shape)
        z = z + z0 + np.random.normal(0, diff_to_loc_ratio*mSL, z.shape)

        label = np.repeat(model, seglength)
        segment = np.row_stack([x,y,z]).astype(float).T
        x0 = segment[-1:,0]
        y0 = segment[-1:,1]
        z0 = segment[-1:,2]

        individual_track.append(segment)
        individual_labels.append(label)
    # put every list in individual_track after each other in array
    #AD.create_segmented_dataset()
        AD.create_segmented_dataset
    individual_track = np.vstack(individual_track)
    individual_labels = np.hstack(individual_labels)

    hetero_tracks.append(individual_track)
    hetero_labels.append(individual_labels)

print('Saving to pickle...')
pickle.dump([hetero_tracks, hetero_labels], 
            open(datapath+'ANDI_{}_hetero_2D3Dtracks_N{}_maxlen{}_D2noise{}_maxsegm{}.pkl'.format(note, N_save, max_T, diff_to_loc_ratio, max_number_of_segments), 'wb'))
# %%


# %%
hetero_tracks = np.array(hetero_tracks)
print(hetero_tracks.shape)

fog, ax = plt.subplots(1,3,figsize=(12,3))
ax[0].hist(np.hstack(segment_lengths_all), bins=100)
ax[0].set_title('Segment lengths')
ax[1].hist([np.sum(s) for s in segment_lengths_all], bins=100)
ax[1].set_title('Track lengths')
ax[2].hist(number_of_segments_all, bins=100)
ax[2].set_title('number_of_segments_all')
plt.tight_layout()
plt.show()
plt.close()


plt.figure()
plt.hist(mSL_model0, bins=100, range=(0,10), density=True, alpha=0.5)
plt.hist(mSL_model1, bins=100, range=(0,10), density=True, alpha=0.5)
plt.hist(mSL_model2, bins=100, range=(0,10), density=True, alpha=0.5)
plt.hist(mSL_model3, bins=100, range=(0,10), density=True, alpha=0.5)
plt.hist(mSL_model4, bins=100, range=(0,10), density=True, alpha=0.5)

# plot subplot in a grid

plt.subplots(5,5, figsize=(8,8))
for p in range(25):
    ax = plt.subplot(5,5,p+1)
    i = np.random.randint(len(hetero_tracks))

    sl, cps, val = find_segments(hetero_labels[i])
    frames = np.arange(len(hetero_tracks[i]))

    print(hetero_tracks[i].shape)
    print(hetero_labels[i].shape)
    print(val)
    print(sl)
    color_dict = {0:'r', 1:'b', 2:'g', 3:'y', 4:'k'}
    print(AD.avail_models_name)
    for j in range(len(sl)):
        start = cps[j]
        end = cps[j+1]+1
        ax.plot(hetero_tracks[i][start:end,0],
                hetero_tracks[i][start:end,1], '-',
                color=color_dict[hetero_labels[i][start]],
                lw=1, ms=2)
    ax.set_aspect('equal')
    ax.axis('off')
plt.axis('off')
plt.show()
plt.close()
print(color_dict)


