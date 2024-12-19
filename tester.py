# %%
import pickle

file = 'deepspt_results/analytics/frame_change_pruned_DeepSPT.pkl'
f = pickle.load(open(file, 'rb'))


f1 = [t for t in f]
pickle.dump(f1, 
            open('deepspt_results/analytics/frame_change_pruned_DeepSPT_v1.pkl', 'wb'))



file = 'deepspt_results/analytics/AP2_tracks_DeepSPT.pkl'
f = pickle.load(open(file, 'rb'))


f1 = [t for t in f]
pickle.dump(f1, 
            open('deepspt_results/analytics/AP2_tracks_DeepSPT_v1.pkl', 'wb'))


# %%

# load npy file

import numpy as np

path = '/Users/jacobkh/Documents/PhD/SPT/github_final/DeepSPT/_For_publication/baseline_methods/'
file_GT = 'hmm_rota_changepoints_GT.npy'
file_ML = 'hmm_rota_changepoints_ML.npy'

GT = np.load(path+file_GT, allow_pickle=True)
ML = np.load(path+file_ML, allow_pickle=True)

GT, ML

# %%
12.61**2