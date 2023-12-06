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

class ChangePointLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, 
                 num_layers, maxlens, 
                 bidirectional):
        super(ChangePointLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.maxlens = maxlens

        self.lstm = nn.LSTM(self.input_dim,
                            self.hidden_dim,
                            self.num_layers,
                            batch_first=True,
                            bidirectional=bidirectional)
        self.gru = nn.GRU(self.input_dim,
                            self.hidden_dim,
                            self.num_layers,
                            batch_first=True,
                            bidirectional=bidirectional)
                                    
        self.d = 1 if not self.lstm.bidirectional else 2 
        self.fc = nn.Linear(self.hidden_dim, 2)  # Predicting a single value (the changepoint)

    def forward(self, x):
        #out, (hidden, _) = self.lstm(x)
        out, hidden = self.gru(x) # gru 
        if self.d==2:
            out = out[:,:,:self.hidden_dim] + out[:,:,self.hidden_dim:]
        out = self.fc(out)  # Use the final output
        return out
    
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
frame_change = []
for i in range(len(changing_diffusion_list_all[:n_changing_traces])):
    first = changing_diffusion_list_all[i]
    second = changing_diffusion_list_all[i+n_changing_traces]
    # move second trace to end of first trace and add noise
    second[:,0] += np.random.normal(first[-1,0], 
                    np.sqrt(dim*dt*np.mean(Drandomranges_pairs)))
    second[:,1] += np.random.normal(first[-1,1],
                    np.sqrt(dim*dt*np.mean(Drandomranges_pairs)))
    frame_change.append(len(first))
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

frame_change

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
savename_score = 'deepspt_results/analytics/test2deepspt_ensemble_score.pkl'
savename_pred = 'deepspt_results/analytics/test2deepspt_ensemble_pred.pkl'
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

window_size = 20
selected_features = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,
                            19,20,21,23,24,25,27,28,29,30,31,
                            32,33,34,35,36,37,38,39,40,41,42])

# run fingerprint module of temporally DeepSPT
results2 = Parallel(n_jobs=2)(
        delayed(make_tracks_into_FP_timeseries)(
            track, pred_track, window_size=window_size, selected_features=selected_features,
            fp_datapath=fp_datapath, hmm_filename=hmm_filename, dim=dim, dt=dt)
        for track, pred_track in zip(glued_tracks, ensemble_pred))
timeseries_clean = np.array([r[0] for r in results2])

# %%

# cross-validation
y_groups = np.array(range(len(timeseries_clean))) # no specific groups but can be changed here
gss = GroupKFold(n_splits=5)
gss2 = GroupKFold(n_splits=2)

# split train and test from data_padded into index
train_idx_final = []
test_idx_final = []
val_idx_final = []
direct_idx = np.array(range(len(timeseries_clean)))
for train_index, test_all_index in gss.split(direct_idx, groups=y_groups):
    for test_index, val_index in gss2.split(direct_idx[test_all_index], groups=y_groups[test_all_index]):
        train_idx_final.append(direct_idx[train_index])
        test_idx_final.append(direct_idx[test_all_index][test_index])
        val_idx_final.append(direct_idx[test_all_index][val_index])

# prep data
X_padtoken = -1 # pre-pad tracks to get them equal length, -1 so that it is not confused with 0
length_track = np.array([len(t) for t in timeseries_clean])
maxlens = np.max(length_track)
print('maxlens', maxlens)
data = [torch.from_numpy(t).float() for t in timeseries_clean]
data_padded = [nn.ConstantPad1d((maxlens-len(x), 0), X_padtoken)(x.T).float().T for x in data]
data_padded = torch.stack(data_padded)
print(data_padded.shape, len(data_padded))

# Train the model
torch.manual_seed(0)
num_epochs = 75 # need to be high (>50) for convergence
Fold = 0 # placeholder for cross-validation fold

# Training loop (takes a while if not on gpu)
test_outputs_list = []
test_targets_list = []
test_probs_list = []
train_idx_check = []
test_idx_check = []
val_idx_check = []
X_test_idx_all = []
import datetime
starttime = datetime.datetime.now()
for i in range(len(train_idx_final)):

    model = ChangePointLSTM(input_dim=40, 
                            hidden_dim=10, 
                            num_layers=5, 
                            maxlens=maxlens,
                            bidirectional=True)
    
    X_train_idx = train_idx_final[i]
    X_test_idx = test_idx_final[i]
    X_val_idx = val_idx_final[i]

    print()
    print('Fold', Fold)
    Fold += 1
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_idx_check.append(X_train_idx)
    test_idx_check.append(X_test_idx)
    val_idx_check.append(X_val_idx)

    # split train and test from data_padded into index
    X_train = data_padded[X_train_idx]
    X_val = data_padded[X_val_idx]
    X_test = data_padded[X_test_idx]

    train_length_track = length_track[X_train_idx]
    val_length_track = length_track[X_val_idx]
    test_length_track = length_track[X_test_idx]

    temporal_y = []
    for i,f in enumerate(frame_change):
        offset = maxlens-length_track[i]
        f = int(f+offset)
        ty = np.zeros(maxlens)
        ty[:f] = 0
        ty[f:] = 1
        temporal_y.append(ty)
    temporal_y = np.array(temporal_y)

    y_train = torch.from_numpy(temporal_y)[X_train_idx]
    y_val = torch.from_numpy(temporal_y)[X_val_idx]
    y_test = torch.from_numpy(temporal_y)[X_test_idx]

    from torch.utils.data import TensorDataset, DataLoader
    TrainDataset = TensorDataset(X_train, y_train)
    ValDataset = TensorDataset(X_val, y_val)
    TestDataset = TensorDataset(X_test, y_test)

    # Assume we have some DataLoader objects for the training and validation data
    val_batch_size = 32
    train_loader = DataLoader(TrainDataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(ValDataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(TestDataset, batch_size=32, shuffle=False)

    # Train the model
    best_val_loss = 0
    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = 0
            for i, (o,t) in enumerate(zip(outputs, targets)):
                tl = train_length_track[i]
                loss += criterion(o[maxlens-tl:], t[maxlens-tl:].long())

            loss.backward()
            optimizer.step()

        # Validate the model
        model.eval()
        with torch.no_grad():
            total_val_loss = 0
            total_perc_correct = []
            total_recall = 0
            total_samples = 0
            changepoint_pred = []
            changepoint_true = []
            for vi, (inputs, targets) in enumerate(val_loader):
                targets_pre = targets
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 2)

                total_samples += targets.size(0)
                for i, (p,t) in enumerate(zip(predicted, targets)):
                    lower, upper = int(vi*val_batch_size), int((vi+1)*val_batch_size)
                    vl = val_length_track[lower:upper][i]
                    
                    sgl, cp, v = find_segments(p[maxlens-vl:])
                    changepoint_pred.append(cp[-2])

                    sgl, cp, v = find_segments(t[maxlens-vl:])
                    changepoint_true.append(cp[-2])

                    total_perc_correct.append((p[maxlens-vl:] == t[maxlens-vl:]).sum().item()/len(p[maxlens-vl:]))
                    recall_0 = torch.mean((p[maxlens-vl:][t[maxlens-vl:]==0]==0).float())
                    recall_0 = recall_0 if recall_0>0 else torch.tensor(0, device=device)
                    recall_1 = torch.mean((p[maxlens-vl:][t[maxlens-vl:]==1]==1).float())
                    recall_1 = recall_1 if recall_1>0 else torch.tensor(0, device=device)
                    total_recall += (recall_0+recall_1)/2

                outputs = outputs.view(-1, outputs.shape[-1]).float()  # shape : (batch_size*sequence_length, num_classes)
                targets = targets.view(-1).long() 
                val_loss = criterion(outputs, targets)
                total_val_loss += val_loss.item()

            val_attempt = total_recall/total_samples
            if val_attempt > best_val_loss:
                best_val_loss = val_attempt
                best_model = model
                best_predicted, best_targets_pre = predicted, targets_pre
                torch.save(best_model.state_dict(), 'deepspt_results/analytics/usage_ex2_GRU_CVfold{}.pt'.format(Fold))
                print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {np.round(total_val_loss/len(val_loader),2)}, total_perc_correct: {np.round(np.mean(total_perc_correct),2), np.round(np.std(total_perc_correct, ddof=1),2)}, total_recall/total_samples: {val_attempt.item()}, frame error {np.mean(np.abs(np.array(changepoint_pred)-np.array(changepoint_true)))}')

    for ti, (inputs, targets) in tqdm(enumerate(test_loader)):
        targets_pre = targets
        inputs, targets = inputs.to(device), targets.to(device)
        
        test_outputs = best_model(inputs)
        _, test_predicted = torch.max(test_outputs.data, 2) 
        for i,(tp, tt, to) in enumerate(zip(test_predicted, targets, test_outputs)):
            lower, upper = int(ti*val_batch_size), int((ti+1)*val_batch_size)
            tl = test_length_track[lower:upper][i]
            test_outputs_list.append(tp.cpu().detach().numpy()[maxlens-tl:])
            test_targets_list.append(tt.cpu().detach().numpy()[maxlens-tl:])
            test_probs_list.append(to.cpu().detach().numpy()[maxlens-tl:])
            lower, upper = int(ti*val_batch_size), int((ti+1)*val_batch_size)
            X_test_idx_all.append(X_test_idx[lower:upper][i])
    print(datetime.datetime.now()-starttime, len(X_test))
    print(datetime.datetime.now(),starttime, len(X_test))

acc = [np.mean(test_outputs_list[i]==test_targets_list[i]) for i in range(len(test_outputs))]
pickle.dump(acc, open('deepspt_results/analytics/usage_ex2_testacc.pkl', 'wb'))
pickle.dump(test_outputs_list, open('deepspt_results/analytics/usage_ex2_test_outputs.pkl', 'wb'))
pickle.dump(test_targets_list, open('deepspt_results/analytics/usage_ex2_test_targets.pkl', 'wb'))
pickle.dump(test_probs_list, open('deepspt_results/analytics/usage_ex2_test_probs.pkl', 'wb'))
pickle.dump(X_test_idx_all, open('deepspt_results/analytics/usage_ex2_Xtest_idx_all.pkl', 'wb'))

# %%
import pickle
import sys
sys.path.append('../')
from deepspt_src import find_segments
import matplotlib.pyplot as plt
import numpy as np


acc = pickle.load(open('deepspt_results/analytics/usage_ex2_testacc.pkl', 'rb'))
test_outputs_list = pickle.load(open('deepspt_results/analytics/usage_ex2_test_outputs.pkl', 'rb'))
test_targets_list = pickle.load(open('deepspt_results/analytics/usage_ex2_test_targets.pkl', 'rb'))
test_probs = pickle.load(open('deepspt_results/analytics/usage_ex2_test_probs.pkl', 'rb'))
X_test_idx_all = pickle.load(open('deepspt_results/analytics/usage_ex2_Xtest_idx_all.pkl', 'rb'))

test_changepoint_pred = []
test_changepoint_true = []
for i in range(len(test_outputs_list)):
    sgl, cp, v = find_segments(test_outputs_list[i])
    test_changepoint_pred.append(cp[-2])

    sgl, cp, v = find_segments(test_targets_list[i])
    test_changepoint_true.append(cp[-2])

print('test_changepoint_pred', test_changepoint_pred)
print('test_changepoint_true', test_changepoint_true)

frame_error = np.abs(np.array(test_changepoint_pred)-np.array(test_changepoint_true))
MAE_frame = np.mean(frame_error)
MedianAE_frame = np.median(frame_error)
print('MAE_frame', MAE_frame, 'MedianAE_frame', MedianAE_frame)

plt.figure()
plt.title('True vs predicted changepoint')
plt.scatter(test_changepoint_true, test_changepoint_pred)
plt.xlabel('True changepoint')
plt.ylabel('Predicted changepoint')
plt.show()

plt.figure()
plt.title('Absolute frame error')
plt.hist(frame_error, bins=50, range=(0, np.max(frame_error)))
plt.ylabel('Frequency')
plt.xlabel('Absolute frame error')
plt.show()

plt.figure()
plt.title('Accuracy (percentage correct per track)')
plt.hist(acc, bins=50, range=(0, 1))
plt.ylabel('Frequency')
plt.xlabel('Accuracy')
plt.show()

i = np.random.randint(len(test_outputs_list))
tidx = X_test_idx_all[i]

track_to_plot = glued_tracks[tidx]

fig, ax = plt.subplots(1,2, figsize=(10,5))
acc_i = np.mean(test_outputs_list[i]==test_targets_list[i])
print('ACC {}:'.format(i), acc_i, 'frame_error {} :'.format(i), frame_error[i])
ax[0].set_title('Ground truth track '+str(i))
ax[0].plot(glued_tracks[tidx][:frame_change[tidx]+1,0], 
           glued_tracks[tidx][:frame_change[tidx]+1,1], c='green', lw=2)
ax[0].plot(glued_tracks[tidx][frame_change[tidx]:,0], 
           glued_tracks[tidx][frame_change[tidx]:,1], c='purple', lw=2) 
ax[0].set_xlabel('x')
ax[0].set_ylabel('y')
ax[0].set_aspect('equal')

ax[1].set_title('Prediction, Acc: '+str(np.round(acc_i,2)))
ax[1].plot(glued_tracks[tidx][:test_changepoint_pred[i]+1,0], 
           glued_tracks[tidx][:test_changepoint_pred[i]+1,1], c='green', lw=2)
ax[1].plot(glued_tracks[tidx][test_changepoint_pred[i]:,0], 
           glued_tracks[tidx][test_changepoint_pred[i]:,1], c='purple', lw=2) 
ax[1].set_xlabel('x')
ax[1].set_ylabel('y')
ax[1].set_aspect('equal')

plt.tight_layout()
plt.show()

# %%
