# %%
import numpy as np
from deepspt_src import *
import matplotlib.pyplot as plt
import random
from torch.utils.data import DataLoader
import pickle
from global_config import globals
import optuna
from optuna.trial import TrialState

#**********************Initiate variables**********************
# global config variables
globals._parse({})

# get consistent result
seed = globals.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
device = globals.device
print(globals.device)

#### define dataset and method to find specific model trained on such
#datasets = ['SimDiff_dim2_ntraces300000_Drandom0.01-1_dt3.3e-02_N5-600_B0.1-2_R7-25_subA0-0.7_superA1.3-2_Q1-16']
datasets = ['SimDiff_dim3_ntraces300000_Drandom0.0001-0.5_dt1.0e+00_N5-600_B0.05-0.25_R5-25_subA0-0.7_superA1.3-2_Q1-16']
methods = ['XYZ_SL_DP']
dim = 3 if 'dim3' in datasets[0] else 2
methods = ['XYZ_SL_DP']
n_trials = 100
# find the model
dir_name = ''
modelpath = 'Unet_results/mlruns/'
modeldir = '36'
use_mlflow = False

# find models
dim = 3 if 'dim3' in datasets[0] else 2

# find the model
if use_mlflow:
    import mlflow
    mlflow.set_tracking_uri('file:'+join(os.getcwd(), join("Unet_results", "mlruns")))
    best_models_sorted = find_models_for(datasets, methods)
else:
    def find_models_for_from_path(path):
        # load from path
        files = sorted(glob(path+'/*/*_UNETmodel.torch', recursive=True))
        return files

    # not sorted tho
    path = '/nfs/datasync4/jacobkh/SPT/mlruns/{}'.format(modeldir)
    best_models_sorted = find_models_for_from_path(path)
    print(best_models_sorted)

# load model
index_model = 0
if use_mlflow:
    model = load_UnetModels(best_models_sorted[index_model], dir=modelpath, device=device, dim=dim)
else:
    model = load_UnetModels_directly(best_models_sorted[index_model], device=device, dim=dim)
    
# Load data 
features = globals.features 
method = "_".join(features,)
print(method,'method')
datapath = '_Data/Simulated_diffusion_tracks/'


if dim==3:
    val_filename_X = '2023711550_hypoptVal_SimDiff_coloc_dim3_ntraces32000_Drandom0.0001-0.5_dim3_dt1.0e+00_N5-600_B0.05-0.25_R5-25_subA0-0.7_superA1.3-2_Q1-16_DMtype_active_X.pkl'
    val_filename_y = '2023711550_hypoptVal_SimDiff_coloc_dim3_Drandom0.0001-0.5_dim3_dt1.0e+00_N5-600_B0.05-0.25_R5-25_subA0-0.7_superA1.3-2_Q1-16_DMtype_active_timeresolved_y.pkl'

if dim==2:
    val_filename_X = '2023711359_hypoptVal_SimDiff_coloc_dim2_ntraces32000_Drandom0.0001-0.5_dim2_dt1.0e+00_N5-600_B0.05-0.25_R5-25_subA0-0.7_superA1.3-2_Q1-16_DMtype_active_X.pkl'
    val_filename_y = '2023711359_hypoptVal_SimDiff_coloc_dim2_Drandom0.0001-0.5_dim2_dt1.0e+00_N5-600_B0.05-0.25_R5-25_subA0-0.7_superA1.3-2_Q1-16_DMtype_active_timeresolved_y.pkl'

X_val = np.array(pickle.load(open(datapath+val_filename_X, 'rb')), dtype=object)
y_val = np.array(pickle.load(open(datapath+val_filename_y, 'rb')), dtype=object)
X_val = prep_tracks(X_val)
X_val = add_features(X_val, features)

min_max_len = 601 # or 2001 dependent on model
batch_size = 1 #globals.batch_size
X_padtoken = globals.X_padtoken
y_padtoken = globals.y_padtoken

def objective(trial):
    # Ensure model is in eval mode
    model.eval()
    calib_model = model.to(device)
    calib_model.eval()

    # reach in to layer before softmax
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    # Dataloaders
    D_val = CustomDataset(X_val, y_val, X_padtoken=X_padtoken, y_padtoken=y_padtoken, device=device)
    val_loader = DataLoader(D_val, batch_size = 64)

    # Temperature being optimized
    temperature = trial.suggest_float("temperature", 1e-5, 10, log=True)

    loss = 0
    count = 0
    with torch.no_grad():
        for idx, xb in enumerate(val_loader):
            x,y = xb
            calib_model.module_list[-1].conv_softmax[1].register_forward_hook(get_activation(str(idx)))
            output = calib_model(xb) 
            criterion = nn.NLLLoss()
            pred = activation[str(idx)]/temperature
            pred = nn.LogSoftmax(dim=0)(pred)
            
            for i in range(len(y)):
                mask_idx = sum(y[i].ge(globals.y_padtoken))
                masked_y = y[i][mask_idx:].unsqueeze(0)
                masked_pred = pred[i][:,mask_idx:].unsqueeze(0)
                loss += criterion(masked_pred, masked_y.long())
                count += 1
    loss /= count
    return loss

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.NopPruner())
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

temperature = trial.params['temperature']
pickle.dump(temperature, open('Unet_results/temperature.pkl', 'wb'))
datapath = 'tracks/'

# # R = 7-25
# filenames_X = ['2023711713_hypopttest_SimDiff_coloc_dim2_ntraces32000_Drandom0.0001-0.5_dim2_dt1.0e+00_N5-600_B0.05-0.25_R5-25_subA0-0.7_superA1.3-2_Q1-16_DMtype_active_X.pkl']
# filenames_y = ['2023711713_hypopttest_SimDiff_coloc_dim2_Drandom0.0001-0.5_dim2_dt1.0e+00_N5-600_B0.05-0.25_R5-25_subA0-0.7_superA1.3-2_Q1-16_DMtype_active_timeresolved_y.pkl']

# # # # R = 7-25 3 D
filenames_X = ['2023711912_hypopttest_SimDiff_coloc_dim3_ntraces32000_Drandom0.0001-0.5_dim3_dt1.0e+00_N5-600_B0.05-0.25_R5-25_subA0-0.7_superA1.3-2_Q1-16_DMtype_active_X.pkl']
filenames_y = ['2023711912_hypopttest_SimDiff_coloc_dim3_Drandom0.0001-0.5_dim3_dt1.0e+00_N5-600_B0.05-0.25_R5-25_subA0-0.7_superA1.3-2_Q1-16_DMtype_active_timeresolved_y.pkl']

# prep X and y
X_to_eval, y_to_eval = load_Xy(datapath, filenames_X[0], filenames_y[0], features=['XYZ','SL', 'DP'])

ensemble_score = temperature_pred(X_to_eval, y_to_eval, model, 
                                  temperature = temperature, 
                                  X_padtoken=X_padtoken, 
                                  y_padtoken=y_padtoken, 
                                  device=device)
number_quantiles = 20
print('temp', temperature)
print(len(ensemble_score), len(y_to_eval))
savename = 'deepspt_results/figures/temp_cali_reliability_mldir'+best_models_sorted[0].split('/')[0]
reliability_plot(ensemble_score, y_to_eval, number_quantiles = 20, savename=savename)

# %%
import pickle
temperature = pickle.load(open('Unet_results/temperature.pkl', 'rb'))
temperature
