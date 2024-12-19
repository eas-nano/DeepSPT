# %%
import os
import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms
import sys
sys.path.append('../')
from deepspt_src import *
from global_config import globals
import random

# global config variables
globals._parse({})

#**********************Initiate variables**********************
# get consistent result
seed = 42 # np.random.randint(0, int(10**6))
print('seed:', seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


# choose dim to run hyp opt for
dim = 2

print('############### DIMENSIONS: ',dim,' ##################')

# saving
features = globals.features 
method = "_".join(features,)
datapath = globals.datapath

# load data
if dim==3:
    train_filename_X = '20211221757_SimDiff_dim3_ntraces100000_Drandom0.01-1_dt5.0e+00_N5-600_B0.1-2_R7-25_subA0-0.7_superA1.3-2_Q1-16_X.pkl'
    train_filename_y = '20211221757_SimDiff_dim3_Drandom0.01-1_dt5.0e+00_N5-600_B0.1-2_R7-25_subA0-0.7_superA1.3-2_Q1-16_timeresolved_y.pkl'
    val_filename_X = '202112201643_hypoptVal_SimDiff_dim3_ntraces20000_Drandom0.01-1_dt5.0e+00_N5-600_B0.1-2_R7-25_subA0-0.7_superA1.3-2_Q1-16_X.pkl'
    val_filename_y = '202112201643_hypoptVal_SimDiff_dim3_Drandom0.01-1_dt5.0e+00_N5-600_B0.1-2_R7-25_subA0-0.7_superA1.3-2_Q1-16_timeresolved_y.pkl'

if dim==2:
    train_filename_X = '20211230545_SimDiff_dim2_ntraces100000_Drandom0.01-1_dt3.3e-02_N5-600_B0.1-2_R7-25_subA0-0.7_superA1.3-2_Q1-16_X.pkl'
    train_filename_y = '20211230545_SimDiff_dim2_Drandom0.01-1_dt3.3e-02_N5-600_B0.1-2_R7-25_subA0-0.7_superA1.3-2_Q1-16_timeresolved_y.pkl'
    val_filename_X = '202112291814_hypopt_SimDiff_dim2_ntraces20000_Drandom0.01-1_dt3.3e-02_N5-600_B0.1-2_R7-25_subA0-0.7_superA1.3-2_Q1-16_X.pkl'
    val_filename_y = '202112291814_hypopt_SimDiff_dim2_Drandom0.01-1_dt3.3e-02_N5-600_B0.1-2_R7-25_subA0-0.7_superA1.3-2_Q1-16_timeresolved_y.pkl'

# training
epochs = 30
n_trials = 20
batch_size = globals.batch_size
shuffle = globals.shuffle

# model variables
device = globals.device
print('device', device)
X_padtoken = globals.X_padtoken # pad token for U-net
y_padtoken = globals.y_padtoken # padtoken for U-net
val_size, test_size = globals.val_size, globals.test_size # validation and test set sizes


#**********************Model and Data**********************
# Load data 
X = np.array(pickle.load(open(datapath+train_filename_X, 'rb')), dtype=object)
y = np.array(pickle.load(open(datapath+train_filename_y, 'rb')), dtype=object)

X_val = np.array(pickle.load(open(datapath+val_filename_X, 'rb')), dtype=object)
y_val = np.array(pickle.load(open(datapath+val_filename_y, 'rb')), dtype=object)


# Add features
X = prep_tracks(X)
X = add_features(X, features)

X_val = prep_tracks(X_val)
X_val = add_features(X_val, features)

n_features = X[0].shape[1] # number of input features
n_classes = len(np.unique(np.hstack(y)))

def define_model(trial):

    init_channels = trial.suggest_int("init_channels", 8, 256, step=2)
    channel_multiplier = trial.suggest_int("channel_multiplier", 2, 4, step=2)
    depth = trial.suggest_int("depth", 1, 4)
    enc_kernel = trial.suggest_int("enc_kernel", 3, 7, step=2)
    outconv_kernel = trial.suggest_int("outconv_kernel", 3, 7, step=2)
    dil_rate = trial.suggest_int("dil_rate", 1, 2)
    pooling = trial.suggest_categorical("pooling", ['avg', 'max'])
    enc_conv_nlayers = trial.suggest_int("enc_conv_nlayers", 1, 4)
    dec_conv_nlayers = trial.suggest_int("dec_conv_nlayers", 1, 4)
    bottom_conv_nlayers = trial.suggest_int("bottom_conv_nlayers", 1, 4)
    out_nlayers = trial.suggest_int("out_nlayers", 1,4)
    batchnorm = trial.suggest_categorical("batchnorm", [True, False])
    batchnormfirst = trial.suggest_categorical("batchnormfirst", [True, False])
    
    #print('init_channels',init_channels,
    #  '\nchannel_multiplier',channel_multiplier,
    #  '\ndepth',depth,       
    #  '\nenc_kernel',enc_kernel,
    #  '\noutconv_kernel',outconv_kernel,
    #  '\ndil_rate',dil_rate,
    #  '\npooling',pooling,
    #  '\nenc_conv_nlayers',enc_conv_nlayers,
    #  '\ndec_conv_nlayers',dec_conv_nlayers,
    #  '\nbottom_conv_nlayers',bottom_conv_nlayers,
    #  '\nout_nlayers',out_nlayers,
    #  '\nbatchnorm', batchnorm,
    #  '\nbatchnormfirst', batchnormfirst)

    pools = [2,2,2,2,2,2,2] # the pooling operation
    model = hypoptUNet(n_features = n_features,
                       init_channels = init_channels, 
                       n_classes = n_classes,
                       depth = depth,
                       enc_kernel = enc_kernel,
                       dec_kernel = enc_kernel,
                       outconv_kernel = outconv_kernel, 
                       dil_rate = dil_rate, 
                       pools = pools, 
                       pooling = pooling,
                       enc_conv_nlayers = enc_conv_nlayers,
                       bottom_conv_nlayers = bottom_conv_nlayers,
                       dec_conv_nlayers = dec_conv_nlayers,
                       out_nlayers = out_nlayers,
                       X_padtoken = X_padtoken, 
                       
                       y_padtoken = y_padtoken,
                       channel_multiplier = channel_multiplier, 
                       batchnorm = batchnorm, 
                       batchnormfirst = batchnormfirst, 
                       device = device)
    return model


def objective(trial):

    # Generate the model.
    model = define_model(trial).to(device)
    print(next(model.parameters()).is_cuda)
    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Get the dataset.
    D = CustomDataset(X, y, X_padtoken=X_padtoken, y_padtoken=y_padtoken, device=device)
    D_train, _, _ = datasplitter(D, val_size, test_size, seed)
    
    D_val = CustomDataset(X_val, y_val, X_padtoken=X_padtoken, y_padtoken=y_padtoken, device=device)
 
    # Dataloaders
    batch_size = trial.suggest_int("batch_size", 32, 256, step=32)
    train_loader = DataLoader(D_train, batch_size = batch_size, shuffle = shuffle)
    val_loader = DataLoader(D_val, batch_size = batch_size)

    #print('batch_size',batch_size,
    #      'optimizer_name', optimizer_name,
    #      'lr', lr)

    # Training of the model.
    best_acc = 0
    for epoch in range(epochs):
        _, _ = train_epoch(model, optimizer, train_loader, device)
        _, val_acc = validate(model, optimizer, val_loader, device)
        trial.report(val_acc, epoch)
        print(val_acc)
        if val_acc > best_acc:
            best_acc = val_acc

    print('last epoch', epoch)
    return best_acc


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.NopPruner())
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
