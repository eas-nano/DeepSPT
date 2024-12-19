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
features = globals.features 
method = "_".join(features,)

# choose dim to run hyp opt for
dim = 2

# load ANDI data for 2D or 3D and prepare here
datapath = '../Andi_Challenge_Code/ANDI_challenge_testsets/'

# loading data
N_save = 400000
N_save_train = N_save
save_dataset = True
path_datasets = datapath+'X3_3D_{}_train'.format(N_save)
_, _, _, _, X3_test2, Y3_test2 = challenge_theory_dataset(
                                                  N = N_save, 
                                                  tasks = 3, 
                                                  dimensions = [2,3], 
                                                  min_T = 200,
                                                  max_T = 200,
                                                  N_save = N_save,
                                                  t_save = N_save,
                                                  save_dataset = save_dataset,
                                                  path_datasets = path_datasets,
                                                  load_dataset = True,)
print('Data is dim: ', dim)
if dim==2:
    X = np.array([np.column_stack([track[:200],track[200:]]).astype(float) for track in X3_test2[1]])
    Y3_2D_ = np.vstack(Y3_test2[1])
    y = make_andi_challenge_y_temporal(Y3_2D_, X)
    
    print(X.shape, y.shape)
    print(type(X), type(y))
    print(X.dtype, y.dtype)
elif dim==3:
    X = np.array([np.column_stack([track[:200],track[200:400], track[400:]]).astype(float) for track in X3_test2[2]])
    Y3_3D_ = np.vstack(Y3_test2[2])
    y = make_andi_challenge_y_temporal(Y3_3D_, X)
    print(X.shape, y.shape)
    print(type(X), type(y))


# loading data
N_save = 20000
N_save_val = N_save
save_dataset = True
path_datasets = datapath+'X3_3D_{}_hypoptVal'.format(N_save)
_, _, _, _, X3_test2, Y3_test2 = challenge_theory_dataset(
                                                  N = N_save, 
                                                  tasks = 3, 
                                                  dimensions = [2,3], 
                                                  min_T = 200,
                                                  max_T = 200,
                                                  N_save = N_save,
                                                  t_save = N_save,
                                                  save_dataset = save_dataset,
                                                  path_datasets = path_datasets,
                                                  load_dataset = True,)
print('Data is dim: ', dim)
if dim==2:
    X_val = np.array([np.column_stack([track[:200],track[200:]]).astype(float) for track in X3_test2[1]])
    Y3_2D_ = np.vstack(Y3_test2[1])
    y_val = make_andi_challenge_y_temporal(Y3_2D_, X_val)
    print(X_val.shape, y_val.shape)
    print(X_val.dtype, y_val.dtype)
elif dim==3:
    X_val = np.array([np.column_stack([track[:200],track[200:400], track[400:]]).astype(float) for track in X3_test2[2]])
    Y3_3D_ = np.vstack(Y3_test2[2])
    y_val = make_andi_challenge_y_temporal(Y3_3D_, X_val)
    print(X_val.shape, y_val.shape)
    print(type(X_val), type(y_val))

print('############### DIMENSIONS: ',dim,' ##################')

epochs = globals.epochs
n_trials = 10
batch_size = globals.batch_size
shuffle = globals.shuffle

# model variables
device = globals.device
print('device', device)
X_padtoken = globals.X_padtoken # pad token for U-net
y_padtoken = globals.y_padtoken # padtoken for U-net
val_size, test_size = globals.val_size, globals.test_size # validation and test set sizes

# Add features
X = prep_tracks(X)
X = add_features(X, features)

X_val = prep_tracks(X_val)
X_val = add_features(X_val, features)

X = np.array(X).astype(float)
y = np.array(y).astype(int)
X_val = np.array(X_val).astype(float)
y_val = np.array(y_val).astype(int)

print('X.shape', X.shape)
print('y.shape', y.shape)
print('X_val.shape', X_val.shape)
print('y_val.shape', y_val.shape)

n_features = X[0].shape[1] # number of input features
n_classes = len(np.unique(np.hstack(y)))
print(n_classes, 'n_classes')

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
    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Get the dataset.
    D = CustomDataset(X, y, X_padtoken=X_padtoken, y_padtoken=y_padtoken, device=device)
    D_train, _, _ = datasplitter(D, val_size, test_size, seed)
    
    D_val = CustomDataset(X_val, y_val, X_padtoken=X_padtoken, y_padtoken=y_padtoken, device=device)
 
    # Dataloaders
    batch_size = trial.suggest_int("batch_size", 32, 128, step=32)
    print(batch_size, 'batch_size')
    train_loader = DataLoader(D_train, batch_size = batch_size, shuffle = shuffle)
    val_loader = DataLoader(D_val, batch_size = batch_size)

    print("Current run:  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    #print('batch_size',batch_size,
    #      'optimizer_name', optimizer_name,
    #      'lr', lr)

    # Training of the model.
    best_acc = 0
    for epoch in range(epochs):
        # print true if model is on cuda
        print('cuda', next(model.parameters()).is_cuda)
        _, train_acc = train_epoch(model, optimizer, train_loader, device)
        _, val_acc = validate(model, optimizer, val_loader, device)
        trial.report(val_acc, epoch)
        print('epoch:', epoch, 'dim', dim, ' val_acc:', val_acc, ' train_acc:', train_acc)
        if val_acc > best_acc:
            print('best_val_acc', val_acc)
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

    # save study as txt
    # make identifier for exact timepoint
    import datetime
    import datetime
    now = datetime.datetime.now()
    now = now.strftime("%YY-%mM-%dD-%Hhr-%Mmin-%Ss")
    now = now.replace('-','_')
    study_name = now+'_ANDI_hypopt_'+'dim'+str(dim)+'_NsaveTrain'+str(N_save_train)+'_NsaveVal'+str(N_save_val)+'_ntrials'+str(n_trials)+'_epochs'+str(epochs)+'_seed'+str(seed)
    study_path = '../DEEPSPT_hypopt/'+study_name
    if not os.path.exists(study_path): 
        os.makedirs(study_path)
    with open(study_path+'_study.txt', 'w') as f:
        f.write("%s: %s\n" % ('Best value', trial.value))
        for key, value in trial.params.items():
            f.write("%s: %s\n" % (key, value))
    f.close()
