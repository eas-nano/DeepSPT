# %%

from global_config import globals
from torch.utils.data import DataLoader
import sys
sys.path.append('../')
from deepspt_mlflow_utils.convenience_functions import find_experiments_by_tags,\
                                               make_experiment_name_from_tags
from deepspt_src import *
import numpy as np
import datetime
import pickle
import mlflow
import torch
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

# saving
features = globals.features 
method = "_".join(features,)
datapath = ...

# choose dim to train for
dim = 2
N_save = 400000
trainset_id = '1' # change: '', 1, 2, for different train sets
N_save_train = N_save
modelname = 'DeepSPT_ANDI'
datapath = '../Andi_Challenge_Code/ANDI_challenge_testsets/'

"""
# Training data names
'X3_3D_400000_traintask3.txt'
'X3_3D_400000_train1task3.txt', 
'X3_3D_400000_train2task3.txt', 
"""

best_model_save_name = timestamper() + '_DEEPSPT_ANDI_model.torch'
best_model_save_name = best_model_save_name
data_naming = 'ANDI_task3_'+str(N_save)+'_train'+'_dim'+str(dim) + '_model_v2'
data_naming = data_naming
filename_X = 'X3_3D_'+str(N_save)+'_train'+str(trainset_id)
filename_X = filename_X
path_datasets = datapath+filename_X


min_T = 200 # all tracks are 200 in task 3
max_T = 200 # all tracks are 200 in task 3
assert max_T == min_T == 200 # only works for length 200, otherwise need to change the code below
save_dataset = False

print('Data is dim: ', dim)
print('Data is from: ', datapath)
print('Data is named: ', data_naming, 'trainset_id', trainset_id)

# %%

if dim == 2:
    lr = 0.0000356 # learning rate
    batch_size = 64 # batch size
    optim_choice = optim.Adam # optimizer choice

    # model variables
    init_channels = 142 # number of initial channels in model - these will multiply with channel_multiplier during encoding
    channel_multiplier = 2 # channel multiplier size
    dil_rate = 1 # dilation rate of U-net encoder
    depth = 4 # depth of U-net

    kernelsize = 7 # kernel size of encoder and decoder!
    outconv_kernel = 7 # kernel size in output block of model

    pooling = 'max' # pooling type max / avg

    enc_conv_nlayers = 1 # number of layers in encoder block of model
    dec_conv_nlayers = 1 # number of layers in decoder block of model
    bottom_conv_nlayers = 4 # number of layers in bottom block of model
    out_nlayers = 4 # number of layers in output block of model

    batchnorm = False # bool of batchnorm
    batchnormfirst = True # batchnorm before relu

if dim == 3:
    lr = 7.374e-05 # learning rate
    batch_size = 128 # batch size
    optim_choice = optim.Adam # optimizer choice

    # model variables
    init_channels = 144 # number of initial channels in model - these will multiply with channel_multiplier during encoding
    channel_multiplier = 2 # channel multiplier size
    dil_rate = 1 # dilation rate of U-net encoder
    depth = 4 # depth of U-net

    kernelsize = 7 # kernel size of encoder and decoder!
    outconv_kernel = 5 # kernel size in output block of model

    pooling = 'max' # pooling type max / avg

    enc_conv_nlayers = 1 # number of layers in encoder block of model
    dec_conv_nlayers = 1# number of layers in decoder block of model
    bottom_conv_nlayers = 1 # number of layers in bottom block of model
    out_nlayers = 4 # number of layers in output block of model

    batchnorm = False # bool of batchnorm
    batchnormfirst = False # batchnorm before relu

# training
epochs = globals.epochs # number of epochs to train for
shuffle = globals.shuffle # shuffle dataloader
device = globals.device # device of model
pools = globals.pools # size of the pooling operation
X_padtoken = globals.X_padtoken # pad token for U-net
y_padtoken = globals.y_padtoken # padtoken for U-net
val_size, test_size = globals.val_size, globals.test_size # validation and test set sizes

print('Device is: ', device)

#**********************Model and Data**********************
# load ANDI data for 2D or 3D and prepare here

# loading data
_, _, _, _, X3_test2, Y3_test2 = challenge_theory_dataset(
                                                  N = N_save, 
                                                  tasks = 3, # task 3
                                                  dimensions = [2,3], 
                                                  min_T = min_T,
                                                  max_T = max_T,
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


# %%
#********************** Training **********************
# everything below is static
    
# Add features
X = prep_tracks(X)
X = add_features(X, features)
n_features = X[0].shape[1] # number of input features
n_classes = len(np.unique(np.hstack(y)))

X = np.array(X).astype(float)
y = np.array(y).astype(int)

# check nan and inf
for i in range(len(X)):
    assert np.sum(np.isnan(X[i])) == 0
    assert np.sum(np.isinf(X[i])) == 0
    assert np.sum(np.isnan(y[i])) == 0
    assert np.sum(np.isinf(y[i])) == 0

print('max length', np.max([len(x) for x in X]))

model = hypoptUNet(n_features = n_features,
                    init_channels = init_channels,
                    n_classes = n_classes,
                    depth = depth,
                    enc_kernel = kernelsize,
                    dec_kernel = kernelsize,
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

optimizer = optim_choice(model.parameters(), lr=lr)

#**********************Run Experiment**********************
# Auto-saving
print({'DATASET': data_naming, 'METHOD': method, 'MODEL': modelname})
tags = {'DATASET': data_naming, 'METHOD': method, 'MODEL': modelname}
exp = find_experiments_by_tags(tags)
if len(exp) == 0:
    experiment_name = make_experiment_name_from_tags(tags)
    e_id = mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    for t in tags.keys():
        mlflow.tracking.MlflowClient().set_experiment_tag(e_id, t, tags[t])
elif len(exp) == 1:
    mlflow.set_experiment(exp[0].name)
    experiment_id = exp[0].experiment_id
else:
    raise RuntimeError("There should be at most one experiment for a given tag combination!")

mlflow.start_run()
for i, seed in enumerate(globals.seeds):
    # Pytorch prep and Split
    print('Data prep of DL...')
    D = CustomDataset(X, y, X_padtoken=X_padtoken, y_padtoken=y_padtoken, device=device)
    D_train, D_val, D_test = datasplitter(D, val_size, test_size, seed)
    
    # Dataloaders
    train_loader = DataLoader(D_train, batch_size = batch_size, shuffle = shuffle)
    val_loader = DataLoader(D_val, batch_size = batch_size)

    path ='mlruns/'+str(experiment_id)+'/'+str(mlflow.active_run().info.run_id)+'/'
    cv_indices_path = os.path.join(path, 'CV_indices')
    if not os.path.exists(cv_indices_path): 
        os.makedirs(cv_indices_path)
    torch.save(D_train.indices, cv_indices_path+'/CVfold'+str(i)+'_D_train_idx.pt')
    torch.save(D_val.indices, cv_indices_path+'/CVfold'+str(i)+'_D_val_idx.pt')
    torch.save(D_test.indices, cv_indices_path+'/CVfold'+str(i)+'_D_test_idx.pt')

    # check model is on cuda
    torch.autograd.set_detect_anomaly(True)
    model = model.to(device)
    # print model devie
    print('Model is on device: ', next(model.parameters()).device)
    print('Data is on device: ', next(iter(train_loader))[0].device)
    
    print('Starting epochs...')
    best_val_acc = 0
    for epoch in range(1, epochs + 1):
        starttime = datetime.datetime.now()
        
        train_loss, train_acc = train_epoch(model, optimizer, train_loader, device)
        val_loss, val_acc = validate(model, optimizer, val_loader, device)

        improved = best_val_acc < val_acc
        if improved:
            best_model = model
            best_val_acc = val_acc
                    
        mlflow.log_metric('CVfold'+str(i)+'_TRAIN_LOSS', train_loss)
        mlflow.log_metric('CVfold'+str(i)+'_TRAIN_ACC', train_acc)
        mlflow.log_metric('CVfold'+str(i)+'_VAL_LOSS', val_loss)
        mlflow.log_metric('CVfold'+str(i)+'_VAL_ACC', val_acc)

        print('CVfold:', str(i+1)+'/'+str(len(globals.seeds)),
            'Epoch:', epoch, 
            'Train loss:', train_loss, 
            'Train Acc:', train_acc, 
            'Val loss:', val_loss, 
            'Val Acc:', val_acc, 
            'Best Val Acc:', best_val_acc, 
            'time/epoch:', datetime.datetime.now()-starttime)

        if improved:
            best_model.eval()
            test_loader = DataLoader(D_test, batch_size = batch_size)
            masked_pred, masked_score, masked_y = best_model.predict(test_loader)
            testpred_indices_path = os.path.join(path, 'TestPredictions')
            torch.save({'epoch': epoch,
                        'model_state_dict': best_model.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict(),
                        'loss': train_loss,
                        }, path+'CV_fold'+str(i)+'_'+best_model_save_name)
            if not os.path.exists(testpred_indices_path): 
                os.makedirs(testpred_indices_path)
            torch.save({'Scores': masked_score,
                        'Preds': masked_pred,
                        'Labels': masked_y
                        }, testpred_indices_path+'/CVfold'+str(i)+'_TestPredictionsDict.pt')

mlflow.log_param('Data set', data_naming)
mlflow.log_param('Data features', method)
mlflow.log_param('Data path', datapath)
mlflow.log_param('Filename', filename_X)

mlflow.log_metric('Learning rate', lr)
mlflow.log_param('Trained for Epochs', epochs)

mlflow.log_metric('Best Val Acc', best_val_acc)

mlflow.log_metric('Number of input features', n_features)
mlflow.log_metric('Number of classes', n_classes)

mlflow.log_metric('Initial Number of Channels', init_channels)
mlflow.log_metric('Channel multiplier', channel_multiplier)
mlflow.log_metric('Depth of model', depth)
mlflow.log_metric('Dilation rate', dil_rate)

mlflow.log_metric('Number encoder layers', enc_conv_nlayers)
mlflow.log_metric('Number bottom layers', bottom_conv_nlayers)
mlflow.log_metric('Number decoder layers', dec_conv_nlayers)
mlflow.log_metric('Number output layers', out_nlayers)

mlflow.log_param('Batchnorm', batchnorm)
mlflow.log_param('BN before ReLU', batchnormfirst)

mlflow.log_metric('Unet kernel size', kernelsize)
mlflow.log_metric('Output block kernel size', outconv_kernel)

mlflow.log_metric('Val size', val_size)
mlflow.log_metric('Test size', test_size)

mlflow.log_param('Pooling type', pooling)
for i in range(len(pools)):
    mlflow.log_metric('pools'+str(i), pools[i])

mlflow.log_metric('Cross Validation Folds', len(globals.seeds))
for i, seed in enumerate(globals.seeds):
    mlflow.log_metric('Seed'+str(i), seed)
mlflow.log_metric('Validation size', val_size)
mlflow.log_metric('Test size', test_size)
mlflow.log_param('Dataloader shuffle', shuffle)
mlflow.log_metric('Batch size', batch_size)
mlflow.log_param('X padtoken', X_padtoken)
mlflow.log_param('y padtoken', y_padtoken)

mlflow.end_run()
