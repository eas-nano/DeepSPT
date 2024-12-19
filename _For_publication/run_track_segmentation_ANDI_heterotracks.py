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

# choose dim to train for
dim = 3
trainset_id = 3 # change: 1, 2, 3 for different train sets

modelname = 'DeepSPT_ANDI_heterotracks'
datapath = '../Andi_Challenge_Code/ANDI_challenge_testsets/'

"""

cd /nfs/datasync4/jacobkh/SPT/DEEPSPT_GITHUB
conda activate livedrop
CUDA_VISIBLE_DEVICES=5 python run_track_segmentation_ANDI_heterotracks.py


"""

N_save = 400000
max_T = 600 # maximum track length
max_number_of_segments = 5 # maximum number of segments => 4 change points
diff_to_loc_ratio = 0.5 # ratio of diffusion to localization error
best_model_save_name = timestamper() + '_DEEPSPT_ANDIheterotracks_model.torch'
data_naming = 'ANDI_heterotracks_'+str(N_save)+'_train'+'_dim'+str(dim)+'_model_v6'
filename_X = 'ANDI_{}_hetero_2D3Dtracks_N{}_maxlen{}_D2noise{}_maxsegm{}.pkl'.format('train'+str(trainset_id), N_save, max_T, diff_to_loc_ratio, max_number_of_segments)
filename_X = filename_X
path_datasets = datapath+filename_X
N_save_train = N_save

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


#**********************Model and Data**********************
# load ANDI data for 2D or 3D and prepare here
# everything below is static
# training
epochs = globals.epochs # number of epochs to train for
shuffle = globals.shuffle # shuffle dataloader
device = globals.device # device of model
pools = globals.pools # size of the pooling operation
X_padtoken = globals.X_padtoken # pad token for U-net
y_padtoken = globals.y_padtoken # padtoken for U-net
val_size, test_size = globals.val_size, globals.test_size # validation and test set sizes
print('Device is: ', device)

# loading data
X, y = pickle.load(open(datapath+filename_X, 'rb'))
if dim==2:
    X = [x[:,:2] for x in X]
print('data shapes', len(X), len(y), X[0].shape, y[0].shape)

# Add features
X = prep_tracks(X)
X = add_features(X, features)
n_features = X[0].shape[1] # number of input features
n_classes = len(np.unique(np.hstack(y)))

X = [np.array(a).astype(float) for a in X]
y = [np.array(a).astype(int) for a in y]

for i in range(len(X)):
    assert len(X[i]) == len(y[i])
min_max_length = np.max([len(x) for x in X])

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
    D = CustomDataset(X, y, X_padtoken=X_padtoken, y_padtoken=y_padtoken, 
                      min_max_length=min_max_length, device=device)
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
            print('best_val_acc {}, Saving model...'.format(best_val_acc))
            torch.save({'epoch': epoch,
                'model_state_dict': best_model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'loss': train_loss,
                }, path+'CV_fold'+str(i)+'_'+best_model_save_name)
                    
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

mlflow.log_metric('best_val_acc', best_val_acc)

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
