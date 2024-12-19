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


#**********************Load data**********************
dim = 3

trainset_id = 1 # change: 1, 2, 3 for different train sets, but just 1 for now
modelname = 'DeepSPT_ANDI_heterotracks'
datapath = '../Andi_Challenge_Code/ANDI_challenge_testsets/'

N_save = 400000
max_T = 600 # maximum track length
max_number_of_segments = 5 # maximum number of segments => 4 change points
diff_to_loc_ratio = 0.5 # ratio of diffusion to localization error
data_naming = 'ANDI_heterotracks_'+str(N_save)+'_train'+'_dim'+str(dim)
filename_X = 'ANDI_{}_hetero_2D3Dtracks_N{}_maxlen{}_D2noise{}_maxsegm{}.pkl'.format('train'+str(trainset_id), N_save, max_T, diff_to_loc_ratio, max_number_of_segments)
path_datasets = datapath+filename_X
N_save_train = N_save

save_dataset = False

print('Data is dim: ', dim)
print('Data is from: ', datapath)
print('Data is named: ', data_naming, 'trainset_id', trainset_id)
X, y = pickle.load(open(datapath+filename_X, 'rb'))
print('train data shape:', len(X), len(y), X[0].shape, y[0].shape)

N_save = 20000
data_naming = 'ANDI_heterotracks_'+str(N_save)+'_train'+'_dim'+str(dim)
filename_X = 'ANDI_{}_hetero_2D3Dtracks_N{}_maxlen{}_D2noise{}_maxsegm{}.pkl'.format('HypoptVal', N_save, max_T, diff_to_loc_ratio, max_number_of_segments)
path_datasets = datapath+filename_X
N_save_val = N_save
save_dataset = False

print('Data is dim: ', dim)
print('Data is from: ', datapath)
X_val, y_val = pickle.load(open(datapath+filename_X, 'rb'))
print('train data shape:', len(X_val), len(y_val), X_val[0].shape, y_val[0].shape)

print('############### START HYPOPT, DIMENSIONS: ',dim,' ##################')

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

X = np.array([a.astype(float) for a in X])
y = np.array([a.astype(int) for a in y])
X_val = np.array([a.astype(float) for a in X_val])
y_val = np.array([a.astype(int) for a in y_val])

# %%
print('max length', np.max([len(x) for x in X]))
print('max length', np.max([len(x) for x in X_val]))

for i in range(len(X)):
    assert len(X[i]) == len(y[i])
for i in range(len(X_val)):
    assert len(X_val[i]) == len(y_val[i])

min_max_length = 600

# plt.subplots(5,5, figsize=(8,8))
# for p in range(25):
#     ax = plt.subplot(5,5,p+1)
#     i = np.random.randint(len(X))

#     sl, cps, val = find_segments(y[i])
#     frames = np.arange(len(X[i]))

#     print(X[i].shape)
#     print(y[i])
#     print(val)
#     print(sl)
#     from Andi_Challenge_Code.andi_datasets.datasets_challenge import datasets_theory
#     AD = datasets_theory()
    
#     color_dict = {0:'r', 1:'b', 2:'g', 3:'y', 4:'k'}
#     for j in range(len(sl)):
#         start = cps[j]
#         end = cps[j+1]+1
#         ax.plot(X[i][start:end,0],
#                 X[i][start:end,1], '-',
#                 color=color_dict[y[i][start]],
#                 lw=1, ms=2)
#     ax.set_title(val)
#     ax.set_aspect('equal')
#     ax.axis('off')
# plt.axis('off')
# plt.show()
# plt.close()
# print(color_dict)
# print(AD.avail_models_name)

# for i in range(len(X)):
#     # check nan and inf
#     assert np.sum(np.isnan(X[i])) == 0, i
#     assert np.sum(np.isinf(X[i])) == 0, i
#     assert np.sum(np.isnan(y[i])) == 0, i
#     assert np.sum(np.isinf(y[i])) == 0, i

X[9], y[9]

# %%
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
    lr = trial.suggest_float("lr", 1e-6, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Get the dataset.
    D = CustomDataset(X, y, X_padtoken=X_padtoken, y_padtoken=y_padtoken, device=device,
                      min_max_length=min_max_length)
    D_train, _, _ = datasplitter(D, val_size, test_size, seed)
    
    D_val = CustomDataset(X_val, y_val, X_padtoken=X_padtoken, y_padtoken=y_padtoken, device=device,
                          min_max_length=min_max_length)
 
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
        import time
        start_time = time.time()
        _, train_acc = train_epoch(model, optimizer, train_loader, device)
        _, val_acc = validate(model, optimizer, val_loader, device)

        def check_pred(model, data_loader, ypadtoken=10):
            model.eval()
            preds = []
            with torch.no_grad():
                for xb in data_loader:
                    _, _, pred = model(xb)
                    pred = pred.cpu().numpy()
                    preds.append([p[p!=ypadtoken] for p in pred])
            return preds

        train_pred = check_pred(model, train_loader)
        val_pred = check_pred(model, val_loader)

        print('train_pred', train_pred[0])
        print('train_true', next(iter(train_loader))[1][0][next(iter(train_loader))[1][0]!=10])

        print('val_pred', val_pred[0])
        print('val_true', next(iter(val_loader))[1][0][next(iter(train_loader))[1][0]!=10])

        trial.report(val_acc, epoch)
        print(start_time, ' epoch:', epoch, 'dim', dim, ' val_acc:', val_acc, ' train_acc:', train_acc, 'time/epoch:', time.time()-start_time)
        if val_acc > best_acc:
            print('best_val_acc', val_acc)
            best_acc = val_acc

    print('last epoch', epoch)
    return best_acc

# %%

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
    study_name = now+'_ANDI_heterotracks_hypopt_'+'dim'+str(dim)+'_NsaveTrain'+str(N_save_train)+'_NsaveVal'+str(N_save_val)+'_ntrials'+str(n_trials)+'_epochs'+str(epochs)+'_seed'+str(seed)
    study_path = '../DEEPSPT_hypopt/'+study_name
    if not os.path.exists(study_path): 
        os.makedirs(study_path)
    with open(study_path+'_study.txt', 'w') as f:
        f.write("%s: %s\n" % ('Best value', trial.value))
        for key, value in trial.params.items():
            f.write("%s: %s\n" % (key, value))
    f.close()
