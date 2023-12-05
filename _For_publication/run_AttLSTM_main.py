# %%
from baseline_methods.Att_BiLSTM.config import opt
import os
import torch as t
from baseline_methods.Att_BiLSTM.models.lstmAttNet import LSTMATTNet
from torch.utils.data import DataLoader
import random
import numpy as np
import time
from torch import nn
import pandas as pd
from baseline_methods.Att_BiLSTM.utils.metrics import *
from baseline_methods.Att_BiLSTM.utils.helper_functions import *
from baseline_methods.Att_BiLSTM.data_process import *
import pickle
from global_config import globals


# reset model weight
def weight_reset(self):
    if isinstance(self, nn.LSTM) or isinstance(self, nn.Linear):
        self.reset_parameters()


@t.no_grad() 
def test(test_data):
    opt._parse({})

    # configure model
    in_dim = opt.in_dim
    hidden_dim = opt.hidden_dim
    n_layer = opt.n_layer
    n_class = opt.n_class
    device = opt.device

    attention_win = opt.attention_win
    model = LSTMATTNet(in_dim, hidden_dim, n_layer, n_class, device, attention_win)
   
    # accelerate model
    if opt.use_gpu:
        model.to(opt.device)

    # load test data,from feature extraction folder
    test_dataloader = DataLoader(test_data, opt.batch_size, shuffle=False) 

    # test
    iou_threshold = opt.iou_threshold

    label_num = t.zeros([len(test_data),1])    
    pred_num = t.zeros([len(test_data),1])    
    tracklet_num = t.zeros([len(test_data),1])    
    trackletPred_num = t.zeros([len(test_data),1])    
    precision_class = t.zeros([len(iou_threshold),len(test_data),n_class])

    for ii,(data,label) in enumerate(test_dataloader):
        if opt.use_gpu:
            input = data.to(opt.device)
            label = label.to(opt.device)
        else: 
            input = data
            label = label

	    # input:float32
        input = input.float()
        # label:long
        label = label.view(-1).long()

        score, lstm_out  = model(input)
        _, pred = t.max(score, 1)
        
        # frameAcc
        label_num[ii] = len(label)
        pred_num[ii] = (pred == label).sum()
        # trackletAcc
        tracklet_num[ii],trackletPred_num[ii] = getTrackletPredNum(label,pred)
        # precision
        for jj in range(len(iou_threshold)):
            precision_class[jj,ii,:] = getPrecision(label,pred,n_class,iou_threshold[jj])

    frameAcc = getFrameAcc(label_num,pred_num)
    trackletAcc = getTrackletAcc(tracklet_num,trackletPred_num)
    mAP = t.zeros([1,len(iou_threshold)])
    for jj in range(len(iou_threshold)):
       mAP[0,jj] = getmAP(precision_class[jj])
    print("==> test frameAcc:  {:.2f} %.".format(frameAcc * 100))
    print("==> test trackletAcc:  {:.2f} %.".format(trackletAcc * 100))
    print("==> iou threshold:  {}.".format(iou_threshold))
    np.set_printoptions(precision=2, suppress=True)
    print("==> test mAP:  {} %.".format(mAP.numpy() * 100))

def train(train_dataloader, val_dataloader):
    opt._parse({})

#    import ipdb
#    ipdb.set_trace()

    # start visdom
    #vis = Visualizer(opt.env,port = opt.vis_port)
    
    
    # compute the train time
    if opt.use_gpu:
        t.cuda.synchronize()
    start = time.time()

    # repeat the training process for ksplit times
    ksplit = opt.ksplit
    train_loss_Array=[]
    train_acc_Array=[]
    val_loss_Array=[]
    val_acc_Array=[]
    best_acc_Array=[]
    for kk in range(ksplit):       
        
        for attention_win in opt.attention_win:
            # step1: configure model    
            in_dim = opt.in_dim
            hidden_dim = opt.hidden_dim
            n_layer = opt.n_layer
            n_class = opt.n_class
            device = opt.device

            print("attention_win:{}".format(attention_win))
            model = LSTMATTNet(in_dim, hidden_dim, n_layer, n_class, device, attention_win)
            

            if opt.use_gpu:
                model.to(opt.device)

            # step2: criterion and optimizer
            criterion = t.nn.CrossEntropyLoss()
            lr = opt.lr
            optimizer = model.get_optimizer(lr, opt.weight_decay)
            # dynamically adjust the learning rate
            scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',verbose=1,patience=5,factor=0.5,eps=1e-06)

            # reset parameter
            all_train_loss = []
            all_train_acc = []
            all_val_loss = []
            all_val_acc = []
            best_acc = 0.0
            

            for epoch in range(opt.max_epoch):
               
                # train
                model.train() 
                running_loss = 0
                running_acc = 0
                num_label_train = 0
                for ii,(data,label) in enumerate(train_dataloader):

                    if opt.use_gpu:
                        input = data.to(opt.device)
                        target = label.to(opt.device)
                    else:
                        input = data
                        target = label
                
                    # input:float32,format:(1,56,2)/(256,20,3) batchsize,seq,feature_dim
                    input = input.float()

                    raw_target = target
                    target = target.view(-1).long()

                    #score = model(input)
                    #loss = criterion(score,target)

                    score, lstm_out  = model(input)
                    loss = criterion(score,target)
                
                    running_loss += loss.item() * target.size(0)
                    _, pred = t.max(score, 1)
                    num_correct = (pred == target).sum()
                    running_acc += num_correct.item()
                    num_label_train += len(target)

                    #print('train frame len:{}, total:{}'.format(len(target),num_label_train))
                    # vis.log('train frame len:{}, total:{}'.format(len(target),num_label_train))

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if (ii + 1)%opt.print_freq == 0:
                        print('[{}/{}] time: {} Loss: {:.6f}, Acc: {:.6f}'.format(
                             epoch + 1, opt.max_epoch, ii+1 , running_loss / num_label_train, running_acc / num_label_train))
                    #    vis.log('[{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format(
                    #            epoch + 1, opt.max_epoch, running_loss / num_label_train, running_acc / num_label_train))

                #print('loop ii: {}, finish {} epoch, train_Loss: {:.6f}, train_Acc: {:.6f}'.format(
                #             ii+1, epoch + 1, running_loss / num_label_train, running_acc / num_label_train))
                #vis.log('finish {} epoch, train_Loss: {:.6f}, train_Acc: {:.6f}'.format(
                #            epoch + 1, running_loss / num_label_train, running_acc / num_label_train))

                train_loss = running_loss / num_label_train
                all_train_loss.append(train_loss)
                train_acc = running_acc / num_label_train
                all_train_acc.append(train_acc)
                #vis.plot('train_loss', 'epoch','loss',epoch,train_loss)
                #vis.plot('train_accuracy', 'epoch','accuracy',epoch,train_acc)

                # validate
                model.eval()
                eval_loss = 0
                eval_acc = 0
                num_label_val = 0

                for ii, (val_input, label) in enumerate(val_dataloader):
                    if opt.use_gpu:
                        with t.no_grad():
                            val_input = val_input.to(opt.device)
                            target = label.to(opt.device)
                    else:
                        with t.no_grad():
                            val_input = val_input
                            target = label

                    # val_input:float32,format:(1,56,2) batchsize,seq,feature_dim
                    val_input = val_input.float()

                    # target:long,format:(1,56,1)-->(56) / (256,20,1)-->(5120)
                    #target = target.squeeze(-1).squeeze(-2).long()
                    raw_target = target
                    target = target.view(-1).long()
                    score, lstm_out  = model(val_input)
                    loss = criterion(score,target)

                    eval_loss += loss.item() * target.size(0)
                    _, pred = t.max(score, 1)
                    num_correct = (pred == target).sum()
                    eval_acc += num_correct.item()
                    num_label_val += len(target)
                    #print('val frame len:{}, total:{}'.format(len(target),num_label_val))
                    #vis.log('val frame len:{}, total:{}'.format(len(target),num_label_val))

                val_loss = eval_loss / num_label_val
                all_val_loss.append(val_loss)
                val_acc = eval_acc / num_label_val
                all_val_acc.append(val_acc)
                #vis.plot('val_loss','epoch','loss',epoch,val_loss)
                #vis.plot('val_accuracy','epoch','accuracy',epoch,val_acc)

                # update lr if val_loss increase beyond patience
                scheduler.step(val_loss) 
                lr = optimizer.param_groups[0]['lr']

                print('finish {} epoch, lr: {}, val_Loss: {:.6f}, val_Acc: {:.6f}'.format(
                            epoch + 1, lr, val_loss, val_acc))
                #vis.log('finish {} epoch, lr: {}, val_Loss: {:.6f}, val_Acc: {:.6f}'.format(
                #            epoch + 1, lr, val_loss, val_acc))



                # save model base on val_acc
                if best_acc < val_acc:
                    best_acc = val_acc
                    best_model = model
                    

            print('ksplit:{},Best val Acc: {:4f}'.format(kk+1,best_acc))
            best_acc_Array.append(best_acc)
        
            print('train_loss:{},train_acc:{},val_loss:{},val_acc:{}'.format(all_train_loss,all_train_acc,all_val_loss,all_val_acc))
            train_loss_Array.append(all_train_loss)
            train_acc_Array.append(all_train_acc)
            val_loss_Array.append(all_val_loss)
            val_acc_Array.append(all_val_acc)

    allResult=[]
    allResult.extend(train_loss_Array)
    allResult.extend(train_acc_Array)
    allResult.extend(val_loss_Array)
    allResult.extend(val_acc_Array)
    df = pd.DataFrame(allResult)
 
    #print('best val acc===>')
    #print(best_acc_Array)
    print('attention win:{}, best val acc:{}'.format(attention_win,best_acc_Array))
    
    # compute train time
    if opt.use_gpu:
        t.cuda.synchronize()
    time_elapsed = time.time()-start
    #print(time_elapsed)
    print('==> Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return best_model, best_acc

def predict(input, model):
    import math
    scores = []
    preds = []
    for data in input:
        data = data.reshape(1, data.shape[0], data.shape[1])
        if opt.use_gpu:
            input = data[:,:,:-1].to(opt.device)
            label = data[:,:,-1].to(opt.device)
        else: 
            input = data[:,:,:-1]
            label = data[:,:,-1]

	    # input:float32
        input = input.float()
        # label:long
        label = label.view(-1).long()

        with torch.no_grad():
            score, lstm_out  = model(input)
            _, pred = t.max(score, 1)
            scores.append(score.numpy())
            preds.append(pred.numpy())
        
    return scores, preds

globals._parse({})

# get consistent result
seed = globals.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
print(opt.n_class)
# Load data
import pickle

datapath = '../_Data/Simulated_diffusion_tracks/'
filename_y = '2022424512_SimDiff_dim2_Drandom0.0001-0.5_dt1.0e+00_N5-600_B0.05-0.25_R5-25_subA0-0.7_superA1.3-2_Q1-16_timeresolved_y.pkl'
filename_X = '2022424512_SimDiff_dim2_ntraces300000_Drandom0.0001-0.5_dt1.0e+00_N5-600_B0.05-0.25_R5-25_subA0-0.7_superA1.3-2_Q1-16_X.pkl'

test_X_path = '2022422185_SimDiff_indeptest_dim2_ntraces20000_Drandom0.0001-0.5_dt1.0e+00_N5-600_B0.05-0.25_R5-25_subA0-0.7_superA1.3-2_Q1-16_X.pkl'
test_y_path = '2022422185_SimDiff_indeptest_dim2_Drandom0.0001-0.5_dt1.0e+00_N5-600_B0.05-0.25_R5-25_subA0-0.7_superA1.3-2_Q1-16_timeresolved_y.pkl'

X = np.array(pickle.load(open(datapath+filename_X, 'rb')), dtype=object)
y = np.array(pickle.load(open(datapath+filename_y, 'rb')), dtype=object)

testX = np.array(pickle.load(open(datapath+test_X_path, 'rb')), dtype=object)
testy = np.array(pickle.load(open(datapath+test_y_path, 'rb')), dtype=object)

val_size = 0.2
test_size = 0
seed = globals.seed
    
# split data
D = CustomDataset(X, y)
train_data, val_data, test_data = datasplitter(D, val_size, test_size, seed)
path ='baseline_methods/Att_BiLSTM/'
torch.save(train_data.indices, path+'saved_datasets/'+filename_X+'_D_train_idx.pt')
torch.save(val_data.indices, path+'saved_datasets/'+filename_X+'_D_val_idx.pt')

# prep
processed_train_val, min_state = data_prep([D[i] for i in range(len(D)) if i in train_data.indices+val_data.indices])
processed_train = processed_train_val[:len(train_data.indices)]
processed_val = processed_train_val[len(train_data.indices):]

windowed_train_x, windowed_train_y = getSmallTrainSet(processed_train, min_state, 50, 20)
windowed_val_x, windowed_val_y = getSmallTrainSet(processed_val, min_state, 50, 20)

D_train = CustomDataset(windowed_train_x, windowed_train_y)
D_val = CustomDataset(windowed_val_x, windowed_val_y)

# Run and save
train_dataloader = DataLoader(D_train, opt.batch_size, shuffle=True)
val_dataloader = DataLoader(D_val, opt.batch_size, shuffle=False) 
best_model, best_val_acc = train(train_dataloader, val_dataloader)
torch.save({'best_val_acc': best_val_acc,
            'model_state_dict': best_model.state_dict()
            }, path+'bestmodel_'+filename_X)

# %%
# Evaluate  
Dtest = CustomDataset(testX, testy)
processed_test, _ = data_prep([t for t in Dtest])   
starttime = time.time()
score, pred = predict([torch.tensor(t) for t in processed_test], best_model)
print('time/trace:', (time.time()-starttime)/len(processed_test))

import matplotlib.pyplot as plt

flat_acc = np.mean(np.hstack(pred)==np.hstack(testy))
track_acc = [np.mean(pred[i]==testy[i]) for i in range(len(testy))]
np.median(track_acc), np.mean(track_acc), np.std(track_acc, ddof=1)

plt.hist(track_acc, bins=50, range=(0,1))

# torch.save({'Scores': score,
#             'Preds': pred,
#             'Labels': testy,
#             'track_acc': track_acc,
#             'flat_acc': flat_acc
#             }, path+'predictions_'+test_X_path)

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import seaborn as sns
import pickle
from joblib import Parallel, delayed
import time, os, sys
from tqdm import tqdm
from scipy.optimize import curve_fit
from sklearn.metrics import (roc_auc_score, roc_curve, 
                             precision_recall_curve, 
                             classification_report,
                             confusion_matrix)
import os
import torch

loaded_results = torch.load('baseline_methods/Att_BiLSTM/predictions_2022422185_SimDiff_indeptest_dim2_ntraces20000_Drandom0.0001-0.5_dt1.0e+00_N5-600_B0.05-0.25_R5-25_subA0-0.7_superA1.3-2_Q1-16_X.pkl')
track_acc = loaded_results['track_acc']
acc = loaded_results['flat_acc']

pred_list = loaded_results['Preds']
testy = loaded_results['Labels']

loaded_valresults = torch.load('baseline_methods/Att_BiLSTM/bestmodel_2022424512_SimDiff_dim2_ntraces300000_Drandom0.0001-0.5_dt1.0e+00_N5-600_B0.05-0.25_R5-25_subA0-0.7_superA1.3-2_Q1-16_X.pkl')
loaded_valresults['best_val_acc']

plt.figure()
plt.hist(track_acc)
print(acc)
print(np.median(track_acc))

pred_list_3classes = []
testy_3classes = []
for t,p in zip(testy.copy(), pred_list):
    t = np.array(t)
    t[t==2] = 3
    testy_3classes.append(t)

    p = np.array(p)
    p[p==2] = 3
    pred_list_3classes.append(p)

testy_3classes = np.array(testy_3classes, dtype=object)
pred_list_3classes = np.array(pred_list_3classes, dtype=object)

pred_list_2classes = []
testy_2classes = []
for t,p in zip(testy.copy(), pred_list):
    t = np.array(t)
    t[t==2] = 3
    t[t==1] = 0
    testy_2classes.append(t)

    p = np.array(p)
    p[p==1] = 0
    p[p==2] = 3
    pred_list_2classes.append(p)
testy_2classes = np.array(testy_2classes, dtype=object)
pred_list_2classes = np.array(pred_list_2classes, dtype=object)

acc_3classes = np.mean(np.hstack(testy_3classes)==np.hstack(pred_list_3classes))
track_acc_3classes = [np.mean(testy_3classes[i]==pred_list_3classes[i]) for i in range(len(pred_list_3classes))]
acc_2classes = np.mean(np.hstack(testy_2classes)==np.hstack(pred_list_2classes))
track_acc_2lasses = [np.mean(testy_2classes[i]==pred_list_2classes[i]) for i in range(len(pred_list_2classes))]


plt.figure()
plt.hist(track_acc_3classes)
print(acc_3classes)
print(np.median(track_acc_3classes))

plt.figure()
plt.hist(track_acc_2lasses)
print(acc_2classes)
print(np.median(track_acc_2lasses))

flat_test_true = np.hstack(pred_list)
flat_test_pred = np.hstack(testy)

cf_matrix = confusion_matrix(flat_test_true, flat_test_pred)
group_counts = ["{0:0.0f}K".format(value/1000) for value in
                cf_matrix.flatten()]

cf_matrix = confusion_matrix(flat_test_true, flat_test_pred, normalize='true')         
group_percentages = ["{0:.0%}".format(value) for value in
                     cf_matrix.flatten()]
labels = [f"{v3}\n{v2}" for v2, v3 in zip(group_counts,group_percentages)]
labels = np.asarray(labels).reshape(4,4)

fontsize = 22

plt.figure(figsize=(8,6))
ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues', annot_kws={"fontsize":fontsize})
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)
plt.xlabel('Predicted', size=20)
plt.ylabel('True', size=20)
diffs =  ['ND', 'DM', 'CD', 'SD']
plt.xticks(np.linspace(0.5, 3.5, 4), diffs, rotation=45, size=20)
plt.yticks(np.linspace(0.5, 3.5, 4), diffs, rotation=0, size=20)
flat_acc = np.mean(flat_test_pred == flat_test_true)
from sklearn.metrics import f1_score
f1_ = f1_score(flat_test_true, flat_test_pred, average='macro')
plt.title('N: {}, Accuracy: {:.3f}, F1: {:.3f}'.format(len(flat_test_true), flat_acc, f1_), size=24)
plt.tight_layout()
plt.savefig('../deepspt_results/figures/4class_BiLSTM_confusion_matrix.pdf')
plt.show()
print(classification_report(flat_test_true, flat_test_pred, target_names=diffs))
print('Accuracy:', np.mean(np.array(flat_test_pred)==np.array(flat_test_true)))


flat_test_true = np.hstack(pred_list_3classes)
flat_test_pred = np.hstack(testy_3classes)

cf_matrix = confusion_matrix(flat_test_true, flat_test_pred)
group_counts = ["{0:0.0f}K".format(value/1000) for value in
                cf_matrix.flatten()]

cf_matrix = confusion_matrix(flat_test_true, flat_test_pred, normalize='true')         
group_percentages = ["{0:.0%}".format(value) for value in
                     cf_matrix.flatten()]
labels = [f"{v3}\n{v2}" for v2, v3 in zip(group_counts,group_percentages)]
labels = np.asarray(labels).reshape(3,3)

plt.figure(figsize=(8,6))
ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues', annot_kws={"fontsize":fontsize})
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)
plt.xlabel('Predicted', size=20)
plt.ylabel('True', size=20)
diffs =  ['ND', 'DM', 'Rest.']
plt.xticks(np.linspace(0.5, 2.5, 3), diffs, rotation=45, size=20)
plt.yticks(np.linspace(0.5, 2.5, 3), diffs, rotation=0, size=20)
flat_acc = np.mean(flat_test_pred == flat_test_true)
from sklearn.metrics import f1_score
f1_ = f1_score(flat_test_true, flat_test_pred, average='macro')
plt.title('N: {}, Accuracy: {:.3f}, F1: {:.3f}'.format(len(flat_test_true), flat_acc, f1_), size=24)
plt.tight_layout()
plt.savefig('../deepspt_results/figures/3class_BiLSTM_confusion_matrix.pdf')
plt.show()
print(classification_report(flat_test_true, flat_test_pred, target_names=diffs))
print('Accuracy:', np.mean(np.array(flat_test_pred)==np.array(flat_test_true)))


flat_test_true = np.hstack(testy_2classes)
flat_test_pred = np.hstack(pred_list_2classes)

cf_matrix = confusion_matrix(flat_test_true, flat_test_pred)
group_counts = ["{0:0.0f}K".format(value/1000) for value in
                cf_matrix.flatten()]

cf_matrix = confusion_matrix(flat_test_true, flat_test_pred, normalize='true')         
group_percentages = ["{0:.0%}".format(value) for value in
                     cf_matrix.flatten()]
labels = [f"{v3}\n{v2}" for v2, v3 in zip(group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)

plt.figure(figsize=(8,6))
ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues', annot_kws={"fontsize":fontsize})
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)
plt.xlabel('Predicted', size=20)
plt.ylabel('True', size=20)
diffs =  ['Free', 'Rest.']
plt.xticks(np.linspace(0.5, 1.5, 2), diffs, rotation=45, size=20)
plt.yticks(np.linspace(0.5, 1.5, 2), diffs, rotation=0, size=20)
flat_acc = np.mean(flat_test_pred == flat_test_true)
from sklearn.metrics import f1_score
f1_ = f1_score(flat_test_true, flat_test_pred, average='macro')
plt.title('N: {}, Accuracy: {:.3f}, F1: {:.3f}'.format(len(flat_test_true), flat_acc, f1_), size=24)
plt.tight_layout()
plt.savefig('../deepspt_results/paper/2class_BiLSTM_confusion_matrix.pdf')
plt.show()
print(classification_report(flat_test_true, flat_test_pred, target_names=diffs))
print('Accuracy:', np.mean(np.array(flat_test_pred)==np.array(flat_test_true)))

# %%

