# %%
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
import seaborn as sns
import matplotlib.pyplot as plt

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

if __name__ == '__main__':

    _use_dual_labelling = True

    if _use_dual_labelling:
        timeseries_clean = pickle.load(open('../deepspt_results/analytics/timeseries_clean_DeepSPT.pkl', 'rb'))
        frame_change_pruned = pickle.load(open('../deepspt_results/analytics/frame_change_pruned_DeepSPT.pkl', 'rb'))
        length_track = pickle.load(open('../deepspt_results/analytics/length_track_DeepSPT.pkl', 'rb'))
        uniq_like_pairs = pickle.load(open('../deepspt_results/analytics/uniq_like_pairs_DeepSPT.pkl', 'rb'))
        
        import networkx as nx
        G = nx.Graph()
        for i in range(len(uniq_like_pairs)):
            G.add_edge(uniq_like_pairs[i][0], 
                    uniq_like_pairs[i][1])
            
        # find connected components
        connected_components = list(nx.connected_components(G))
        connected_components = np.array([list(c) for c in connected_components])
        connected_components, len(connected_components)

        y_groups = np.array(range(len(timeseries_clean)))
        new_idx_list = list(range(np.max(y_groups)+1,np.max(y_groups)+len(connected_components)+1))
        for i, c in enumerate(connected_components):
            new_i = new_idx_list[i]
            y_groups[c] = new_i

        gss = GroupKFold(n_splits=5)
        gss2 = GroupKFold(n_splits=2)

        train_idx_final = []
        test_idx_final = []
        val_idx_final = []
        direct_idx = np.array(range(len(timeseries_clean)))
        for train_index, test_all_index in gss.split(direct_idx, groups=y_groups):
            for test_index, val_index in gss2.split(direct_idx[test_all_index], groups=y_groups[test_all_index]):
                train_idx_final.append(direct_idx[train_index])
                test_idx_final.append(direct_idx[test_all_index][test_index])
                val_idx_final.append(direct_idx[test_all_index][val_index])

        # check if train, test, val idx are unique
        for i in range(len(train_idx_final)):
            print(set(train_idx_final[i]).intersection(set(test_idx_final[i])), 
                    set(train_idx_final[i]).intersection(set(val_idx_final[i])),
                    set(test_idx_final[i]).intersection(set(val_idx_final[i])))

    else:
        timeseries_clean = pickle.load(open('../deepspt_results/analytics/timeseries_clean560nm_DeepSPT.pkl', 'rb'))
        frame_change_pruned = pickle.load(open('../deepspt_results/analytics/frame_change_pruned560nm_DeepSPT.pkl', 'rb'))
        length_track = pickle.load(open('../deepspt_results/analytics/length_track560nm_DeepSPT.pkl', 'rb'))
        y_groups = np.array(range(len(timeseries_clean)))

        gss = GroupKFold(n_splits=5)
        gss2 = GroupKFold(n_splits=2)

        train_idx_final = []
        test_idx_final = []
        val_idx_final = []
        direct_idx = np.array(range(len(timeseries_clean)))
        for train_index, test_all_index in gss.split(direct_idx, groups=y_groups):
            for test_index, val_index in gss2.split(direct_idx[test_all_index], groups=y_groups[test_all_index]):
                train_idx_final.append(direct_idx[train_index])
                test_idx_final.append(direct_idx[test_all_index][test_index])
                val_idx_final.append(direct_idx[test_all_index][val_index])

    X_padtoken = -1
    maxlens = np.max([len(t) for t in timeseries_clean])
    print('maxlens', maxlens)
    print('frame_change_pruned', np.mean(frame_change_pruned))
    data = [torch.from_numpy(t).float() for t in timeseries_clean]
    data_padded = [nn.ConstantPad1d((maxlens-len(x), 0), X_padtoken)(x.T).float().T for x in data]
    data_padded = torch.stack(data_padded)

    print(data_padded.shape, len(data_padded))

    # Train the model

    torch.manual_seed(0)
    X_test_idx_all = []

    from sklearn.model_selection import KFold
    num_epochs = 150
    
    Fold = 0

    val_outputs = []
    val_targets = []
    val_benchmark = []
    val_probs = []
    train_idx_check = []
    test_idx_check = []
    val_idx_check = []
    for i in range(len(train_idx_final)):

        if _use_dual_labelling:
            model = ChangePointLSTM(input_dim=40, 
                                    hidden_dim=25, 
                                    num_layers=5, 
                                    maxlens=maxlens,
                                    bidirectional=True)

        if not _use_dual_labelling:
            model = ChangePointLSTM(input_dim=40, 
                                    hidden_dim=5, 
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

        print(len(val_length_track), len(test_length_track), len(train_length_track))

        temporal_y = []
        for i,f in enumerate(frame_change_pruned):
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
                total_correct = 0
                total_recall = 0
                total_samples = 0
                for inputs, targets in val_loader:
                    targets_pre = targets
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 2)

                    total_samples += targets.size(0)
                    for i, (p,t) in enumerate(zip(predicted, targets)):
                        vl = val_length_track[i]
                        total_correct += (p[maxlens-vl:] == t[maxlens-vl:]).sum().item()
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
                    if _use_dual_labelling:
                        torch.save(best_model.state_dict(), '../deepspt_results/analytics/best_model.pt')
                        torch.save(best_model.state_dict(), '../deepspt_results/analytics/20230823_best_model_GRU_Duallabelled_CV{}.pt'.format(Fold))
                    else:
                        torch.save(best_model.state_dict(), '../deepspt_results/analytics/best_model560nm.pt')
                        #torch.save(best_model.state_dict(), '../deepspt_results/analytics/20230702_best_model_GRU_560nm_CV{}.pt'.format(Fold))
                    # pickle.dump(torch.mean(X_train, dim=0), open('../deepspt_results/analytics/X_train_mean.pkl', 'wb'))
                    # pickle.dump(torch.std(X_train, dim=0)+1, open('../deepspt_results/analytics/X_train_std.pkl', 'wb'))
                    #print(predicted.shape, targets_pre.shape, val_length_track.shape)
                    print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {total_val_loss/len(val_loader)}")

        import datetime
        starttime = datetime.datetime.now()
        for ti, (inputs, targets) in tqdm(enumerate(test_loader)):
            targets_pre = targets
            inputs, targets = inputs.to(device), targets.to(device)
            
            best_model.eval()
            test_outputs = best_model(inputs)
            _, test_predicted = torch.max(test_outputs.data, 2) 
            for i,(tp, tt, to) in enumerate(zip(test_predicted, targets, test_outputs)):
                lower, upper = int(ti*val_batch_size), int((ti+1)*val_batch_size)
                tl = test_length_track[lower:upper][i]
                val_outputs.append(tp.cpu().detach().numpy()[maxlens-tl:])
                val_targets.append(tt.cpu().detach().numpy()[maxlens-tl:])
                val_probs.append(to.cpu().detach().numpy()[maxlens-tl:])
                X_test_idx_all.append(X_test_idx[lower:upper][i])
        print(datetime.datetime.now()-starttime, len(X_test))
        print(datetime.datetime.now(),starttime, len(X_test))


    acc = [np.mean(val_outputs[i]==val_targets[i]) for i in range(len(val_outputs))]
    if _use_dual_labelling:
        pickle.dump(acc, open('../deepspt_results/analytics/acc.pkl', 'wb'))
        print('len(val_outputs)', len(val_outputs))
        pickle.dump(val_outputs, open('../deepspt_results/analytics/val_outputs.pkl', 'wb'))
        pickle.dump(val_targets, open('../deepspt_results/analytics/val_targets.pkl', 'wb'))
        pickle.dump(val_probs, open('../deepspt_results/analytics/val_probs.pkl', 'wb'))
        pickle.dump(X_test_idx_all, open('../deepspt_results/analytics/X_test_idx_all.pkl', 'wb'))
        pickle.dump(train_idx_check, open('../deepspt_results/analytics/train_idx_check.pkl', 'wb'))
        pickle.dump(test_idx_check, open('../deepspt_results/analytics/test_idx_check.pkl', 'wb'))
        pickle.dump(val_idx_check, open('../deepspt_results/analytics/val_idx_check.pkl', 'wb'))
    else:
        pickle.dump(acc, open('../deepspt_results/analytics/acc560nm.pkl', 'wb'))
        print('len(val_outputs)', len(val_outputs))
        pickle.dump(val_outputs, open('../deepspt_results/analytics/val_outputs560nm.pkl', 'wb'))
        pickle.dump(val_targets, open('../deepspt_results/analytics/val_targets560nm.pkl', 'wb'))
        pickle.dump(val_probs, open('../deepspt_results/analytics/val_probs560nm.pkl', 'wb'))
        pickle.dump(X_test_idx_all, open('../deepspt_results/analytics/X_test_idx_all560nm.pkl', 'wb'))
        pickle.dump(train_idx_check, open('../deepspt_results/analytics/train_idx_check560nm.pkl', 'wb'))
        pickle.dump(test_idx_check, open('../deepspt_results/analytics/test_idx_check560nm.pkl', 'wb'))
        pickle.dump(val_idx_check, open('../deepspt_results/analytics/val_idx_check560nm.pkl', 'wb'))

    # %%

    # above code is meant to run on gpu while below is more notebook like
    import matplotlib.pyplot as plt
    import numpy as np
    import pickle
    import sys
    sys.path.append('../')
    from deepspt_src import find_segments

    _use_dual_labelling = True
    features_used = 'all' 

    if _use_dual_labelling:
        try:
            timeseries_clean = pickle.load(open('../deepspt_results/analytics/timeseries_clean_DeepSPT.pkl', 'rb'))
            frame_change_pruned = pickle.load(open('../deepspt_results/analytics/frame_change_pruned_DeepSPT.pkl', 'rb'))
            frame_change_pruned_v2 = np.zeros(len(frame_change_pruned))
            length_track = pickle.load(open('../deepspt_results/analytics/length_track_DeepSPT.pkl', 'rb'))
            uncoating_tracks = pickle.load(open('../deepspt_results/analytics/escape_tracks_all_DeepSPT.pkl', 'rb'))
            VP7uncoating_tracks = pickle.load(open('../deepspt_results/analytics/VP7escape_tracks_all_DeepSPT.pkl', 'rb'))
            train_idx_check = pickle.load(open('../deepspt_results/analytics/train_idx_check.pkl', 'rb'))
            test_idx_check = pickle.load(open('../deepspt_results/analytics/test_idx_check.pkl', 'rb'))
            val_idx_check = pickle.load(open('../deepspt_results/analytics/val_idx_check.pkl', 'rb'))
        except:
            timeseries_clean = pickle.load(open('../deepspt_results/analytics/timeseries_clean.pkl', 'rb'))
            frame_change_pruned = pickle.load(open('../deepspt_results/analytics/frame_change_pruned.pkl', 'rb'))
            frame_change_pruned_v2 = np.zeros(len(frame_change_pruned))
            length_track = pickle.load(open('../deepspt_results/analytics/length_track.pkl', 'rb'))
            uncoating_tracks = pickle.load(open('../deepspt_results/analytics/escape_tracks_all.pkl', 'rb'))
            VP7uncoating_tracks = pickle.load(open('../deepspt_results/analytics/VP7escape_tracks_all.pkl', 'rb'))
            train_idx_check = pickle.load(open('../deepspt_results/analytics/train_idx_check.pkl', 'rb'))
            test_idx_check = pickle.load(open('../deepspt_results/analytics/test_idx_check.pkl', 'rb'))
            val_idx_check = pickle.load(open('../deepspt_results/analytics/val_idx_check.pkl', 'rb'))
    else:
        timeseries_clean = pickle.load(open('../deepspt_results/analytics/timeseries_clean560nm_DeepSPT.pkl', 'rb'))
        frame_change_pruned = pickle.load(open('../deepspt_results/analytics/frame_change_pruned560nm_DeepSPT.pkl', 'rb'))
        frame_change_pruned_v2 = pickle.load(open('../deepspt_results/analytics/frame_change_pruned560nm_v2_DeepSPT.pkl', 'rb'))
        length_track = pickle.load(open('../deepspt_results/analytics/length_track560nm_DeepSPT.pkl', 'rb'))
        uncoating_tracks = pickle.load(open('../deepspt_results/analytics/escape_tracks_all560nm_DeepSPT.pkl', 'rb'))
        train_idx_check = pickle.load(open('../deepspt_results/analytics/train_idx_check560nm.pkl', 'rb'))
        test_idx_check = pickle.load(open('../deepspt_results/analytics/test_idx_check560nm.pkl', 'rb'))
        val_idx_check = pickle.load(open('../deepspt_results/analytics/val_idx_check560nm.pkl', 'rb'))

    if _use_dual_labelling:
        acc = pickle.load(open('../deepspt_results/analytics/acc.pkl', 'rb'))
        val_outputs = pickle.load(open('../deepspt_results/analytics/val_outputs.pkl', 'rb'))
        val_targets = pickle.load(open('../deepspt_results/analytics/val_targets.pkl', 'rb'))
        val_probs = pickle.load(open('../deepspt_results/analytics/val_probs.pkl', 'rb'))
        X_test_idx_all = pickle.load(open('../deepspt_results/analytics/X_test_idx_all.pkl', 'rb'))
    else:
        acc = pickle.load(open('../deepspt_results/analytics/acc560nm.pkl', 'rb'))
        val_outputs = pickle.load(open('../deepspt_results/analytics/val_outputs560nm.pkl', 'rb'))
        val_targets = pickle.load(open('../deepspt_results/analytics/val_targets560nm.pkl', 'rb'))
        val_probs = pickle.load(open('../deepspt_results/analytics/val_probs560nm.pkl', 'rb'))
        X_test_idx_all = pickle.load(open('../deepspt_results/analytics/X_test_idx_all560nm.pkl', 'rb'))
    X_test_idx_all = np.hstack(X_test_idx_all)

    pred_change = []
    true_change = []
    for v,y in zip(val_outputs, val_targets):
        segl,cp,val = find_segments(v)
        if len(val)==2:
            if val[-1]==1:
                pred_change.append(cp[-2])
            else:
                pred_change.append(0)
        elif len(val)==1:
            pred_change.append(0)
        elif len(val)==3:
            pred_change.append(cp[-2])
        else:
            pred_change.append(cp[-2])
        
        segl,cp,val = find_segments(y)
        true_change.append(cp[-2])
    pred_change = np.array(pred_change)
    true_change = np.array(true_change)

    mae_frame = np.abs(true_change-pred_change)
    print('mae frame', np.median(mae_frame), np.mean(mae_frame))
    print('1', len(val_outputs), len(val_targets), len(pred_change), len(true_change))

    print('median abs error', np.median(np.abs(true_change-pred_change)))
    print('mean abs error', np.mean(np.abs(true_change-pred_change)))

    print('mean acc', np.mean(acc))
    print('median acc', np.median(acc))

    tp0 = []
    tp1 = []
    for v,y in zip(val_outputs, val_targets):
        tp0.append(np.nanmean(v[y==0]==0))
        tp1.append(np.nanmean(v[y==1]==1))

    print('mean tp0', np.nanmean(tp0), 'mean tp1', np.nanmean(tp1))
    print('median tp0', np.nanmedian(tp0), 'median tp1', np.nanmedian(tp1))

    # %%

    from sklearn.metrics import confusion_matrix, f1_score
    import seaborn as sns
    
    fig_save_dir = '../deepspt_results/figures/'
    if _use_dual_labelling==False:

        fig, ax = plt.subplots(1,1, figsize=(4,4))
        
        N_test = len(acc)
        xpos = 0.13
        f1_ = f1_score(np.hstack(val_outputs), np.hstack(val_targets), average='macro')
        ax.hist(np.array(acc)*100, bins=50, range=(0,100))
        ax.annotate(f"N: {len(acc):.0f}", (xpos, 0.88), xycoords='axes fraction')
        ax.annotate(f"F1: {f1_*100:.0f}%", (xpos, 0.8), xycoords='axes fraction')
        ax.annotate(f"Mean acc.: {np.mean(acc)*100:.0f}%", (xpos, 0.7), xycoords='axes fraction')
        ax.annotate(f"Median acc.: {np.median(acc)*100:.0f}%", (xpos, 0.6), xycoords='axes fraction')
        ax.annotate(f"Median frame \nerror: {np.median(np.abs(true_change-pred_change)):.1f}",
                    (xpos, 0.2), xycoords='axes fraction')
        ax.set_xlabel('Accuracy (%)')
        ax.set_ylabel('Count')
        ax.set_xlim(-1,101)

        plt.savefig(
        fig_save_dir+'Use_Dual_{}_features_{}_RollingPred_accuracy_v3.pdf'.format(_use_dual_labelling, features_used),
        pad_inches=0.2, bbox_inches='tight')



        val_outputs_flat = np.hstack(val_outputs)
        val_targets_flat = np.hstack(val_targets)

        TP = val_outputs_flat[val_targets_flat==0]==val_targets_flat[val_targets_flat==0]
        TN = val_outputs_flat[val_targets_flat==1]==val_targets_flat[val_targets_flat==1]
        FP = val_outputs_flat[val_targets_flat==0]!=val_targets_flat[val_targets_flat==0]
        FN = val_outputs_flat[val_targets_flat==1]!=val_targets_flat[val_targets_flat==1]

        
        cms = np.array([[np.mean(TP),np.mean(FP)],
                        [np.mean(FN), np.mean(TN)]])
        fontsize = 30
        group_percentages = ["{:0.0f}%".format(mu*100) for mu in cms.flatten()]
        labels = np.asarray(group_percentages).reshape(2,2)

        plt.figure(figsize=(6,5))
        ax = sns.heatmap(cms*100, annot=labels, fmt='', cmap='Blues', annot_kws={"fontsize":fontsize})
        # rm colorbar
        ax.collections[0].colorbar.remove()
        #cbar = ax.collections[0].colorbar
        #cbar.ax.tick_params(labelsize=20)
        plt.xlabel('Predicted', size=20)
        plt.ylabel('True', size=20)
        diffs =  np.array(['Before', 'After'])
        plt.xticks(np.linspace(0.5, 1.5, 2), diffs, rotation=0, size=14)
        plt.yticks(np.linspace(0.5, 1.5, 2), diffs, rotation=0, size=14)

        plt.tight_layout()
        plt.savefig(
            fig_save_dir+'Use_Dual_{}_features_{}_RollingPred_confusion_matrix_v2.pdf'.format(_use_dual_labelling, features_used))

    else:

        fig, ax = plt.subplots(1,1, figsize=(4,4))
        
        N_test = len(acc)
        xpos = 0.13
        f1_ = f1_score(np.hstack(val_outputs), np.hstack(val_targets), average='macro')
        ax.hist(np.array(acc)*100, bins=50, range=(0,100))
        ax.annotate(f"N: {len(acc):.0f}", (xpos, 0.88), xycoords='axes fraction')
        ax.annotate(f"F1: {f1_*100:.0f}%", (xpos, 0.8), xycoords='axes fraction')
        ax.annotate(f"Mean acc.: {np.mean(acc)*100:.0f}%", (xpos, 0.7), xycoords='axes fraction')
        ax.annotate(f"Median acc.: {np.median(acc)*100:.0f}%", (xpos, 0.6), xycoords='axes fraction')
        ax.annotate(f"Median frame \nerror: {np.median(np.abs(true_change-pred_change)):.1f}",
                    (xpos, 0.4), xycoords='axes fraction')
        ax.set_xlabel('Accuracy (%)')
        ax.set_ylabel('Count')
        ax.set_xlim(-1,101)

        plt.savefig(
        fig_save_dir+'Use_Dual_{}_features_{}_RollingPred_accuracy_v3.pdf'.format(_use_dual_labelling, features_used),
        pad_inches=0.2, bbox_inches='tight')


        val_outputs_flat = np.hstack(val_outputs)
        val_targets_flat = np.hstack(val_targets)

        TP = val_outputs_flat[val_targets_flat==0]==val_targets_flat[val_targets_flat==0]
        TN = val_outputs_flat[val_targets_flat==1]==val_targets_flat[val_targets_flat==1]
        FP = val_outputs_flat[val_targets_flat==0]!=val_targets_flat[val_targets_flat==0]
        FN = val_outputs_flat[val_targets_flat==1]!=val_targets_flat[val_targets_flat==1]

        
        cms = np.array([[np.mean(TP),np.mean(FP)],
                        [np.mean(FN), np.mean(TN)]])
        cms_n = np.array([[np.sum(TP), np.sum(FP)],
                          [np.sum(FN), np.sum(TN)]])
        
        print(TP)
        fontsize = 30
        group_percentages = ["{:0.0f}%\n({:0.0f})".format(mu*100,n) for mu,n 
                                in zip(cms.flatten(),cms_n.flatten())]
        labels = np.asarray(group_percentages).reshape(2,2)

        plt.figure(figsize=(6,5))
        ax = sns.heatmap(cms*100, annot=labels, fmt='', cmap='Blues', annot_kws={"fontsize":fontsize})
        # rm colorbar
        ax.collections[0].colorbar.remove()
        #cbar = ax.collections[0].colorbar
        #cbar.ax.tick_params(labelsize=20)
        plt.xlabel('Predicted', size=20)
        plt.ylabel('True', size=20)
        diffs =  np.array(['Before', 'After'])
        plt.xticks(np.linspace(0.5, 1.5, 2), diffs, rotation=0, size=14)
        plt.yticks(np.linspace(0.5, 1.5, 2), diffs, rotation=0, size=14)

        plt.tight_layout()
        plt.savefig(
            fig_save_dir+'Use_Dual_{}_features_{}_RollingPred_confusion_matrix_v2.pdf'.format(_use_dual_labelling, features_used))
    

    # %%

    seen_idx = []

    # %%
    import plotly.graph_objects as go

    if _use_dual_labelling==True:
        idx = np.random.randint(len(val_outputs))
        while idx in seen_idx:
            idx = np.random.randint(len(val_outputs))

        idx = 74
        seen_idx.append(idx)
        # 44, 74, 97, 6
        """
        # 44,2,61, 63, 88, 74, 90, 40, 55, 92  nice
        37, 99 uglys
        """
        print(idx, 'seen:',len(seen_idx), seen_idx) # 0 61 59 3 73, 42
        pred_change_idx = pred_change[idx]
        true_change_idx = true_change[idx]
        print(pred_change_idx, true_change_idx)
        tidx = X_test_idx_all[idx]
        print(tidx)

        uncoat_to_plot = uncoating_tracks[tidx]#-uncoating_tracks[tidx][0]
        vp7_to_plot = VP7uncoating_tracks[tidx]#-VP7uncoating_tracks[tidx][0]

        width = 5
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=uncoat_to_plot[:pred_change_idx+1,0],
            y=uncoat_to_plot[:pred_change_idx+1,1],
            z=uncoat_to_plot[:pred_change_idx+1,2],
            mode='lines',
            name='Pred. pre-uncoating',
            showlegend = False,
            line=dict(width=width,
                    color='cyan'),
            opacity=1
        ))
        fig.add_trace(go.Scatter3d(
            x=uncoat_to_plot[pred_change_idx:,0],
            y=uncoat_to_plot[pred_change_idx:,1],
            z=uncoat_to_plot[pred_change_idx:,2],
            mode='lines',
            name='Pred. post-uncoating',
            showlegend = False,
            line=dict(width=width,
                    color='cyan'),
            opacity=0.75
        ))
        fig.add_trace(go.Scatter3d(
            x=uncoat_to_plot[pred_change_idx:,0],
            y=uncoat_to_plot[pred_change_idx:,1],
            z=uncoat_to_plot[pred_change_idx:,2],
            mode='lines',
            showlegend = False,
            line=dict(width=width,
                    color='navy',
                    dash='dash',
                    ),
            opacity=0.75
        ))
        fig.add_trace(go.Scatter3d(
            x=vp7_to_plot[:,0],
            y=vp7_to_plot[:,1],
            z=vp7_to_plot[:,2],
            mode='lines',
            name='VP7',
            showlegend = False,
            line=dict(width=width,
                    color='magenta',
                    ),
            opacity=0.75
        ))

        if true_change[idx]!=pred_change[idx]:
            fig.add_trace(go.Scatter3d(
                x=uncoat_to_plot[pred_change_idx:pred_change_idx+1,0],
                y=uncoat_to_plot[pred_change_idx:pred_change_idx+1,1],
                z=uncoat_to_plot[pred_change_idx:pred_change_idx+1,2],
                mode='markers',
                name='Predicted event',
                showlegend = False,
                marker=dict(size=5,
                            color='red')
            ))
            fig.add_trace(go.Scatter3d(
                x=uncoat_to_plot[true_change_idx:true_change_idx+1,0],
                y=uncoat_to_plot[true_change_idx:true_change_idx+1,1],
                z=uncoat_to_plot[true_change_idx:true_change_idx+1,2],
                mode='markers',
                name='Labelled event',
                showlegend = False,
                marker=dict(size=5,
                            color='purple')
            ))
        else:
            fig.add_trace(go.Scatter3d(
                x=uncoat_to_plot[true_change_idx:true_change_idx+1,0],
                y=uncoat_to_plot[true_change_idx:true_change_idx+1,1],
                z=uncoat_to_plot[true_change_idx:true_change_idx+1,2],
                mode='markers',
                name='Labelled+predicted event',
                showlegend = False,
                marker=dict(size=8,
                            color='steelblue')),

            )
            fig.add_trace(go.Scatter3d(
                x=uncoat_to_plot[pred_change_idx:pred_change_idx+1,0],
                y=uncoat_to_plot[pred_change_idx:pred_change_idx+1,1],
                z=uncoat_to_plot[pred_change_idx:pred_change_idx+1,2],
                mode='markers',
                name='Labelled+predicted event',
                showlegend = False,
                marker=dict(size=5,
                            color='black')),

            )
        fig.add_trace(go.Scatter3d(
                x=[48.3,48.3],
                y=[25.5,26],
                z=[3.8,3.8],
                mode='lines',
                showlegend = False,
                line=dict(width=10,
                            color='black')),

            )
        xrange = np.max(np.vstack(uncoat_to_plot)[:, 0]) - np.min(np.vstack(uncoat_to_plot)[:, 0])
        yrange = np.max(np.vstack(uncoat_to_plot)[:, 1]) - np.min(np.vstack(uncoat_to_plot)[:, 1])
        zrange = np.max(np.vstack(uncoat_to_plot)[:, 2]) - np.min(np.vstack(uncoat_to_plot)[:, 2])

        max_range = np.array([xrange, yrange, zrange]).max() 

        print(xrange/max_range, yrange/max_range, zrange/max_range, max_range)
        # change aspect ratio per axis
        fig.update_layout(scene_aspectmode='manual',
                        scene_aspectratio=dict(x=xrange/max_range, 
                                                y=yrange/max_range, 
                                                z=zrange/max_range))
        fig.update_layout(scene=dict(xaxis=dict(showbackground=False,
                                            backgroundcolor='white',
                                            showticklabels=True,
                                            showgrid=True,
                                            gridcolor='lightgrey',
                                            zeroline=True,
                                            zerolinecolor='lightgrey'),
                                yaxis=dict(showbackground=False,
                                            backgroundcolor='white',
                                            showticklabels=True,
                                            showgrid=True,
                                            gridcolor='lightgrey',
                                            zeroline=False,
                                            zerolinecolor='lightgrey'),
                                zaxis=dict(showbackground=False,
                                            backgroundcolor='white',
                                            showticklabels=True,
                                            showgrid=True,
                                            gridcolor='lightgrey',
                                            zeroline=True,
                                            zerolinecolor='lightgrey'),
                                            ))
        # align origin to bottom left corner
        # fig.update_layout(scene=dict(xaxis=dict(range=[np.min(uncoat_to_plot[:,0]), np.max(uncoat_to_plot[:,0])]),
        #                              yaxis=dict(range=[np.min(uncoat_to_plot[:,1]), np.max(uncoat_to_plot[:,1])]),
        #                              zaxis=dict(range=[np.min(uncoat_to_plot[:,2])-0.5, np.max(uncoat_to_plot[:,2])])))

        fig.update_layout(margin=dict(l=0.2, r=0, b=0, t=0))
        # increase plot size
        #fig.update_layout(width=1600, height=1000)
        # camera
        fig.update_layout(scene_camera=dict(eye=dict(x=-1, y=0.12, z=.7),
                                            center=dict(x=0, y=0., z=0),
                                            up=dict(x=0, y=0, z=1),))
        fig.write_image('../deepspt_results/figures/3Dplot_pred_{}.pdf'.format(idx))
        fig.show()
        print('../deepspt_results/figures/3Dplot_pred_{}.pdf'.format(idx))
    #%%

    if _use_dual_labelling==True:
        print(idx) 
        pred_change_idx = pred_change[idx]
        true_change_idx = true_change[idx]

        tidx = X_test_idx_all[idx]
        len_uncoat_to_plot = len(uncoating_tracks[tidx])

        h = .012
        plt.figure()
        plt.scatter(true_change_idx, h, label='pred', color='black', 
                    zorder=10, s=70)
        plt.scatter(pred_change_idx, 0, label='pred', color='steelblue', 
                    zorder=10, s=70)

        plt.scatter(true_change_idx, h, label='pred', color='white', 
                    zorder=9, s=95)
        plt.scatter(pred_change_idx, 0, label='pred', color='white', 
                    zorder=9, s=95)


        plt.plot([0,true_change_idx],[h,h], label='pred',
                color='magenta', lw=5)
        plt.plot([0,len_uncoat_to_plot],[0,0], label='pred',
                color='cyan', lw=5)
        plt.plot([pred_change_idx,len_uncoat_to_plot],[0,0], label='pred',
                color='cyan', lw=5)
        plt.plot([pred_change_idx,len_uncoat_to_plot],[0,0], label='pred',
                color='navy', lw=5, linestyle='dotted')
        
        plt.ylim(-.1,.2)

        plt.savefig('../deepspt_results/figures/1D_predplot_{}_features_{}.pdf'.format(idx, features_used)) 

    if _use_dual_labelling==True:
        val_probs = np.array(val_probs)
        val_probs_sm = []
        for v in val_probs:
            val_probs_sm.append(np.exp(v)/np.sum(np.exp(v), axis=1)[:,None])
        val_probs_sm = np.array(val_probs_sm)
        val_probs_sm[74][:5], val_probs[74][:5]

        plt.figure(figsize=(6,.5))
        plt.plot(val_probs_sm[idx][:,0], 'k', lw=3, label='Before')
        plt.plot(val_probs_sm[idx][:,1], 'dimgrey', lw=3, label='After')
        plt.ylim(-.5, 1.5)
        plt.xlim(-5,len(val_probs_sm[idx][:,0])+5)
        plt.savefig('../deepspt_results/figures/1D_probabilityplot_{}_features_{}_v2.pdf'.format(idx, features_used)) 
   
    if _use_dual_labelling==True:
        # softmax valprobs
        val_probs = np.array(val_probs)
        val_probs_sm = []
        for v in val_probs:
            val_probs_sm.append(np.exp(v)/np.sum(np.exp(v), axis=1)[:,None])
        val_probs_sm = np.array(val_probs_sm)

        idx = 74
        plt.figure(figsize=(12,4))
        plt.stackplot(range(len(val_probs_sm[idx])), val_probs_sm[idx][:,0], val_probs_sm[idx][:,1],
                    colors=['lightgrey', 'grey'], alpha=1)
        plt.ylim(-1,2)
        plt.title(idx)
        plt.savefig('../deepspt_results/figures/1D_probabilityplot_{}_features_{}.pdf'.format(idx, features_used)) 

    # %%

    # check if train, test, val idx are unique
    print(len(train_idx_check))
    print(train_idx_check, val_idx_check, test_idx_check)
    for i in range(len(train_idx_check)):
        print(set(train_idx_check[i]).intersection(set(test_idx_check[i])), 
              set(train_idx_check[i]).intersection(set(val_idx_check[i])),
              set(test_idx_check[i]).intersection(set(val_idx_check[i])))

# %%
