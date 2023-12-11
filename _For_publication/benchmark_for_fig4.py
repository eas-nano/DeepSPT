# %%
import numpy as np
import pickle 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import pandas as pd
import os
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import random
import joblib



# Define the MLP architecture
class MLP(nn.Module):
    def __init__(self, layers, device='cpu'):
        super(MLP, self).__init__()

        architecture = []
        for i in range(len(layers)-1):
            architecture.append(nn.Linear(layers[i], layers[i+1]))
            if i != len(layers)-2:
                architecture.append(nn.ReLU())
            else:
                architecture.append(nn.Softmax(dim=1))
        self.model = nn.Sequential(*architecture)
        self.to(device)

    def forward(self, x):
        x = self.model(x)
        return x


class EndosomeDataset(Dataset):
    def __init__(self, X, Y, device='cpu'):
        self.X = torch.from_numpy(X).to(device).float()
        self.Y = torch.from_numpy(Y).to(device).long()
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# Define a function to compute the validation accuracy
def validate(model, val_loader, criterion):
    model.eval()
    y_true = []
    y_pred = []
    val_loss = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            val_loss += criterion(outputs, labels).item()/len(predicted)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(predicted.cpu().numpy().tolist())
    val_acc = accuracy_score(y_true, y_pred)
    return val_acc, val_loss


seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random_state = seed

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

exp_type = 'ap2'
exp_type = 'E_vs_N'


# [:,1:2] only use D
# [:,0:1] only use alpha
# [:,0:2] use both
ap2_acc_all = []
ap2_acc_mean = []
ap2_acc_stds = []
ap2_recall1_mean = []
ap2_recall1_stds = []
ap2_recall2_mean = []
ap2_recall2_stds = []

endo_acc_all = []
endo_acc_mean = []
endo_acc_stds = []
endo_recall1_mean = []
endo_recall1_stds = []
endo_recall2_mean = []
endo_recall2_stds = []
for exp_type in ['ap2', 'E_vs_N']: # ['ap2']: # 
    for conditions_to_pred in ['alpha', 'D', 'D+alpha', 'all']: #
        print('conditions_to_pred', conditions_to_pred)
        if exp_type == 'ap2':
            FP_all = np.array(pickle.load(open('../deepspt_results/analytics/AP2_FPX_DeepSPT.pkl', 'rb'))[:,1:2])
            if conditions_to_pred == 'alpha':
                FP_all = np.array(pickle.load(open('../deepspt_results/analytics/AP2_FPX_DeepSPT.pkl', 'rb'))[:,0:1])
            elif conditions_to_pred == 'D':
                FP_all = np.array(pickle.load(open('../deepspt_results/analytics/AP2_FPX_DeepSPT.pkl', 'rb'))[:,1:2])
            elif conditions_to_pred == 'D+alpha':
                FP_all = np.array(pickle.load(open('../deepspt_results/analytics/AP2_FPX_DeepSPT.pkl', 'rb'))[:,0:2])
            else:
                FP_all = np.array(pickle.load(open('../deepspt_results/analytics/AP2_FPX_DeepSPT.pkl', 'rb')))

            y_all = np.array(pickle.load(open('../deepspt_results/analytics/AP2_FPy_DeepSPT.pkl', 'rb')))
            kf = StratifiedKFold(n_splits=10, shuffle=True, 
                                 random_state=random_state)
            l1, l2 = 'Dorsal', 'Ventral'
            conf_threshold = 0.5
            lr = 10**-3
            num_epochs =  10
            batch_size = 32
            end_layers = [2]

        elif exp_type == 'E_vs_N':
            if conditions_to_pred == 'alpha':
                FP_all = np.array(pickle.load(open('../deepspt_results/analytics/EEA1_NPC1_only_FPX_DeepSPT.pkl', 'rb'))[:,0:1])
            elif conditions_to_pred == 'D':
                FP_all = np.array(pickle.load(open('../deepspt_results/analytics/EEA1_NPC1_only_FPX_DeepSPT.pkl', 'rb'))[:,1:2])
            elif conditions_to_pred == 'D+alpha':
                FP_all = np.array(pickle.load(open('../deepspt_results/analytics/EEA1_NPC1_only_FPX_DeepSPT.pkl', 'rb'))[:,0:2])
            else:
                FP_all = np.array(pickle.load(open('../deepspt_results/analytics/EEA1_NPC1_only_FPX.pkl', 'rb')))

            y_all = np.array(pickle.load(open('../deepspt_results/analytics/EEA1_NPC1_only_FPy.pkl', 'rb')))
            kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
            l1, l2 = 'EEA1', 'NPC1'
            conf_threshold = 0.5
            lr = 10**-3
            num_epochs =  10
            batch_size = 32
            end_layers = [2]

        """
        fp_names = np.array(['0Alpha', '1D', '2extra', '3pval', '4Efficiency', '5logEfficiency',
                            '6FractalDim','7Gaussianity', '8Kurtosis', '9MSDratio', 
                            '10Trappedness', '11t0', '12t1', '13t2', '14t3', '15lifetime', 
                            '16length', '17avgSL', '18avgMSD', '19AvgDP', '20corrDP',
                            '21signDP', '22sumSL', '23minSL', '24maxSL',
                            '25BroadnessSL', '26Speed', '27CoV', '28FractionSlow', 
                            '29FractionFast', '30Volume', '31perc_ND', '32perc_DM', 
                            '33perc_CD', '34perc_SD', '35num_changepoints', '36inst_msd_D',
                            '37meanSequence', '38medianSequence', '39maxSequence', 
                            '40minSequence', '41stdSequence', '42simSeq'])
        """


        #X = np.column_stack([FP_all.copy()[:,:], tTDP_individuals[:,:].copy()])
        X = FP_all.copy()
        X[np.isnan(X)] = 0
        y = y_all.copy()


        print('X.shape', X.shape, 'y.shape', y.shape)

        # Define your model
        layers = [X.shape[1]]+end_layers    

        # Define your loss function and optimizer
        from torch import optim
        import torch.nn as nn
        import torch
        criterion = nn.CrossEntropyLoss()       

        # CV

        pred_TP = []
        pred_FP = []
        pred_TN = []
        pred_FN = []
        val_loss_all = []
        train_loss_all = []
        val_acc_all = []
        train_acc_all = []
        N_test = []
        y_pred_all = []
        y_test_all = []

        eea1_recall_list = []
        npc1_recall_list = []
        eea1_recall_w_coloc_list = []
        npc1_recall_w_coloc_list = []

        idx_w_coloc_in_split_list_all = []
        idx_w_coloc_in_split_list_sure = []
        idx_w_coloc_in_split_list_unsure = []

        test_acc_all = []
        for X_train_idx, X_test_idx in kf.split(X, y):
            model = MLP(layers, device=device)
            optimizer = optim.Adam(model.parameters(), lr=lr)

            # make X_valid_idx x % of X_train_idx shuffled
            X_valid_idx = np.random.choice(X_train_idx, int(len(X_train_idx)*0.2), replace=False)
            X_train_idx = np.setdiff1d(X_train_idx, X_valid_idx)

            X_train, y_train = X[X_train_idx], y[X_train_idx]
            X_valid, y_valid = X[X_valid_idx], y[X_valid_idx]
            X_test, y_test = X[X_test_idx], y[X_test_idx]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_valid = scaler.transform(X_valid)
            X_test = scaler.transform(X_test)

            from imblearn.over_sampling import RandomOverSampler
            ros = RandomOverSampler(random_state=random_state)
            X_train, y_train = ros.fit_resample(X_train, y_train)
        
            # Create your training and validation datasets
            train_dataset = EndosomeDataset(X_train, y_train, device=device)
            val_dataset = EndosomeDataset(X_valid, y_valid, device=device)
            test_dataset = EndosomeDataset(X_test, y_test, device=device)

            # Create your training and validation data loaders
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            # Train your model

            train_loss_list = []
            val_loss_list = []
            train_acc_list = []
            val_acc_list = []
            best_val_acc = 0
            for epoch in range(num_epochs):
                model.train()
                for i, (inputs, labels) in enumerate(train_loader):
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                train_acc, train_loss = validate(model, train_loader, criterion)
                val_acc, val_loss = validate(model, val_loader, criterion)
                train_loss_list.append(train_loss)
                val_loss_list.append(val_loss)
                train_acc_list.append(train_acc)
                val_acc_list.append(val_acc)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model = model    
                    print(f'Epoch {epoch+1}/{num_epochs}: Train Loss = {train_loss:.4f}, Val loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}')
            if epoch%10==0:
                print(f'Epoch {epoch+1}/{num_epochs}: Train Loss = {train_loss:.4f}, Val loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}')
            train_loss_all.append(train_loss_list)
            val_loss_all.append(val_loss_list)
            train_acc_all.append(train_acc_list)
            val_acc_all.append(val_acc_list)

            # Test your model
            best_model.eval()

            y_true = []
            y_pred = []
            with torch.no_grad():
                for inputs, labels in test_loader:
                    outputs = model(inputs)
                    proba, predicted = torch.max(outputs.data, 1)
                    predicted[proba<conf_threshold] = -1
                    y_true.extend(labels.cpu().numpy().tolist())
                    y_pred.extend(predicted.cpu().numpy().tolist())
            y_true = np.hstack(y_true).astype(int)
            y_pred = np.hstack(y_pred).astype(int)
            test_acc = np.mean(y_true[y_pred!=-1]==y_pred[y_pred!=-1])
            test_acc_all.append(test_acc)
            cm = confusion_matrix(y_true[y_pred!=-1], y_pred[y_pred!=-1], labels=[0,1], normalize='true')
            pred_TP.append(cm[0,0])
            pred_FP.append(cm[0,1])
            pred_TN.append(cm[1,1])
            pred_FN.append(cm[1,0])
            N_test.append(len(y_true[y_pred!=-1]))

            y_pred_all.append(y_pred[y_pred!=-1])
            y_test_all.append(y_test[y_pred!=-1])

            print(f'Test Acc = {test_acc:.4f}')

        print('conditions_to_pred', conditions_to_pred)
        print('exp_type', exp_type)
        print('end_layers', end_layers)
        print('lr', lr)
        print('num_epochs', num_epochs)
        print('batch_size', batch_size)
        print('conf_threshold', conf_threshold)
        print(f'Test Acc = {np.mean(test_acc_all)*100:.4f}+-{np.std(test_acc_all, ddof=1)*100:.4f}')
        print(f'TP1 = {np.mean(pred_TP)*100:.4f}+-{np.std(pred_TP, ddof=1)*100:.4f}')
        print(f'TP2 = {np.mean(pred_TN)*100:.4f}+-{np.std(pred_TN, ddof=1)*100:.4f}')

        if exp_type == 'ap2':
            ap2_acc_mean.append(np.mean(test_acc_all)*100)
            ap2_acc_all.append(test_acc_all)
            ap2_acc_stds.append(np.std(test_acc_all, ddof=1)*100)
            ap2_recall1_mean.append(np.mean(pred_TP)*100)
            ap2_recall1_stds.append(np.std(pred_TP, ddof=1)*100)
            ap2_recall2_mean.append(np.mean(pred_TN)*100)
            ap2_recall2_stds.append(np.std(pred_TN, ddof=1)*100)
        if exp_type == 'E_vs_N':
            endo_acc_mean.append(np.mean(test_acc_all)*100)
            endo_acc_all.append(test_acc_all)
            endo_acc_stds.append(np.std(test_acc_all, ddof=1)*100)
            endo_recall1_mean.append(np.mean(pred_TP)*100)
            endo_recall1_stds.append(np.std(pred_TP, ddof=1)*100)
            endo_recall2_mean.append(np.mean(pred_TN)*100)
            endo_recall2_stds.append(np.std(pred_TN, ddof=1)*100)
        
        cms = np.array([[np.mean(pred_TP),np.mean(pred_FP)],
                        [np.mean(pred_FN), np.mean(pred_TN)]])
        cms_stds = np.array([[np.std(pred_TP, ddof=1),np.std(pred_FP, ddof=1)],
                            [np.std(pred_FN, ddof=1), np.std(pred_TN, ddof=1)]])

        fontsize = 22
        group_counts = ["{}".format(value) for value in N_test]

        group_percentages = ["{:0.2f}\u00B1{:0.2f}%".format(mu*100,sig*100) for mu,sig in
                        zip(cms.flatten(), cms_stds.flatten())]

        labels = np.asarray(group_percentages).reshape(2,2)

        plt.figure(figsize=(6,5))
        ax = sns.heatmap(cms*100, annot=labels, fmt='', cmap='Blues', annot_kws={"fontsize":fontsize},
                         vmin=0, vmax=100)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=20)
        plt.xlabel('Predicted', size=20)
        plt.ylabel('True', size=20)
        diffs =  np.array([l1,l2])
        plt.xticks(np.linspace(0.5, 1.5, 2), diffs, rotation=0, size=14)
        plt.yticks(np.linspace(0.5, 1.5, 2), diffs, rotation=45, size=14)
        flat_acc = np.mean(np.hstack(y_pred_all)[np.hstack(y_pred_all)!=-1] == np.hstack(y_test_all)[np.hstack(y_pred_all)!=-1])
        f1_ = f1_score(np.hstack(y_pred_all)[np.hstack(y_pred_all)!=-1], np.hstack(y_test_all)[np.hstack(y_pred_all)!=-1], average='macro')
        plt.title(
            'N: {} tracks, conf. threshold{} \nAccuracy: {:.2f} {:.2f}, F1-score: {:.3f}'.format(
                np.sum(N_test), conf_threshold, np.mean(np.array(test_acc_all)*100), np.std(np.array(test_acc_all)*100, ddof=1), f1_), size=16)
        plt.tight_layout()
        plt.savefig('../deepspt_results/figures/{}_{}_confthres{}_confusion_matrix.pdf'.format(exp_type, conditions_to_pred, conf_threshold),
                    pad_inches=0.2, bbox_inches='tight')
        plt.show()
        plt.close()
        # print(classification_report(np.hstack(y_test_all)[np.hstack(y_pred_all)!=-1], 
        #                             np.hstack(y_pred_all)[np.hstack(y_pred_all)!=-1], 
        #                             target_names=diffs))

import pickle
pickle.dump(ap2_acc_mean, open('../deepspt_results/analytics/ap2_acc_mean.pkl', 'wb'))
pickle.dump(ap2_acc_all, open('../deepspt_results/analytics/ap2_acc_all.pkl', 'wb'))
pickle.dump(ap2_acc_stds, open('../deepspt_results/analytics/ap2_acc_stds.pkl', 'wb'))
pickle.dump(ap2_recall1_mean, open('../deepspt_results/analytics/ap2_recall1_mean.pkl', 'wb'))
pickle.dump(ap2_recall1_stds, open('../deepspt_results/analytics/ap2_recall1_stds.pkl', 'wb'))
pickle.dump(ap2_recall2_mean, open('../deepspt_results/analytics/ap2_recall2_mean.pkl', 'wb'))
pickle.dump(ap2_recall2_stds, open('../deepspt_results/analytics/ap2_recall2_stds.pkl', 'wb'))

pickle.dump(endo_acc_mean, open('../deepspt_results/analytics/endo_acc_mean.pkl', 'wb'))
pickle.dump(endo_acc_all, open('../deepspt_results/analytics/endo_acc_all.pkl', 'wb'))
pickle.dump(endo_acc_stds, open('../deepspt_results/analytics/endo_acc_stds.pkl', 'wb'))
pickle.dump(endo_recall1_mean, open('../deepspt_results/analytics/endo_recall1_mean.pkl', 'wb'))
pickle.dump(endo_recall1_stds, open('../deepspt_results/analytics/endo_recall1_stds.pkl', 'wb'))
pickle.dump(endo_recall2_mean, open('../deepspt_results/analytics/endo_recall2_mean.pkl', 'wb'))
pickle.dump(endo_recall2_stds, open('../deepspt_results/analytics/endo_recall2_stds.pkl', 'wb'))

# %%
import pickle
import numpy as np
import matplotlib.pyplot as plt

ap2_acc_all = pickle.load(open('../deepspt_results/analytics/ap2_acc_all.pkl', 'rb'))
ap2_acc_mean = pickle.load(open('../deepspt_results/analytics/ap2_acc_mean.pkl', 'rb'))
ap2_acc_stds = pickle.load(open('../deepspt_results/analytics/ap2_acc_stds.pkl', 'rb'))
ap2_recall1_mean = pickle.load(open('../deepspt_results/analytics/ap2_recall1_mean.pkl', 'rb'))
ap2_recall1_stds = pickle.load(open('../deepspt_results/analytics/ap2_recall1_stds.pkl', 'rb'))
ap2_recall2_mean = pickle.load(open('../deepspt_results/analytics/ap2_recall2_mean.pkl', 'rb'))
ap2_recall2_stds = pickle.load(open('../deepspt_results/analytics/ap2_recall2_stds.pkl', 'rb'))

endo_acc_all = pickle.load(open('../deepspt_results/analytics/endo_acc_all.pkl', 'rb'))
endo_acc_mean = pickle.load(open('../deepspt_results/analytics/endo_acc_mean.pkl', 'rb'))
endo_acc_stds = pickle.load(open('../deepspt_results/analytics/endo_acc_stds.pkl', 'rb'))
endo_recall1_mean = pickle.load(open('../deepspt_results/analytics/endo_recall1_mean.pkl', 'rb'))
endo_recall1_stds = pickle.load(open('../deepspt_results/analytics/endo_recall1_stds.pkl', 'rb'))
endo_recall2_mean = pickle.load(open('../deepspt_results/analytics/endo_recall2_mean.pkl', 'rb'))
endo_recall2_stds = pickle.load(open('../deepspt_results/analytics/endo_recall2_stds.pkl', 'rb'))

conditions_to_pred_list = ['Alpha', 'D', 'D & alpha', 'DeepsPT']

print(len(endo_acc_all), len(endo_acc_all[0]))

# calculate welsch t-test for accuracy
from scipy.stats import ttest_ind
import sys
sys.path.append('../')
from deepspt_src import sci_notation

df = pd.DataFrame()

acc_means_1 = []
acc_means_2 = []
acc_std_1 = []
acc_std_2 = []
pvals_all = []
tvals_all = []
dof_all = []
row_names = []
n1_all = []
n2_all = []

print(len(ap2_acc_all))

# calculate welsh t-test degree of freedom
for exp_type in ['ap2', 'E_vs_N']:
    for j in range(3):
        if exp_type == 'E_vs_N':
            acc_lists_1 = endo_acc_all[3]
            acc_lists_2 = endo_acc_all[j]

            t, p = ttest_ind(acc_lists_1, acc_lists_2)
            rowname = 'EEA1 vs NPC1: DeepSPT (1) vs {} (2)'.format(conditions_to_pred_list[j])
            acc_means_1.append(np.round(100*np.mean(acc_lists_1),5))
            acc_means_2.append(np.round(100*np.mean(acc_lists_2),5))
            acc_std_1.append(np.round(100*np.std(acc_lists_1, ddof=1),5))
            acc_std_2.append(np.round(100*np.std(acc_lists_2, ddof=1),5))

            n1 = len(acc_lists_1)
            n2 = len(acc_lists_2)
            vn1 = np.var(acc_lists_1) / n1
            vn2 = np.var(acc_lists_2) / n2

        elif exp_type == 'ap2':
            acc_lists_1 = ap2_acc_all[3]
            acc_lists_2 = ap2_acc_all[j]

            t, p = ttest_ind(acc_lists_1, acc_lists_2)
            rowname = 'Dorsal vs Ventral: DeepSPT (1) vs {} (2)'.format(conditions_to_pred_list[j])
            acc_means_1.append(np.round(100*np.mean(acc_lists_1),5))
            acc_means_2.append(np.round(100*np.mean(acc_lists_2),5))
            acc_std_1.append(np.round(100*np.std(acc_lists_1, ddof=1),5))
            acc_std_2.append(np.round(100*np.std(acc_lists_2, ddof=1),5))

            n1 = len(acc_lists_1)
            n2 = len(acc_lists_2)
            vn1 = np.var(acc_lists_1) / n1
            vn2 = np.var(acc_lists_2) / n2

        # Welch–Satterthwaite equation for dof
        with np.errstate(divide='ignore', invalid='ignore'):
            dof = (vn1 + vn2)**2 / (vn1**2 / (n1 - 1) + vn2**2 / (n2 - 1))

        pvals_all.append(p)
        tvals_all.append(np.round(t,5))
        dof_all.append(np.round(dof,5))
        n1_all.append(n1)
        n2_all.append(n2)
        row_names.append(rowname)

df['p-values'] = [sci_notation(p, 10) for p in pvals_all]
df['Test statistics'] = tvals_all
df['Degrees of freedom'] = dof_all
df['\u03BC (1) (%)'] = acc_means_1
df['\u03C3 (1) (%)'] = acc_std_1
df['N (1)'] = n1_all
df['\u03BC (2) (%)'] = acc_means_2
df['\u03C3 (2) (%)'] = acc_std_2
df['N (2)'] = n2_all
df.index = row_names
print(df)
df.to_csv('../deepspt_results/analytics/benchmarkfig4_ttest.csv')


print(ap2_acc_mean, ap2_acc_all)

fig, ax = plt.subplots(2,1,figsize=(5,5))


ax[1].bar(list(range(len(conditions_to_pred_list))), ap2_acc_mean, 
          color='dimgrey', alpha=0.75)

ax[1].errorbar(list(range(len(conditions_to_pred_list))), 
               ap2_acc_mean, yerr=ap2_acc_stds, fmt='.',
               label='Dorsal Ventral', color='k', capsize=3,
               ecolor='k', elinewidth=1, markeredgewidth=1)

ants = np.repeat(list(range(len(conditions_to_pred_list))),len(ap2_acc_all[0]))
ants_spread = ants + np.random.uniform(-0.1,0.1,len(ants))
ax[1].scatter(ants_spread, 100*np.array(ap2_acc_all), 
              alpha=0.5, color='k', s=7)

#ax[0].errorbar(list(range(len(conditions_to_pred_list))), ap2_recall1_mean, yerr=ap2_recall1_stds, fmt='.')
#ax[0].errorbar(list(range(len(conditions_to_pred_list))), ap2_recall2_mean, yerr=ap2_recall2_stds, fmt='.')
ax[1].legend(fontsize=14, handletextpad=0.1)
ax[1].set_ylim(0,100)
ax[1].set_yticks([0,50,100], size=14)
ax[1].set_yticklabels([0,50,100], size=14)

ax[0].bar(list(range(len(conditions_to_pred_list))), endo_acc_mean, 
          color='dimgrey', alpha=0.75)

ax[0].errorbar(list(range(len(conditions_to_pred_list))), 
               endo_acc_mean, yerr=endo_acc_stds, fmt='.',
               label='EEA1 NPC1', color='k', capsize=3,
               ecolor='k', elinewidth=1, markeredgewidth=1)

ants = np.repeat(list(range(len(conditions_to_pred_list))),len(endo_acc_all[0]))
ants_spread = ants + np.random.uniform(-0.1,0.1,len(ants))
ax[0].scatter(ants_spread, 100*np.array(endo_acc_all), 
              alpha=0.5, color='k', s=7)
#ax[0].errorbar(list(range(len(conditions_to_pred_list))), endo_recall1_mean, yerr=endo_recall1_stds, fmt='.')
#ax[0].errorbar(list(range(len(conditions_to_pred_list))), endo_recall2_mean, yerr=endo_recall2_stds, fmt='.')
ax[0].legend(fontsize=14, loc='upper left', handletextpad=0.1)
ax[0].set_ylim(0,100)
ax[0].set_yticks([0,50,100], size=14)
ax[0].set_yticklabels([0,50,100], size=14)
# shared y label
fig.text(-0.02, 0.6, 'Accuracy (%)', va='center', rotation='vertical', size=16)

# remove ticks ax[0]
ax[0].set_xticks([])
ax[1].set_xticks(list(range(len(conditions_to_pred_list))), conditions_to_pred_list, rotation=45, size=14)

plt.tight_layout()
plt.savefig('../deepspt_results/figures/accuracy_benchmarkfig4j.pdf', bbox_inches='tight',
            pad_inches=0.2)

# %%
print(conditions_to_pred_list)
print(ap2_acc_mean, '\n',ap2_acc_stds)
print(ap2_recall1_mean, '\n', ap2_recall1_stds)
print(ap2_recall2_mean, '\n', ap2_recall2_stds)
print()
print(conditions_to_pred_list)
print(endo_acc_mean, '\n',endo_acc_stds)
print(endo_recall1_mean, '\n',endo_recall1_stds)
print(endo_recall2_mean, '\n',endo_recall2_stds)

# %%
conditions_to_pred = 'all'

results_dict = {}
results_dict_uncertaintracks = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# for exp_type in ['ap2', 'E_vs_N']:
for conditions_to_pred in ['alpha', 'D', 'D+alpha', 'all']:
    for exp_type in ['E_vs_N', 'ap2']: # ['E_vs_N', 'ap2']:
        N_all = []
        acc_all = []
        recall1_all = []
        recall2_all = []

        proba_all = []
        pred_all = []
        X_test_idx_order = []
        for conf_threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
            print('conditions_to_pred', conditions_to_pred)
            if exp_type == 'ap2':
                FP_all = np.array(pickle.load(open('../deepspt_results/analytics/AP2_FPX_DeepSPT.pkl', 'rb'))[:,1:2])
                if conditions_to_pred == 'alpha':
                    FP_all = np.array(pickle.load(open('../deepspt_results/analytics/AP2_FPX_DeepSPT.pkl', 'rb'))[:,0:1])
                elif conditions_to_pred == 'D':
                    FP_all = np.array(pickle.load(open('../deepspt_results/analytics/AP2_FPX_DeepSPT.pkl', 'rb'))[:,1:2])
                elif conditions_to_pred == 'D+alpha':
                    FP_all = np.array(pickle.load(open('../deepspt_results/analytics/AP2_FPX_DeepSPT.pkl', 'rb'))[:,0:2])
                else:
                    FP_all = np.array(pickle.load(open('../deepspt_results/analytics/AP2_FPX_DeepSPT.pkl', 'rb')))

                y_all = np.array(pickle.load(open('../deepspt_results/analytics/AP2_FPy_DeepSPT.pkl', 'rb')))
                kf = KFold(n_splits=10, shuffle=True, random_state=random_state)
                l1, l2 = 'Dorsal', 'Ventral'
                lr = 10**-3
                num_epochs =  10
                batch_size = 32
                end_layers = [2]


            elif exp_type == 'E_vs_N':
                if conditions_to_pred == 'alpha':
                    FP_all = np.array(pickle.load(open('../deepspt_results/analytics/EEA1_NPC1_only_FPX_DeepSPT.pkl', 'rb'))[:,0:1])
                elif conditions_to_pred == 'D':
                    FP_all = np.array(pickle.load(open('../deepspt_results/analytics/EEA1_NPC1_only_FPX_DeepSPT.pkl', 'rb'))[:,1:2])
                elif conditions_to_pred == 'D+alpha':
                    FP_all = np.array(pickle.load(open('../deepspt_results/analytics/EEA1_NPC1_only_FPX_DeepSPT.pkl', 'rb'))[:,0:2])
                else:
                    FP_all = np.array(pickle.load(open('../deepspt_results/analytics/EEA1_NPC1_only_FPX_DeepSPT.pkl', 'rb')))

                y_all = np.array(pickle.load(open('../deepspt_results/analytics/EEA1_NPC1_only_FPy_DeepSPT.pkl', 'rb')))
                kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
                l1, l2 = 'EEA1', 'NPC1'
                lr = 10**-3
                num_epochs =  10
                batch_size = 32
                end_layers = [2]       

            """
            fp_names = np.array(['0Alpha', '1D', '2extra', '3pval', '4Efficiency', '5logEfficiency',
                                '6FractalDim','7Gaussianity', '8Kurtosis', '9MSDratio', 
                                '10Trappedness', '11t0', '12t1', '13t2', '14t3', '15lifetime', 
                                '16length', '17avgSL', '18avgMSD', '19AvgDP', '20corrDP',
                                '21signDP', '22sumSL', '23minSL', '24maxSL',
                                '25BroadnessSL', '26Speed', '27CoV', '28FractionSlow', 
                                '29FractionFast', '30Volume', '31perc_ND', '32perc_DM', 
                                '33perc_CD', '34perc_SD', '35num_changepoints', '36inst_msd_D',
                                '37meanSequence', '38medianSequence', '39maxSequence', 
                                '40minSequence', '41stdSequence', '42simSeq'])
            """

            #X = np.column_stack([FP_all.copy()[:,:], tTDP_individuals[:,:].copy()])
            X = FP_all.copy()
            X[np.isnan(X)] = 0
            y = y_all.copy()

            print('X.shape', X.shape, 'y.shape', y.shape)
            # Define your model
            layers = [X.shape[1]] + end_layers
            print('layers', layers)

            # Define your loss function and optimizer
            from torch import optim
            import torch.nn as nn
            import torch
            criterion = nn.CrossEntropyLoss()
            
            pred_TP = []
            pred_FP = []
            pred_TN = []
            pred_FN = []
            val_loss_all = []
            train_loss_all = []
            val_acc_all = []
            train_acc_all = []
            N_test = []
            y_pred_all = []
            y_test_all = []

            eea1_recall_list = []
            npc1_recall_list = []
            eea1_recall_w_coloc_list = []
            npc1_recall_w_coloc_list = []

            idx_w_coloc_in_split_list_all = []
            idx_w_coloc_in_split_list_sure = []
            idx_w_coloc_in_split_list_unsure = []

            test_acc_all = []
            for X_train_idx, X_test_idx in kf.split(X, y):

                model = MLP(layers, device=device)
                optimizer = optim.Adam(model.parameters(), lr=lr)


                # make X_valid_idx x % of X_train_idx shuffled
                X_valid_idx = np.random.choice(X_train_idx, int(len(X_train_idx)*0.2), replace=False)
                X_train_idx = np.setdiff1d(X_train_idx, X_valid_idx)

                X_train, y_train = X[X_train_idx], y[X_train_idx]
                X_valid, y_valid = X[X_valid_idx], y[X_valid_idx]
                X_test, y_test = X[X_test_idx], y[X_test_idx]

                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_valid = scaler.transform(X_valid)
                X_test = scaler.transform(X_test)

                from imblearn.over_sampling import RandomOverSampler
                ros = RandomOverSampler(random_state=random_state)
                X_train, y_train = ros.fit_resample(X_train, y_train)
                
                # Create your training and validation datasets
                train_dataset = EndosomeDataset(X_train, y_train, device=device)
                val_dataset = EndosomeDataset(X_valid, y_valid, device=device)
                test_dataset = EndosomeDataset(X_test, y_test, device=device)

                # Create your training and validation data loaders
                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

                # Train your model

                train_loss_list = []
                val_loss_list = []
                train_acc_list = []
                val_acc_list = []
                best_val_acc = 0
                for epoch in range(num_epochs):
                    model.train()
                    for i, (inputs, labels) in enumerate(train_loader):
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()

                    train_acc, train_loss = validate(model, train_loader, criterion)
                    val_acc, val_loss = validate(model, val_loader, criterion)
                    train_loss_list.append(train_loss)
                    val_loss_list.append(val_loss)
                    train_acc_list.append(train_acc)
                    val_acc_list.append(val_acc)

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_model = model
                
                    if epoch+1%2==0:
                        print(f'Epoch {epoch+1}/{num_epochs}: Train Loss = {train_loss:.4f}, Val loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}')
                
                train_loss_all.append(train_loss_list)
                val_loss_all.append(val_loss_list)
                train_acc_all.append(train_acc_list)
                val_acc_all.append(val_acc_list)

                # Test your model
                best_model.eval()

                y_true = []
                y_pred = []
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        outputs = model(inputs)
                        proba, predicted = torch.max(outputs.data, 1)
                        predicted_pre = predicted.clone()
                        predicted[proba<conf_threshold] = -1
                        y_true.extend(labels.cpu().numpy().tolist())
                        y_pred.extend(predicted.cpu().numpy().tolist())
                        if conf_threshold==0.5:
                            proba_all.append(proba.cpu().numpy())
                            pred_all.append(predicted_pre.cpu().numpy())
                if conf_threshold==0.5:
                    X_test_idx_order.append(X_test_idx)           
                y_true = np.hstack(y_true).astype(int)
                y_pred = np.hstack(y_pred).astype(int)
                test_acc = np.nanmean(y_true[y_pred!=-1]==y_pred[y_pred!=-1])
                test_acc_all.append(test_acc)
                cm = confusion_matrix(y_true[y_pred!=-1], y_pred[y_pred!=-1], labels=[0,1], normalize='true')
                pred_TP.append(cm[0,0])
                pred_FP.append(cm[0,1])
                pred_TN.append(cm[1,1])
                pred_FN.append(cm[1,0])
                N_test.append(len(y_true[y_pred!=-1]))

                y_pred_all.append(y_pred[y_pred!=-1])
                y_test_all.append(y_test[y_pred!=-1])

                print(f'Test Acc = {test_acc:.4f}')

            print('conf_threshold', conf_threshold)
            print(f'Test Acc = {np.mean(test_acc_all):.4f}+-{np.std(test_acc_all, ddof=1):.4f}')
            cms = np.array([[np.mean(pred_TP),np.mean(pred_FP)],
                            [np.mean(pred_FN), np.mean(pred_TN)]])
            cms_stds = np.array([[np.std(pred_TP, ddof=1),np.std(pred_FP, ddof=1)],
                                [np.std(pred_FN, ddof=1), np.std(pred_TN, ddof=1)]])

            fontsize = 22
            group_counts = ["{}".format(value) for value in N_test]

            group_percentages = ["{:0.0f}\u00B1{:0.0f}%".format(mu*100,sig*100) for mu,sig in
                            zip(cms.flatten(), cms_stds.flatten())]

            labels = np.asarray(group_percentages).reshape(2,2)

            plt.figure(figsize=(6,5))
            ax = sns.heatmap(cms*100, annot=labels, fmt='', cmap='Blues', annot_kws={"fontsize":fontsize},
                            vmin=0, vmax=100)
            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=20)
            plt.xlabel('Predicted', size=20)
            plt.ylabel('True', size=20)
            diffs =  np.array([l1,l2])
            plt.xticks(np.linspace(0.5, 1.5, 2), diffs, rotation=0, size=14)
            plt.yticks(np.linspace(0.5, 1.5, 2), diffs, rotation=45, size=14)
            flat_acc = np.mean(np.hstack(y_pred_all)[np.hstack(y_pred_all)!=-1] == np.hstack(y_test_all)[np.hstack(y_pred_all)!=-1])
            f1_ = f1_score(np.hstack(y_pred_all)[np.hstack(y_pred_all)!=-1], np.hstack(y_test_all)[np.hstack(y_pred_all)!=-1], average='macro')
            plt.title(
                'N: {} tracks, conf. threshold{} \nAccuracy: {:.2f} {:.2f}, F1-score: {:.3f}'.format(
                    np.sum(N_test), conf_threshold, np.mean(np.array(test_acc_all)*100), np.std(np.array(test_acc_all)*100, ddof=1), f1_), size=16)
            plt.tight_layout()
            plt.savefig('../deepspt_results/figures/{}_{}_confusion_matrix.pdf'.format(exp_type, conditions_to_pred),
                        pad_inches=0.2, bbox_inches='tight')
            plt.show()
            plt.close()

            acc_all.append(test_acc_all)
            N_all.append(np.sum(N_test))
            recall1_all.append(pred_TP)
            recall2_all.append(pred_TN)
        
        mean_acc = np.nanmean(acc_all, axis=1)
        std_acc = np.nanstd(acc_all, axis=1, ddof=1)
        pred_TP_all = np.nanmean(recall1_all, axis=1)
        pred_TN_all = np.nanmean(recall2_all, axis=1)

        pred_TP_all_std = np.nanmean(recall1_all, axis=1)
        pred_TN_all_std = np.nanmean(recall2_all, axis=1)

        results_dict_uncertaintracks[exp_type] = [proba_all, pred_all, X_test_idx_order]
        results_dict[exp_type] = [mean_acc, std_acc, N_all, pred_TP_all, pred_TN_all, pred_TP_all_std, pred_TN_all_std, acc_all]

    pickle.dump(results_dict_uncertaintracks, open('../deepspt_results/analytics/{}results_dict_uncertaintracks.pkl'.format(conditions_to_pred), 'wb'))
    pickle.dump(results_dict, open('../deepspt_results/analytics/{}_results_dict.pkl'.format(conditions_to_pred), 'wb'))

# %%

# pred as above but not cell by cell cross validation

import torch
from sklearn.model_selection import LeaveOneGroupOut

results_dict_per_cell = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# for exp_type in ['ap2', 'E_vs_N']:
for conditions_to_pred in ['alpha', 'D', 'D+alpha', 'all']:
    for exp_type in ['E_vs_N', 'ap2']: 
        N_all = []
        acc_all = []
        recall1_all = []
        recall2_all = []

        for conf_threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
            print('conditions_to_pred', conditions_to_pred)
            if exp_type == 'ap2':
                exp_groups = pickle.load(open('../deepspt_results/analytics/AP2_expname_DeepSPT.pkl', 'rb'))
                FP_all = np.array(pickle.load(open('../deepspt_results/analytics/AP2_FPX_DeepSPT.pkl', 'rb'))[:,1:2])
                if conditions_to_pred == 'alpha':
                    FP_all = np.array(pickle.load(open('../deepspt_results/analytics/AP2_FPX_DeepSPT.pkl', 'rb'))[:,0:1])
                elif conditions_to_pred == 'D':
                    FP_all = np.array(pickle.load(open('../deepspt_results/analytics/AP2_FPX_DeepSPT.pkl', 'rb'))[:,1:2])
                elif conditions_to_pred == 'D+alpha':
                    FP_all = np.array(pickle.load(open('../deepspt_results/analytics/AP2_FPX_DeepSPT.pkl', 'rb'))[:,0:2])
                else:
                    FP_all = np.array(pickle.load(open('../deepspt_results/analytics/AP2_FPX_DeepSPT.pkl', 'rb')))
                y_all = np.array(pickle.load(open('../deepspt_results/analytics/AP2_FPy_DeepSPT.pkl', 'rb')))
                l1, l2 = 'Dorsal', 'Ventral'
                lr = 10**-3
                num_epochs =  10
                batch_size = 32
                end_layers = [2]


            elif exp_type == 'E_vs_N':
                exp_groups = pickle.load(open('../deepspt_results/analytics/expname_all_EEA1NPC1_DeepSPT.pkl', 'rb'))
                if conditions_to_pred == 'alpha':
                    FP_all = np.array(pickle.load(open('../deepspt_results/analytics/EEA1_NPC1_only_FPX_DeepSPT.pkl', 'rb'))[:,0:1])
                elif conditions_to_pred == 'D':
                    FP_all = np.array(pickle.load(open('../deepspt_results/analytics/EEA1_NPC1_only_FPX_DeepSPT.pkl', 'rb'))[:,1:2])
                elif conditions_to_pred == 'D+alpha':
                    FP_all = np.array(pickle.load(open('../deepspt_results/analytics/EEA1_NPC1_only_FPX_DeepSPT.pkl', 'rb'))[:,0:2])
                else:
                    FP_all = np.array(pickle.load(open('../deepspt_results/analytics/EEA1_NPC1_only_FPX_DeepSPT.pkl', 'rb')))

                y_all = np.array(pickle.load(open('../deepspt_results/analytics/EEA1_NPC1_only_FPy_DeepSPT.pkl', 'rb')))
                l1, l2 = 'EEA1', 'NPC1'
                lr = 10**-3
                num_epochs =  10
                batch_size = 32
                end_layers = [2]



            #X = np.column_stack([FP_all.copy()[:,:], tTDP_individuals[:,:].copy()])
            X = FP_all.copy()
            X[np.isnan(X)] = 0
            y = y_all.copy()

            print('X.shape', X.shape, 'y.shape', y.shape)
            # Define your model
            layers = [X.shape[1]] + end_layers
            print('layers', layers)

            # Define your loss function and optimizer
            from torch import optim
            import torch.nn as nn
            import torch
            criterion = nn.CrossEntropyLoss()
            
            pred_TP = []
            pred_FP = []
            pred_TN = []
            pred_FN = []
            val_loss_all = []
            train_loss_all = []
            val_acc_all = []
            train_acc_all = []
            N_test = []
            y_pred_all = []
            y_test_all = []

            eea1_recall_list = []
            npc1_recall_list = []
            eea1_recall_w_coloc_list = []
            npc1_recall_w_coloc_list = []

            idx_w_coloc_in_split_list_all = []
            idx_w_coloc_in_split_list_sure = []
            idx_w_coloc_in_split_list_unsure = []

            test_acc_all = []
            kf = LeaveOneGroupOut()
            for X_train_idx, X_test_idx in kf.split(X, y, exp_groups):

                model = MLP(layers, device=device)
                optimizer = optim.Adam(model.parameters(), lr=lr)


                # make X_valid_idx x % of X_train_idx shuffled
                X_valid_idx = np.random.choice(X_train_idx, int(len(X_train_idx)*0.2), replace=False)
                X_train_idx = np.setdiff1d(X_train_idx, X_valid_idx)

                X_train, y_train = X[X_train_idx], y[X_train_idx]
                X_valid, y_valid = X[X_valid_idx], y[X_valid_idx]
                X_test, y_test = X[X_test_idx], y[X_test_idx]

                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_valid = scaler.transform(X_valid)
                X_test = scaler.transform(X_test)

                from imblearn.over_sampling import RandomOverSampler
                ros = RandomOverSampler(random_state=random_state)
                X_train, y_train = ros.fit_resample(X_train, y_train)
                
                # Create your training and validation datasets
                train_dataset = EndosomeDataset(X_train, y_train, device=device)
                val_dataset = EndosomeDataset(X_valid, y_valid, device=device)
                test_dataset = EndosomeDataset(X_test, y_test, device=device)

                # Create your training and validation data loaders
                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

                # Train your model

                train_loss_list = []
                val_loss_list = []
                train_acc_list = []
                val_acc_list = []
                best_val_acc = 0
                for epoch in range(num_epochs):
                    model.train()
                    for i, (inputs, labels) in enumerate(train_loader):
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()

                    train_acc, train_loss = validate(model, train_loader, criterion)
                    val_acc, val_loss = validate(model, val_loader, criterion)
                    train_loss_list.append(train_loss)
                    val_loss_list.append(val_loss)
                    train_acc_list.append(train_acc)
                    val_acc_list.append(val_acc)

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_model = model
                
                    if epoch+1%2==0:
                        print(f'Epoch {epoch+1}/{num_epochs}: Train Loss = {train_loss:.4f}, Val loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}')
                
                train_loss_all.append(train_loss_list)
                val_loss_all.append(val_loss_list)
                train_acc_all.append(train_acc_list)
                val_acc_all.append(val_acc_list)

                # Test your model
                best_model.eval()

                y_true = []
                y_pred = []
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        outputs = model(inputs)
                        proba, predicted = torch.max(outputs.data, 1)
                        predicted_pre = predicted.clone()
                        predicted[proba<conf_threshold] = -1
                        y_true.extend(labels.cpu().numpy().tolist())
                        y_pred.extend(predicted.cpu().numpy().tolist())

                y_true = np.hstack(y_true).astype(int)
                y_pred = np.hstack(y_pred).astype(int)
                test_acc = np.nanmean(y_true[y_pred!=-1]==y_pred[y_pred!=-1])
                
                test_acc_all.append(test_acc)
                cm = confusion_matrix(y_true[y_pred!=-1], y_pred[y_pred!=-1], labels=[0,1], normalize='true')


                pred_TP.append(np.mean(y_true[y_pred!=-1][y_true[y_pred!=-1]==0]==y_pred[y_pred!=-1][y_true[y_pred!=-1]==0]))
                pred_FP.append(1-np.mean(y_true[y_pred!=-1][y_true[y_pred!=-1]==0]==y_pred[y_pred!=-1][y_true[y_pred!=-1]==0]))
                pred_TN.append(np.mean(y_true[y_pred!=-1][y_true[y_pred!=-1]==1]==y_pred[y_pred!=-1][y_true[y_pred!=-1]==1]))
                pred_FN.append(1-np.mean(y_true[y_pred!=-1][y_true[y_pred!=-1]==1]==y_pred[y_pred!=-1][y_true[y_pred!=-1]==1]))
                N_test.append(len(y_true[y_pred!=-1]))

                y_pred_all.append(y_pred[y_pred!=-1])
                y_test_all.append(y_test[y_pred!=-1])

                print(f'Test Acc = {test_acc:.4f}')

            print('conf_threshold', conf_threshold)
            print(f'Test Acc = {np.mean(test_acc_all):.4f}+-{np.std(test_acc_all, ddof=1):.4f}')
            cms = np.array([[np.nanmean(pred_TP),np.nanmean(pred_FP)],
                            [np.nanmean(pred_FN), np.nanmean(pred_TN)]])
            cms_stds = np.array([[np.nanstd(pred_TP, ddof=1),np.nanstd(pred_FP, ddof=1)],
                                [np.nanstd(pred_FN, ddof=1), np.nanstd(pred_TN, ddof=1)]])

            fontsize = 22
            group_counts = ["{}".format(value) for value in N_test]

            group_percentages = ["{:0.0f}\u00B1{:0.0f}%".format(mu*100,sig*100) for mu,sig in
                            zip(cms.flatten(), cms_stds.flatten())]

            labels = np.asarray(group_percentages).reshape(2,2)

            plt.figure(figsize=(6,5))
            ax = sns.heatmap(cms*100, annot=labels, fmt='', cmap='Blues', annot_kws={"fontsize":fontsize},
                            vmin=0, vmax=100)
            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=20)
            plt.xlabel('Predicted', size=20)
            plt.ylabel('True', size=20)
            diffs =  np.array([l1,l2])
            plt.xticks(np.linspace(0.5, 1.5, 2), diffs, rotation=0, size=14)
            plt.yticks(np.linspace(0.5, 1.5, 2), diffs, rotation=45, size=14)
            flat_acc = np.mean(np.hstack(y_pred_all)[np.hstack(y_pred_all)!=-1] == np.hstack(y_test_all)[np.hstack(y_pred_all)!=-1])
            f1_ = f1_score(np.hstack(y_pred_all)[np.hstack(y_pred_all)!=-1], np.hstack(y_test_all)[np.hstack(y_pred_all)!=-1], average='macro')
            plt.title(
                'N: {} tracks, conf. threshold{} \nAccuracy: {:.2f} {:.2f}, F1-score: {:.3f}'.format(
                    np.sum(N_test), conf_threshold, np.mean(np.array(test_acc_all)*100), np.std(np.array(test_acc_all)*100, ddof=1), f1_), size=16)
            plt.tight_layout()
            plt.savefig('../deepspt_results/figures/{}_{}_confthreshold{}_confusion_matrix_percell.pdf'.format(exp_type, conditions_to_pred, conf_threshold),
                        pad_inches=0.2, bbox_inches='tight')
            plt.show()
            plt.close()
            
            acc_all.append(test_acc_all)
            N_all.append(np.sum(N_test))
            recall1_all.append(pred_TP)
            recall2_all.append(pred_TN)
        
        mean_acc = np.nanmean(acc_all, axis=1)
        std_acc = np.nanstd(acc_all, axis=1, ddof=1)
        pred_TP_all = np.nanmean(recall1_all, axis=1)
        pred_TN_all = np.nanmean(recall2_all, axis=1)

        pred_TP_all_std = np.nanmean(recall1_all, axis=1)
        pred_TN_all_std = np.nanmean(recall2_all, axis=1)

        results_dict_per_cell[exp_type] = [mean_acc, std_acc, N_all, pred_TP_all, pred_TN_all, pred_TP_all_std, pred_TN_all_std, acc_all]

    pickle.dump(results_dict_per_cell, open('../deepspt_results/analytics/{}_results_dict_per_cell.pkl'.format(conditions_to_pred), 'wb'))


# %%
import pickle
import numpy as np
import matplotlib.pyplot as plt

conditions_to_pred = 'D+alpha'
results_dict = pickle.load(open('../deepspt_results/analytics/{}_results_dict.pkl'.format(conditions_to_pred), 'rb'))

offset = [0.06, 0.1]

labels_tp = ['EEA1', 'Dorsal']
labels_tn = ['NPC1', 'Ventral']
for i, k in enumerate(list(results_dict.keys())):
    xmin_val = 0
    xmax_val = 101
    if conditions_to_pred=='all':
        if k == 'ap2':
            yinterval = 2
            yinterval2 = 2500
        if k == 'E_vs_N':
            yinterval = 4
            yinterval2 = 2500
    if conditions_to_pred=='D+alpha':
        yinterval = 10
        yinterval2 = 2500
    if conditions_to_pred=='D':
        yinterval = 10
        yinterval2 = 2500
    if conditions_to_pred=='alpha':
        xmin_val = -20
        xmax_val = 101
        yinterval = 10
        yinterval2 = 2500
        xmin_val2 = -2500
        print('if alpha then dont subtract from vmin on Acc. axis')

    if k == 'E_vs_N':
        labels_tn = 'TPR NPC1'
        labels_tp = 'TPR EEA1'
    if k == 'ap2':
        labels_tn = 'TPR Dorsal'
        labels_tp = 'TPR Ventral'

    mean_acc, std_acc, N_all,\
        pred_TP_all, pred_TN_all,\
            pred_TP_all_std, pred_TN_all_std, all_acc = results_dict[k]
    
    mean_acc = np.array(mean_acc)*100
    std_acc = np.array(std_acc)*100
    N_all = np.array(N_all)
    pred_TP_all = np.array(pred_TP_all)*100
    pred_TN_all = np.array(pred_TN_all)*100

    mean_acc[np.isnan(mean_acc)] = 0
    std_acc[np.isnan(std_acc)] = 0
    N_all[np.isnan(N_all)] = 0
    pred_TP_all[np.isnan(pred_TP_all)] = 0
    pred_TN_all[np.isnan(pred_TN_all)] = 0

    print(len(all_acc), len(all_acc[0]))

    fig, ax = plt.subplots(figsize=(8,3.5))
    ax.errorbar(np.arange(len(mean_acc)), mean_acc, yerr=std_acc, 
                fmt='o', label='Accuracy', color='k',
                capsize=5, capthick=1.5, elinewidth=1.5, markeredgewidth=1.5, zorder=10)
    ax.plot(np.arange(len(mean_acc)), mean_acc, color='k', lw=2)
    
    ants = np.repeat(np.arange(len(mean_acc)), len(all_acc[0]))
    ants_spread = ants + np.random.uniform(-0.095, 0.095, len(ants))
    ax.scatter(ants_spread, np.hstack(all_acc)*100,
               color='k', alpha=0.7, zorder=1, s=5)

    ax.plot(np.arange(len(mean_acc)), pred_TP_all, 
            label=labels_tp, zorder=1, color='k', 
            linestyle='dashed', lw=1.5)
    ax.plot(np.arange(len(mean_acc)), pred_TN_all, 
            label=labels_tn, zorder=1, color='k', 
            linestyle='dotted', lw=1.5)
    # ax.errorbar(np.arange(len(mean_acc)), pred_TP_all, yerr=pred_TP_all_std,
    #             fmt='o', label=labels_tp[i], color='grey',
    #             capsize=5, capthick=1.5, elinewidth=1.5, markeredgewidth=1.5)
    # ax.errorbar(np.arange(len(mean_acc)), pred_TN_all, yerr=pred_TN_all_std,
    #             fmt='o', label=labels_tn[i], color='grey',
    #             capsize=5, capthick=1.5, elinewidth=1.5, markeredgewidth=1.5)
    

    ax.set_xticks(np.arange(len(mean_acc)))
    ax.set_xticklabels(['0.5', '0.6', '0.7', '0.8', '0.9'], size=14)

    ax.set_yticks(np.arange(
        np.max([np.round(np.min([mean_acc-std_acc, 
                                 pred_TP_all-pred_TP_all_std, 
                                 pred_TN_all-pred_TN_all_std]))-yinterval,xmin_val]), 
        np.min([np.round(np.max([mean_acc+std_acc, 
                                 pred_TP_all+pred_TP_all_std, 
                                 pred_TN_all+pred_TN_all_std]))+2*yinterval, xmax_val]),
          yinterval))
    
    ax.tick_params(axis='y', labelcolor='k', labelsize=14, width=1.5)

    ax.set_xlabel('Confidence threshold', size=16)
    ax.set_ylabel('Accuracy (%)', size=16)

    
    ax1 = ax.twinx()
    ax1.plot(np.arange(len(mean_acc)), N_all, 'o-', color='grey', label='N tracks', lw=2)
    ax1.set_ylabel('N tracks', size=16, color='dimgrey')

    ax1.set_yticks(np.arange(np.max([np.round(np.min(N_all)/1000)*1000,0]), 
                             np.round(np.max(N_all)/1000)*1000+yinterval2, yinterval2))
    
    ax1.tick_params(axis='y', labelcolor='dimgrey', labelsize=14, width=1.5)
    


    # plot all legends in one box
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax1.get_legend_handles_labels()


    # plot the legend outside and horizontally
    plt.legend(lines[::-1] + lines2[::-1], labels[::-1] + labels2[::-1], loc='upper center', 
               bbox_to_anchor=(0.45, 1.2),
               fancybox=False, shadow=False, ncol=5, fontsize=14,
               frameon=False,
               handletextpad=0.2, columnspacing=0.5, handlelength=1.5,)

    plt.tight_layout()

    plt.savefig('../deepspt_results/figures/{}_ACCnN_vs_confthreshold_{}.pdf'.format(k, conditions_to_pred),
                pad_inches=0.2, bbox_inches='tight')

# %%
import pickle
import numpy as np
import matplotlib.pyplot as plt

conditions_to_pred = 'D+alpha'
print(os.path.exists('../deepspt_results/analytics/{}_results_dict.pkl'.format(conditions_to_pred)))
results_dict_per_cell = pickle.load(open('../deepspt_results/analytics/{}_results_dict_per_cell.pkl'.format(conditions_to_pred), 'rb'))

offset = [0.06, 0.1]

labels_tp = ['EEA1', 'Dorsal']
labels_tn = ['NPC1', 'Ventral']
for i, k in enumerate(list(results_dict_per_cell.keys())):
    xmin_val = 0
    xmax_val = 101
    if conditions_to_pred=='all':
        if k == 'ap2':
            yinterval = 5
            yinterval2 = 2500
        if k == 'E_vs_N':
            yinterval = 5
            yinterval2 = 2500
    if conditions_to_pred=='D+alpha':
        yinterval = 10
        yinterval2 = 2500
    if conditions_to_pred=='D':
        yinterval = 10
        yinterval2 = 2500
    if conditions_to_pred=='alpha':
        xmin_val = -20
        xmax_val = 101
        yinterval = 10
        yinterval2 = 2500
        xmin_val2 = -2500
        print('if alpha then dont subtract from vmin on Acc. axis')

    if k == 'E_vs_N':
        labels_tn = 'TPR NPC1'
        labels_tp = 'TPR EEA1'
    if k == 'ap2':
        labels_tn = 'TPR Dorsal'
        labels_tp = 'TPR Ventral'

    mean_acc, std_acc, N_all,\
        pred_TP_all, pred_TN_all,\
            pred_TP_all_std, pred_TN_all_std, all_acc = results_dict_per_cell[k]
    
    mean_acc = np.array(mean_acc)*100
    std_acc = np.array(std_acc)*100
    N_all = np.array(N_all)
    pred_TP_all = np.array(pred_TP_all)*100
    pred_TN_all = np.array(pred_TN_all)*100

    mean_acc[np.isnan(mean_acc)] = 0
    std_acc[np.isnan(std_acc)] = 0
    N_all[np.isnan(N_all)] = 0
    pred_TP_all[np.isnan(pred_TP_all)] = 0
    pred_TN_all[np.isnan(pred_TN_all)] = 0
    pred_TP_all_std[np.isnan(pred_TP_all_std)] = 0
    pred_TN_all_std[np.isnan(pred_TN_all_std)] = 0

    fig, ax = plt.subplots(figsize=(8,3.5))
    ax.errorbar(np.arange(len(mean_acc)), mean_acc, yerr=std_acc, 
                fmt='o', label='Accuracy', color='k',
                capsize=5, capthick=1.5, elinewidth=1.5, markeredgewidth=1.5, zorder=10)
    ax.plot(np.arange(len(mean_acc)), mean_acc, color='k', lw=2)
    ants = np.repeat(np.arange(len(mean_acc)), len(all_acc[0]))
    ants_spread = ants + np.random.uniform(-0.095, 0.095, len(ants))
    ax.scatter(ants_spread, np.hstack(all_acc)*100,
               color='k', alpha=0.7, zorder=1, s=5)

    ax.plot(np.arange(len(mean_acc)), pred_TP_all, 
            label=labels_tp, zorder=1, color='k', 
            linestyle='dashed', lw=1.5)
    ax.plot(np.arange(len(mean_acc)), pred_TN_all, 
            label=labels_tn, zorder=1, color='k', 
            linestyle='dotted', lw=1.5)
    # ax.errorbar(np.arange(len(mean_acc)), pred_TP_all, yerr=pred_TP_all_std,
    #             fmt='o', label=labels_tp[i], color='grey',
    #             capsize=5, capthick=1.5, elinewidth=1.5, markeredgewidth=1.5)
    # ax.errorbar(np.arange(len(mean_acc)), pred_TN_all, yerr=pred_TN_all_std,
    #             fmt='o', label=labels_tn[i], color='grey',
    #             capsize=5, capthick=1.5, elinewidth=1.5, markeredgewidth=1.5)
    
    ax.set_xticks(np.arange(len(mean_acc)))
    ax.set_xticklabels(['0.5', '0.6', '0.7', '0.8', '0.9'], size=14)

    ax.set_yticks(np.arange(
        np.max([np.round(np.min([mean_acc-std_acc, 
                                 pred_TP_all-pred_TP_all_std, 
                                 pred_TN_all-pred_TN_all_std]))-yinterval,xmin_val]), 
        np.min([np.round(np.max([mean_acc+std_acc, 
                                 pred_TP_all+pred_TP_all_std, 
                                 pred_TN_all+pred_TN_all_std]))+2*yinterval, xmax_val]),
          yinterval))
    
    ax.tick_params(axis='y', labelcolor='k', labelsize=14, width=1.5)

    ax.set_xlabel('Confidence threshold', size=16)
    ax.set_ylabel('Accuracy (%)', size=16)

    
    ax1 = ax.twinx()
    ax1.plot(np.arange(len(mean_acc)), N_all, 'o-', color='grey', label='N tracks', lw=2)
    ax1.set_ylabel('N tracks', size=16, color='dimgrey')

    ax1.set_yticks(np.arange(np.max([np.round(np.min(N_all)/1000)*1000,0]), 
                             np.round(np.max(N_all)/1000)*1000+yinterval2, yinterval2))
    
    ax1.tick_params(axis='y', labelcolor='dimgrey', labelsize=14, width=1.5)
    


    # plot all legends in one box
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax1.get_legend_handles_labels()


    # plot the legend outside and horizontally
    plt.legend(lines[::-1] + lines2[::-1], labels[::-1] + labels2[::-1], loc='upper center', 
               bbox_to_anchor=(0.45, 1.2),
               fancybox=False, shadow=False, ncol=5, fontsize=14,
               frameon=False,
               handletextpad=0.2, columnspacing=0.5, handlelength=1.5,)

    print(k)
    plt.tight_layout()

    plt.savefig('../deepspt_results/figures/{}_ACCnN_vs_confthreshold_percell_{}.pdf'.format(k, conditions_to_pred),
                pad_inches=0.2, bbox_inches='tight')
# %%

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import random 
import torch

FP_all = np.array(pickle.load(open('../deepspt_results/analytics/EEA1_NPC1_only_FPX_DeepSPT.pkl', 'rb')))
y_all = np.array(pickle.load(open('../deepspt_results/analytics/EEA1_NPC1_only_FPy_DeepSPT.pkl', 'rb')))
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
l1, l2 = 'EEA1', 'NPC1'
conf_threshold = 0.6
lr = 10**-3
num_epochs =  10
batch_size = 32
end_layers = [2]

X = FP_all.copy()
X[np.isnan(X)] = 0
y = y_all.copy()

print('X.shape', X.shape, 'y.shape', y.shape)
scaler = StandardScaler()
layers = [X.shape[1], 2]
print('layers', layers)
model = MLP(layers)
print(model)

# Define your loss function and optimizer
from torch import optim
import torch.nn as nn
import torch
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

pred_TP = []
pred_FP = []
pred_TN = []
pred_FN = []
val_loss_all = []
train_loss_all = []
val_acc_all = []
train_acc_all = []
N_test = []
y_pred_all = []
y_test_all = []

eea1_recall_list = []
npc1_recall_list = []
eea1_recall_w_coloc_list = []
npc1_recall_w_coloc_list = []

idx_w_coloc_in_split_list_all = []
idx_w_coloc_in_split_list_sure = []
idx_w_coloc_in_split_list_unsure = []

test_acc_all = []
for cvi, (X_train_idx, X_valid_idx) in enumerate(kf.split(X, y)):
    from imblearn.over_sampling import RandomOverSampler

    X_train, y_train = X[X_train_idx], y[X_train_idx]
    X_valid, y_valid = X[X_valid_idx], y[X_valid_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)

    ros = RandomOverSampler(random_state=random_state)
    X_train, y_train = ros.fit_resample(X_train, y_train)
    
    # Create your training and validation datasets
    train_dataset = EndosomeDataset(X_train, y_train)
    val_dataset = EndosomeDataset(X_valid, y_valid)

    # Create your training and validation data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Train your model
    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []
    best_val_acc = 0
    for epoch in range(num_epochs):
        model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        train_acc, train_loss = validate(model, train_loader, criterion)
        val_acc, val_loss = validate(model, val_loader, criterion)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model   

            modelsavepath = '../deepspt_results/EEA1_NPC1_results/precomputed_files/eea1npc1_classifier/'
            torch.save(model.state_dict(), modelsavepath+'best_model.pt')
            joblib.dump(scaler, modelsavepath+'scaler.pkl')

        if epoch+1%2==0:
            print(f'Epoch {epoch+1}/{num_epochs}: Train Loss = {train_loss:.4f}, Val loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}')
    break
# %%
