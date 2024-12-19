# %%
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from Andi_Challenge_Code.andi_datasets.datasets_theory import datasets_theory

AD = datasets_theory()

methodJ_pred_2D = pickle.load(open('../Andi_Challenge_Code/analytics/MethodJ_{}D_temporal_pred_hetero.pkl'.format(2), 'rb'))
methodJ_true_2D = pickle.load(open('../Andi_Challenge_Code/analytics/MethodJ_{}D_ensemble_temporal_true_hetero.pkl'.format(2), 'rb'))
print('J', np.mean(np.hstack(methodJ_true_2D)==np.hstack(methodJ_pred_2D)))

methodE_pred_2D = pickle.load(open('../Andi_Challenge_Code/analytics/MethodE_{}D_temporal_pred_hetero.pkl'.format(2), 'rb'))
methodE_true_2D = pickle.load(open('../Andi_Challenge_Code/analytics/MethodE_{}D_ensemble_temporal_true_hetero.pkl'.format(2), 'rb'))
print('E', np.mean(np.hstack(methodE_true_2D)==np.hstack(methodE_pred_2D)))

methodE_pred_3D = pickle.load(open('../Andi_Challenge_Code/analytics/MethodE_{}D_temporal_pred_hetero.pkl'.format(3), 'rb'))
methodE_true_3D = pickle.load(open('../Andi_Challenge_Code/analytics/MethodE_{}D_ensemble_temporal_true_hetero.pkl'.format(3), 'rb'))
print('E', np.mean(np.hstack(methodE_true_3D)==np.hstack(methodE_pred_3D)))

DeepSPT_pred_2D = pickle.load(open('../Andi_Challenge_Code/analytics/ANDIdeepspt_hetero_{}D_temporal_pred_hetero.pkl'.format(2), 'rb'))
DeepSPT_true_2D = pickle.load(open('../Andi_Challenge_Code/analytics/ANDIdeepspt_hetero_{}D_ensemble_temporal_true_hetero.pkl'.format(2), 'rb'))
print('DeepSPT', np.mean(np.hstack(DeepSPT_true_2D)==np.hstack(DeepSPT_pred_2D)))

DeepSPT_pred_3D = pickle.load(open('../Andi_Challenge_Code/analytics/ANDIdeepspt_hetero_{}D_temporal_pred_hetero.pkl'.format(3), 'rb'))
DeepSPT_true_3D = pickle.load(open('../Andi_Challenge_Code/analytics/ANDIdeepspt_hetero_{}D_ensemble_temporal_true_hetero.pkl'.format(3), 'rb'))
print('DeepSPT', np.mean(np.hstack(DeepSPT_true_3D)==np.hstack(DeepSPT_pred_3D)))

assert np.mean(np.hstack(methodE_true_2D)==np.hstack(DeepSPT_true_2D))==1
assert np.mean(np.hstack(methodJ_true_2D)==np.hstack(methodE_true_2D))==1
assert np.mean(np.hstack(methodJ_true_2D)==np.hstack(DeepSPT_true_2D))==1
assert np.mean(np.hstack(methodE_true_3D)==np.hstack(methodE_true_3D))==1

for method in ['J_hetero', 'E_hetero', 'DeepSPT_hetero']:
    if method == 'J_hetero':
        pred_2D = methodJ_pred_2D
        true_2D = methodJ_true_2D
        pred_3D = None
        true_3D = None
    elif method == 'E_hetero':
        pred_2D = methodE_pred_2D
        true_2D = methodE_true_2D
        pred_3D = methodE_pred_3D
        true_3D = methodE_true_3D
    elif method == 'DeepSPT_hetero':
        pred_2D = DeepSPT_pred_2D
        true_2D = DeepSPT_true_2D
        pred_3D = DeepSPT_pred_3D
        true_3D = DeepSPT_true_3D

    if pred_2D is not None:
        temporal_pred = pred_2D
        temporal_true = true_2D

        acc = []
        for i in range(len(temporal_pred)):
            acc.append(np.mean(temporal_pred[i]==temporal_true[i]))

        plt.figure(figsize=(5,2))
        plt.hist(acc, bins=50)
        plt.annotate('N tracks: {}'.format(len(acc)), xy=(0.02, 0.90), 
                    xycoords='axes fraction', ha='left', va='center',
                    fontsize=12)
        plt.annotate('Mean acc. per track: {:.0f}%'.format(100*np.mean(acc)), xy=(0.02, 0.77), 
                    xycoords='axes fraction', ha='left', va='center',
                    fontsize=12)
        plt.annotate('Median acc. per track: {:.0f}%'.format(100*np.median(acc)), xy=(0.02, 0.64), 
                    xycoords='axes fraction', ha='left', va='center',
                    fontsize=12)
        plt.xlim(0,1)
        plt.ylim(0,2000)
        plt.title(method+' 2D')
        plt.xticks([0,0.5,1], labels=['0', '50', '100'])
        plt.savefig('../Andi_Challenge_Code/analytics/{}_2D_acc.pdf'.format(method),
                    bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.show()
        plt.close()

        p, t = np.hstack(temporal_pred), np.hstack(temporal_true)

        # confusion matrix

        from sklearn.metrics import confusion_matrix, f1_score
        import seaborn as sns

        F1score = f1_score(t, p, average='macro')*100

        cm = confusion_matrix(t, p)
        cm = (cm / cm.sum(axis=1)[:, np.newaxis])*100
        print(cm.shape)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, cmap='Blues', 
                    xticklabels=AD.avail_models_name, 
                    yticklabels=AD.avail_models_name,
                    annot_kws={"size": 14},
                    vmax=100, vmin=0)
        plt.title(method+'2D \n N tracks: {}, N timepoints: {}\nper-frame acc: {:.0f}%, F1-score: {:.0f}%'\
                    .format(len(acc), len(np.hstack(temporal_true)),
                            100*np.mean(np.hstack(temporal_true)==np.hstack(temporal_pred)),
                            F1score),
                            )
        plt.axis('equal')
        plt.savefig('../Andi_Challenge_Code/analytics/{}_2D_confusion_matrix.pdf'.format(method))
        plt.show()
        plt.close()
    
    if pred_3D is not None:
        temporal_pred = pred_3D
        temporal_true = true_3D

        acc = []
        for i in range(len(temporal_pred)):
            acc.append(np.mean(temporal_pred[i]==temporal_true[i]))

        plt.figure(figsize=(5,2))
        plt.hist(acc, bins=50)
        plt.annotate('N tracks: {}'.format(len(acc)), xy=(0.02, 0.90), 
                    xycoords='axes fraction', ha='left', va='center',
                    fontsize=12)
        plt.annotate('Mean acc. per track: {:.0f}%'.format(100*np.mean(acc)), xy=(0.02, 0.77), 
                    xycoords='axes fraction', ha='left', va='center',
                    fontsize=12)
        plt.annotate('Median acc. per track: {:.0f}%'.format(100*np.median(acc)), xy=(0.02, 0.64), 
                    xycoords='axes fraction', ha='left', va='center',
                    fontsize=12)
        plt.xlim(0,1)
        plt.ylim(0,2500)
        plt.title(method+' 3D')
        plt.xticks([0,0.5,1], labels=['0', '50', '100'])
        plt.savefig('../Andi_Challenge_Code/analytics/{}_3D_acc.pdf'.format(method),
                    bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.show()
        plt.close()

        p, t = np.hstack(temporal_pred), np.hstack(temporal_true)

        # confusion matrix


        from sklearn.metrics import confusion_matrix, f1_score
        import seaborn as sns

        F1score = f1_score(t, p, average='macro')*100

        cm = confusion_matrix(t, p)
        cm = (cm / cm.sum(axis=1)[:, np.newaxis])*100
        print(cm.shape)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, cmap='Blues', 
                    xticklabels=AD.avail_models_name, 
                    yticklabels=AD.avail_models_name,
                    annot_kws={"size": 14},
                    vmax=100, vmin=0)
        plt.title(method+'3D \n N tracks: {}, N timepoints: {}\nper-frame acc: {:.0f}%, F1-score: {:.0f}%'\
                    .format(len(acc), len(np.hstack(temporal_true)),
                            100*np.mean(np.hstack(temporal_true)==np.hstack(temporal_pred)),
                            F1score))
        plt.axis('equal')
        plt.savefig('../Andi_Challenge_Code/analytics/{}_3D_confusion_matrix.pdf'.format(method),
                    bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.show()
        plt.close()