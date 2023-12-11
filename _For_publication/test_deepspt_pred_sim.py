# %%
import numpy as np
import pickle
import matplotlib.pyplot as plt
import datetime
import sys
sys.path.append('../')
from deepspt_src import *

"""Generate a simulated data """

# controls
plot = False
save = True

# variables
n_per_clean_diff = 1 # gets multiplied by n_classes=4
n_classes = 4 # diffusion types if 5 stuck is added
n_changing_traces = 500

random_D = True
multiple_dt = False

Drandomranges_pairs = [
    [[4*10**-3,   8*10**-3],   [4*10**-3,   8*10**-3]],
    [[3.5*10**-3, 7.5*10**-3], [4.5*10**-3, 8.5*10**-3]],
    [[3*10**-3,   7*10**-3],   [5*10**-3,   9*10**-3]],
    [[2.5*10**-3, 6.5*10**-3], [5.5*10**-3, 9.5*10**-3]],
    [[2*10**-3,   6*10**-3],   [6*10**-3,   10*10**-3]],
    [[1.5*10**-3, 5.5*10**-3], [6.5*10**-3, 10.5*10**-3]],
    [[1*10**-3, 5*10**-3],     [7*10**-3,   11*10**-3]]
    ]

accuracy_all = []
accuracy_all_lists = []
accuracy_std_all = []
histogram_intersection_all = []
benchmark_all = []

conditions_to_use = 'all' # 'all' or 'D' or 'alpha' or 'D+alpha'
print(conditions_to_use)
for i,Drandomranges in enumerate(Drandomranges_pairs):
    Nrange = [150,200]
    Brange = [0.05,0.25] 
    Rranges = [[5,12],[8,15]]
    subalpharanges = [[0.3,0.6], [0.4, 0.7]]
    superalpharange = [1.3, 2] 
    Qrange = [6,16] 
    Dfixed = 0.1
    dir_motion = 'active'

    dim = 3 # 2D or 3D
    dt = 1 # s
    max_changepoints = 4 # number of times changing diffusion traces can change
    min_parent_len = 5 # minimum length of subtrace
    total_parents_len = Nrange[1] # max len of changing diff traces
    
    path = '../deepspt_results/tracks/'
    output_name = 'test_deepSPTpred_dim'+str(dim)+'_Didx_'+str(i)
    print(path+output_name)
    if not os.path.exists(path+output_name+'.pkl'):
        changing_diffusion_list_all = []
        print(n_per_clean_diff, n_changing_traces)
        for i in range(2):
            print("Generating data")
            subalpharange = subalpharanges[i]
            Rrange = Rranges[i]
            Drandomrange = Drandomranges[i]
            params_matrix = Get_params(n_per_clean_diff, dt, random_D, multiple_dt,
                                    Nrange = Nrange, Brange = Brange, 
                                    Rrange = Rrange, 
                                    subalpharange = subalpharange,
                                    superalpharange = superalpharange, 
                                    Qrange = Qrange, 
                                    Drandomrange = Drandomrange,
                                    Dfixed = Dfixed)
            NsND, NsAD, NsCD, NsDM, NstD = [params_matrix[i] for i in range(5)]
            Ds, r_cs, ellipse_dims, angles, vs, wiggle, r_stuck, subalphas, superalphas, sigmaND, sigmaAD, sigmaCD, sigmaDM, sigmaStD = params_matrix[7:]

            # Changing diffusion types
            s = datetime.datetime.now()
            changing_diffusion_list, changing_label_list = Gen_changing_diff(n_changing_traces, 
                                                                            max_changepoints, 
                                                                            min_parent_len, 
                                                                            total_parents_len, 
                                                                            dt, random_D=random_D, 
                                                                            n_classes=n_classes, dim=dim,
                                                                            Nrange = Nrange, Brange = Brange, 
                                                                            Rrange = Rrange, 
                                                                            subalpharange = subalpharange,
                                                                            superalpharange = superalpharange, 
                                                                            Qrange = Qrange, 
                                                                            Drandomrange = Drandomrange,
                                                                            Dfixed = Dfixed,
                                                                            DMtype=dir_motion)
            for cl in changing_diffusion_list:
                changing_diffusion_list_all.append(cl)
        pickle.dump(changing_diffusion_list_all, open(path+output_name+'.pkl', 'wb'))

    else:
        changing_diffusion_list_all = pickle.load(open(path+output_name+'.pkl', 'rb'))


    # get consistent result
    seed = 42

    # device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # prep data
    tracks = changing_diffusion_list_all
    X = [x-x[0] for x in tracks]
    print(len(X), 'len X')
    features = ['XYZ', 'SL', 'DP']
    X_to_eval = add_features(X, features)
    y_to_eval = [np.ones(len(x))*0.5 for x in X_to_eval]

    # define dataset and method that model was trained on to find the model
    datasets = ['SimDiff_dim3_ntraces300000_Drandom0.0001-0.5_dt1.0e+00_N5-600_B0.05-0.25_R5-25_subA0-0.7_superA1.3-2_Q1-16']
    methods = ['XYZ_SL_DP']
    dim = 3 if 'dim3' in datasets[0] else 2
    # find the model
    dir_name = ''
    modelpath = '../mlruns/'
    modeldir = '36'
    use_mlflow = False
    if use_mlflow:
        import mlflow
        mlflow.set_tracking_uri('file:'+join(os.getcwd(), "Unet_results"))
        best_models_sorted = find_models_for(datasets, methods)
    else:
        # not sorted tho
        path = '../mlruns/{}'.format(modeldir)
        best_models_sorted = find_models_for_from_path(path)
        print(best_models_sorted)

    # model params
    min_max_len = 601
    X_padtoken = 0
    y_padtoken = 10
    batch_size = 32
    
    savename_score = '../deepspt_results/analytics/testdeepspt_ensemble_score.pkl'
    savename_pred = '../deepspt_results/analytics/testdeepspt_ensemble_pred.pkl'
    rerun_segmentaion = False
    ensemble_score, ensemble_pred = run_temporalsegmentation(
                                 best_models_sorted, 
                                 X_to_eval, y_to_eval,
                                 use_mlflow=use_mlflow,  
                                 dir_name=dir_name, 
                                 device=device, 
                                 dim=dim, 
                                 min_max_len=min_max_len, 
                                 X_padtoken=X_padtoken, 
                                 y_padtoken=y_padtoken,
                                 batch_size=batch_size,
                                 rerun_segmentaion=rerun_segmentaion,
                                 savename_score=savename_score,
                                 savename_pred=savename_pred)


    fp_datapath = '../_Data/Simulated_diffusion_tracks/'
    hmm_filename = 'simulated2D_HMM.json'
    selected_features = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,
                                19,20,21,23,24,25,27,28,29,30,31,
                                32,33,34,35,36,37,38,39,40,41,42])

    FP_1 = create_temporalfingerprint(changing_diffusion_list_all[:n_changing_traces], 
                                        ensemble_pred[:n_changing_traces], fp_datapath, hmm_filename, dim, dt,
                                        selected_features=selected_features)

    FP_2 = create_temporalfingerprint(changing_diffusion_list_all[n_changing_traces:], 
                                        ensemble_pred[n_changing_traces:], fp_datapath, hmm_filename, dim, dt,
                                        selected_features=selected_features)

    def histogram_intersection(h1, h2):
        return np.sum(np.minimum(h1, h2))/np.sum(h1)

    fp_names = np.array(['Alpha', 'D', 'extra', 'pval', 'Efficiency', 'logEfficiency', 'FractalDim', 
                        'Gaussianity', 'Kurtosis', 'MSDratio', 
                        'Trappedness', 't0', 't1', 't2', 't3', 'lifetime', 
                        'avgSL', 'avgMSD', 'AvgDP', 'corrDP',
                        'signDP', 'minSL', 'maxSL',
                        'BroadnessSL', 'CoV', 'FractionSlow', 
                        'FractionFast', 'Volume', 'perc_ND', 'perc_DM', 
                        'perc_CD', 'perc_SD', 'num_changepoints', 'inst_msd_D',
                        'meanSequence', 'medianSequence', 'maxSequence', 
                        'minSequence', 'stdSequence', 'simSeq'])

    featnum = 33
    bins = 50

    fig, ax = plt.subplots(1,1, figsize=(5,5))
    plt.title(fp_names[featnum])
    ax.hist(FP_1[:,featnum], bins=bins, 
            range=(np.min([FP_1[:,featnum], FP_2[:,featnum]]),np.max([FP_1[:,featnum], FP_2[:,featnum]])),
            alpha=0.5)
    ax.hist(FP_2[:,featnum], bins=bins, 
            range=(np.min([FP_1[:,featnum], FP_2[:,featnum]]),np.max([FP_1[:,featnum], FP_2[:,featnum]])),
            alpha=0.5)

    histoverlap_all = []
    for fn in range(FP_1.shape[1]):
        bins_range = np.linspace(np.min([FP_1[:,fn], FP_2[:,fn]]), 
                                np.max([FP_1[:,fn], FP_2[:,fn]]),
                                bins)

        hs1, _ = np.histogram(FP_1[:,fn], bins=bins_range)
        hs2, _ = np.histogram(FP_2[:,fn], bins=bins_range)

        histoverlap = histogram_intersection(hs2, hs1)
        histoverlap_all.append(histoverlap)

    plt.figure()
    plt.plot(histoverlap_all)
    np.argsort(histoverlap_all), 
    print(histoverlap_all[33])

    # acc of linear classifier on FP1 and FP2
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold, StratifiedShuffleSplit
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier

    scaler = StandardScaler()

    FP_pred = np.vstack([FP_1, FP_2])
    if conditions_to_use == 'all':
        FP_pred = FP_pred
    elif conditions_to_use == 'D':
        FP_pred = FP_pred[:, 1:2]
    elif conditions_to_use == 'alpha':
        FP_pred = FP_pred[:, 0:1]
    elif conditions_to_use == 'D+alpha':
        FP_pred = FP_pred[:, 0:2]
    y_before_after = np.hstack([np.zeros(FP_1.shape[0]), np.ones(FP_2.shape[0])])

    kf = StratifiedShuffleSplit(n_splits=5, random_state=0, test_size=0.1)
    kf.get_n_splits(FP_pred)

    accuracy = []
    precision = []
    recall = []
    f1 = []
    TP1 = []
    TP2 = []
    TP3 = []
    FP1 = []
    FP2 = []
    FP3 = []
    print(FP_pred.shape, y_before_after.shape)
    for train_index, test_index in kf.split(FP_pred, y_before_after):
        X_train, X_test = FP_pred[train_index], FP_pred[test_index]
        y_train, y_test = y_before_after[train_index], y_before_after[test_index]

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        X_train[np.isnan(X_train)] = 0
        X_test[np.isnan(X_test)] = 0

        # random oversampling
        from imblearn.over_sampling import RandomOverSampler
        ros = RandomOverSampler(random_state=0)
        X_train, y_train = ros.fit_resample(X_train, y_train)

        clf = LogisticRegression(random_state=0, solver='lbfgs', max_iter=10000,
                                multi_class='multinomial').fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        accuracy.append(accuracy_score(y_test, y_pred))
        precision.append(precision_score(y_test, y_pred, average='micro'))
        recall.append(recall_score(y_test, y_pred, average='macro'))
        f1.append(f1_score(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred, normalize='true')
        TP1.append(cm[0,0])
        FP1.append(cm[0,1])
        TP2.append(cm[1,1])
        FP2.append(cm[1,0])
        #TP3.append(cm[2,2])
        #FP3.append(cm[2,0])

    print('TP1: ', np.round(np.mean(TP1),3), '+/-', np.round(np.std(TP1),3))
    print('TP2: ', np.round(np.mean(TP2),3), '+/-', np.round(np.std(TP2),3))
    print('TP3: ', np.round(np.mean(TP3),3), '+/-', np.round(np.std(TP3),3))
    print('FP1: ', np.round(np.mean(FP1),3), '+/-', np.round(np.std(FP1),3))
    print('FP2: ', np.round(np.mean(FP2),3), '+/-', np.round(np.std(FP2),3))
    print('FP3: ', np.round(np.mean(FP3),3), '+/-', np.round(np.std(FP3),3))
    print('Accuracy: ', np.round(np.mean(accuracy),3), '+/-', np.round(np.std(accuracy),3))
    print('Precision: ', np.round(np.mean(precision),3), '+/-', np.round(np.std(precision),3))
    print('Recall: ', np.round(np.mean(recall),3), '+/-', np.round(np.std(recall),3))
    print('F1: ', np.round(np.mean(f1),3), '+/-', np.round(np.std(f1),3))

    accuracy_all.append(np.mean(accuracy))
    accuracy_all_lists.append(accuracy)
    accuracy_std_all.append(np.std(accuracy, ddof=1))
    histogram_intersection_all.append(histoverlap_all[33])
if save:
    pickle.dump(accuracy_all, open('../deepspt_results/analytics/simDeepSPTpred_cond_'+conditions_to_use+'_accuracy_all.pkl', 'wb'))
    pickle.dump(accuracy_all_lists, open('../deepspt_results/analytics/simDeepSPTpred_cond_'+conditions_to_use+'_accuracy_all_lists.pkl', 'wb'))
    pickle.dump(accuracy_std_all, open('../deepspt_results/analytics/simDeepSPTpred_cond_'+conditions_to_use+'_accuracy_std_all.pkl', 'wb'))
    pickle.dump(histogram_intersection_all, open('../deepspt_results/analytics/simDeepSPTpred_cond_'+conditions_to_use+'_histogram_intersection_all.pkl', 'wb'))

# %%

import pickle
import numpy as np
import matplotlib.pyplot as plt

conditions_to_use = 'all' # 'all' or 'D' or 'alpha' or 'D+alpha'
accuracy_all = pickle.load(open('../deepspt_results/analytics/simDeepSPTpred_cond_'+conditions_to_use+'_accuracy_all.pkl', 'rb'))
accuracy_std_all = pickle.load(open('../deepspt_results/analytics/simDeepSPTpred_cond_'+conditions_to_use+'_accuracy_std_all.pkl', 'rb'))
histogram_intersection_all = pickle.load(open('../deepspt_results/analytics/simDeepSPTpred_cond_'+conditions_to_use+'_histogram_intersection_all.pkl', 'rb'))

print(accuracy_all)
print(accuracy_std_all)
print(histogram_intersection_all)

plt.figure(figsize=(7,3))
plt.errorbar(np.array(histogram_intersection_all), accuracy_all, yerr=accuracy_std_all, 
             fmt='o', markersize=5, color='black', ecolor='k', elinewidth=1, capsize=5,
             capthick=1)
plt.xlabel('Diffusion coefficient overlap')
plt.ylabel('Accuracy')
plt.ylim(0.,1.15)

for txt in range(len(histogram_intersection_all)):
    plt.annotate("{:.0f}%".format(np.round(accuracy_all[txt]*100,0)), 
                 (histogram_intersection_all[txt]-0.025, 
                  accuracy_all[txt]+.04),
                  fontsize=14)

savedir = '../deepspt_results/figures'
# plt.savefig(savedir+'/simDeepSPTpred_accuracy_vs_Doverlap.pdf', bbox_inches='tight')
plt.show()


# %%

import pickle
import numpy as np
import matplotlib.pyplot as plt

conditions_to_use = 'all' # 'all' or 'D' or 'alpha' or 'D+alpha'
plt.figure(figsize=(6, 3.5))

colors_list = ['black', 'darkgrey', 'grey', 'dimgrey']
shape_list = ['o', 'v', '^', 'd']

for i, conditions_to_use in enumerate(['all', 'alpha', 'D', 'D+alpha']):
    accuracy_all = pickle.load(open('../deepspt_results/analytics/simDeepSPTpred_cond_'+conditions_to_use+'_accuracy_all.pkl', 'rb'))
    accuracy_std_all = pickle.load(open('../deepspt_results/analytics/simDeepSPTpred_cond_'+conditions_to_use+'_accuracy_std_all.pkl', 'rb'))
    histogram_intersection_all = pickle.load(open('../deepspt_results/analytics/simDeepSPTpred_cond_'+conditions_to_use+'_histogram_intersection_all.pkl', 'rb'))
    accuracy_all_lists = pickle.load(open('../deepspt_results/analytics/simDeepSPTpred_cond_'+conditions_to_use+'_accuracy_all_lists.pkl', 'rb'))

    name_to_use = conditions_to_use if conditions_to_use != 'all' else 'DeepSPT'
    if name_to_use == 'D+alpha':
        name_to_use = 'D & alpha'
    if name_to_use == 'alpha':
        name_to_use = 'Alpha'

    print()
    print()
    print(accuracy_all)
    print(accuracy_std_all)
    print(histogram_intersection_all)
    
    plt.errorbar(np.array(histogram_intersection_all), accuracy_all, yerr=accuracy_std_all, 
                fmt=shape_list[i], markersize=5, color=colors_list[i], ecolor=colors_list[i], 
                elinewidth=1, capsize=5,
                capthick=1, label=name_to_use)
    
    ants = np.repeat(histogram_intersection_all,len(accuracy_all_lists[0]))
    ants_spread = ants + np.random.normal(0,0.005,len(ants))
    plt.scatter(ants_spread, 
                np.hstack(accuracy_all_lists), s=7, alpha=.8, color=colors_list[i])

    plt.xlabel('Diffusion coefficient overlap (%)', fontsize=16)
    plt.ylabel('Accuracy (%)', fontsize=16)
    plt.ylim(0.,1.15)
    plt.xlim(0.2,.86)
    plt.yticks([0,0.25,0.5,0.75,1.0], labels=[0,25,50,75,100])
    plt.xticks([0.2,0.3,0.4,0.5,0.6,0.7,0.8], labels=[20,30,40,50,60,70,80])

    # if conditions_to_use == 'all':
    #     for txt in range(len(histogram_intersection_all)):
    #         plt.annotate("{:.0f}%".format(np.round(accuracy_all[txt]*100,0)), 
    #                     (histogram_intersection_all[txt]-0.0275, 
    #                     accuracy_all[txt]+.07),
    #                     fontsize=14)

    savedir = '../deepspt_results/figures'

# horizontal legend
plt.legend(bbox_to_anchor=(.44, .01, 1., .102), loc='lower left',
           ncol=2, borderaxespad=0.,
           fontsize=14, frameon=False,
           handletextpad=0.5, columnspacing=.5, labelspacing=0.3,
           borderpad=0.2, handlelength=1.)   

plt.savefig(savedir+'/simDeepSPTpred_accuracy_vs_Doverlap_multiple.pdf', bbox_inches='tight')
plt.show()
# %%

print(len(changing_diffusion_list_all)//2)

path = '../deepspt_results/tracks/'
output_name = 'test_deepSPTpred_dim'+str(3)+'_Didx_'+str(3)
changing_diffusion_list_all_to_plot = pickle.load(open(path+output_name+'.pkl', 'rb'))

tracks1 = changing_diffusion_list_all_to_plot[:n_changing_traces]
tracks2 = changing_diffusion_list_all_to_plot[n_changing_traces:]

import numpy as np

to_plot = 1
N = n_changing_traces//to_plot  # Number of grid positions
x = np.repeat(np.linspace(1,9,N//to_plot),to_plot)
y = np.tile(np.linspace(1,9,N//to_plot),to_plot)
print(x.shape, len(tracks1[::to_plot]))
fig, ax = plt.subplots(1,1, figsize=(10,5))

for i, t in enumerate(tracks2[::to_plot]):
    ax.plot(t[:,0]+12.5, t[:,1], alpha=.25, color='purple')

for i, t in enumerate(tracks1[::to_plot]):
    ax.plot(t[:,0], t[:,1], alpha=.25, color='green')


ax.set_xlim(-8,20)
ax.set_ylim(-7,7.5)

savedir = '../deepspt_results/figures'
print(savedir+'/simDeepSPTpred_tracks_to_plot_{}.pdf'.format(to_plot))
plt.savefig(savedir+'/simDeepSPTpred_tracks_to_plot_{}.pdf'.format(to_plot), bbox_inches='tight')
plt.show()


# %%
from scipy.stats import ttest_ind

# calculate welsh t-test
c_list = ['all', 'alpha', 'D', 'D+alpha']
acc_lists_all = pickle.load(open('../deepspt_results/analytics/simDeepSPTpred_cond_'+c_list[0]+'_accuracy_all_lists.pkl', 'rb'))
acc_lists_alpha = pickle.load(open('../deepspt_results/analytics/simDeepSPTpred_cond_'+c_list[1]+'_accuracy_all_lists.pkl', 'rb'))
acc_lists_D = pickle.load(open('../deepspt_results/analytics/simDeepSPTpred_cond_'+c_list[2]+'_accuracy_all_lists.pkl', 'rb'))
acc_lists_Dalpha = pickle.load(open('../deepspt_results/analytics/simDeepSPTpred_cond_'+c_list[3]+'_accuracy_all_lists.pkl', 'rb'))
histogram_intersection_all = pickle.load(open('../deepspt_results/analytics/simDeepSPTpred_cond_'+conditions_to_use+'_histogram_intersection_all.pkl', 'rb'))

print(len(acc_lists_all), acc_lists_Dalpha[0])

i = 5
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
for i in range(7):
    
    # calculate welsh t-test degree of freedom
    for j in range(3):
        if j == 0:
            t, p = ttest_ind(acc_lists_all[i], acc_lists_Dalpha[i])
            rowname = 'DeepSPT (1) vs D&alpha (2)'
            acc_means_1.append(np.round(100*np.mean(acc_lists_all[i]),5))
            acc_means_2.append(np.round(100*np.mean(acc_lists_Dalpha[i]),5))
            acc_std_1.append(np.round(100*np.std(acc_lists_all[i], ddof=1),5))
            acc_std_2.append(np.round(100*np.std(acc_lists_Dalpha[i], ddof=1),5))

            n1 = len(acc_lists_all[i])
            n2 = len(acc_lists_Dalpha[i])
            vn1 = np.var(acc_lists_all[i]) / n1
            vn2 = np.var(acc_lists_Dalpha[i]) / n2

        elif j == 1:
            t, p = ttest_ind(acc_lists_all[i], acc_lists_alpha[i])
            rowname = 'DeepSPT (1) vs alpha (2)'
            acc_means_1.append(np.round(100*np.mean(acc_lists_all[i]),5))
            acc_means_2.append(np.round(100*np.mean(acc_lists_alpha[i]),5))
            acc_std_1.append(np.round(100*np.std(acc_lists_all[i], ddof=1),5))
            acc_std_2.append(np.round(100*np.std(acc_lists_alpha[i], ddof=1),5))

            n1 = len(acc_lists_all[i])
            n2 = len(acc_lists_alpha[i])
            vn1 = np.var(acc_lists_all[i]) / n1
            vn2 = np.var(acc_lists_alpha[i]) / n2

        elif j == 2:
            t, p = ttest_ind(acc_lists_all[i], acc_lists_D[i])
            rowname = 'DeepSPT (1) vs D (2)'
            acc_means_1.append(np.round(100*np.mean(acc_lists_all[i]),5))
            acc_means_2.append(np.round(100*np.mean(acc_lists_D[i]),5))
            acc_std_1.append(np.round(100*np.std(acc_lists_all[i], ddof=1),5))
            acc_std_2.append(np.round(100*np.std(acc_lists_D[i], ddof=1),5))

            n1 = len(acc_lists_all[i])
            n2 = len(acc_lists_D[i])
            vn1 = np.var(acc_lists_all[i]) / n1
            vn2 = np.var(acc_lists_D[i]) / n2

        # Welch–Satterthwaite equation for dof
        with np.errstate(divide='ignore', invalid='ignore'):
            dof = (vn1 + vn2)**2 / (vn1**2 / (n1 - 1) + vn2**2 / (n2 - 1))

        pvals_all.append(p)
        tvals_all.append(np.round(t,5))
        dof_all.append(np.round(dof,5))
        n1_all.append(n1)
        n2_all.append(n2)
        row_names.append(rowname+', D overlap: '+str(np.round(histogram_intersection_all[i]*100,0))+ '%')

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
print()
df.to_csv('../deepspt_results/analytics/simDeepSPTpred_ttest.csv')

# %%
# cdot
print('\u22C5')
print('\u03BC')

# 10^2 superscript 2
print('3')
