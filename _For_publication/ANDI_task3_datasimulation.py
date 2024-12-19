# %%
import numpy as np
import pickle
import matplotlib.pyplot as plt
import datetime
import sys
sys.path.append('../')
from deepspt_src import *
from global_config import globals
import warnings
from Andi_Challenge_Code.andi_datasets.datasets_theory import datasets_theory
from Andi_Challenge_Code.andi_datasets.datasets_challenge import challenge_theory_dataset

warnings.filterwarnings("ignore")

datapath = '../Andi_Challenge_Code/ANDI_challenge_testsets/'
filename = 'task3.txt'

# DT = datasets_theory()
# dataset = DT.create_dataset(T = 30, N_models = 1000, exponents = np.arange(0.1,1,0.05), models = [0,1,2,4], 
#                               load_trajectories = True, path = datapath)

N_save = 20
print('N_save', N_save)
save_dataset = True
path_datasets = datapath+'X3_3D_{}_train1'.format(N_save)
path_datasets = datapath+'X3_3D_{}_hypoptVal'.format(N_save)
path_datasets = datapath+'X3_3D_{}_tester'.format(N_save)
print('save', path_datasets)
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
                                                  load_dataset = False,)

X3_2D_test2 = np.array([np.column_stack([track[:200],track[200:]]).astype(float) for track in X3_test2[1]])
Y3_2D_test2 = np.vstack(Y3_test2[1])

X3_3D_test2 = np.array([np.column_stack([track[:200],track[200:400], track[400:]]).astype(float) for track in X3_test2[2]])
Y3_3D_test2 = np.vstack(Y3_test2[2])


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



print(X3_2D_test2.shape, Y3_2D_test2.shape)
print(X3_3D_test2.shape, Y3_3D_test2.shape)

os.listdir(datapath)
