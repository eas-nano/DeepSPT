changepoints_ML = [];
changepoints_GT = [];
for c = 1:1556
    data = importdata(append('/Users/jacobkh/Documents/PhD/SPT/github_final/DeepSPT/_For_publication/baseline_methods/HMM_rota/analysis_temporalsim_',string(c),'.mat')).results;
    ML_states = data.ML_states;

    for i = 2:length(ML_states)
        if ML_states(i) ~= ML_states(1)
            cp_ML = i-1;
            changepoints_ML(end+1) = i-1;
            break;  % Exit the loop once the first deviation is found
        elseif i == length(ML_states)
            changepoints_ML(end+1) = 0;
        end
    end

    test_y_path = 'tester_set3_dim3_glued_labels_D1.pkl'
    filename_y = append('/Users/jacobkh/Documents/PhD/SPT/github_final/DeepSPT/_Data/Simulated_diffusion_tracks/', test_y_path);
    fid=py.open(filename_y,'rb');
    target=py.pickle.load(fid);
    py_target = target(c);
    GT = double(py.array.array('d', py.numpy.nditer(py_target)));

    for i = 2:length(GT)
        if GT(i) ~= GT(1)
            cp_ML = i-1;
            changepoints_GT(end+1) = i-1;
            break;  % Exit the loop once the first deviation is found
        elseif i == length(GT)
            changepoints_GT(end+1) = 0;
        end
    end

end

% Saving changepoints to .mat file
save('/Users/jacobkh/Documents/PhD/SPT/github_final/DeepSPT/_For_publication/baseline_methods/hmm_tempsim_changepoints_ML.mat', 'changepoints_ML');
save('/Users/jacobkh/Documents/PhD/SPT/github_final/DeepSPT/_For_publication/baseline_methods/hmm_tempsim_changepoints_GT.mat', 'changepoints_GT');

% Saving changepoints to .txt files
dlmwrite('/Users/jacobkh/Documents/PhD/SPT/github_final/DeepSPT/_For_publication/baseline_methods/hmm_tempsim_changepoints_ML.txt', changepoints_ML, 'delimiter', '\t');
dlmwrite('/Users/jacobkh/Documents/PhD/SPT/github_final/DeepSPT/_For_publication/baseline_methods/hmm_tempsim_changepoints_GT.txt', changepoints_GT, 'delimiter', '\t');

% Convert MATLAB arrays to NumPy arrays
np_changepoints_ML = py.numpy.array(changepoints_ML);
np_changepoints_GT = py.numpy.array(changepoints_GT);

% Save as .npy files
py.numpy.save('/Users/jacobkh/Documents/PhD/SPT/github_final/DeepSPT/_For_publication/baseline_methods/hmm_tempsim_changepoints_ML.npy', np_changepoints_ML);
py.numpy.save('/Users/jacobkh/Documents/PhD/SPT/github_final/DeepSPT/_For_publication/baseline_methods/hmm_tempsim_changepoints_GT.npy', np_changepoints_GT);
