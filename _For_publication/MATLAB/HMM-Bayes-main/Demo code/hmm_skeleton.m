% HMM_SKELETON An example file demonstrating how to create a matrix of
% displacements from a trajectory, initialize HMM parameters, run the HMM
% on that series of displacements, and then plot the result.


% Load a trajectory

test_X_path = '2022422185_SimDiff_indeptest_dim2_ntraces20000_Drandom0.0001-0.5_dt1.0e+00_N5-600_B0.05-0.25_R5-25_subA0-0.7_superA1.3-2_Q1-16_X.pkl';
test_y_path = '2022422185_SimDiff_indeptest_dim2_Drandom0.0001-0.5_dt1.0e+00_N5-600_B0.05-0.25_R5-25_subA0-0.7_superA1.3-2_Q1-16_timeresolved_y.pkl';

filename_X = append('data/', test_X_path);
fid=py.open(filename_X,'rb');
data=py.pickle.load(fid);

filename_Y = append('data/', test_y_path);
fid=py.open(filename_Y,'rb');
target=py.pickle.load(fid);

data_ex = importdata('data/example_track.mat');
track_ex = data_ex.track_fig1;

for c = 19000:20000
    if mod(c,200) ~= 0
        continue
    end
    c
    py_data = data(c);
    track = transpose(str2num(py_data.string));
    
    cfg.fs = 1;
    cfg.locerror = 0;
    cfg.umperpx = 1;
    
    % cfg is a structure with three members:
    % cfg.umperpx -> micrometers per pixel conversion factor
    % cfg.fs -> framerate of the movie
    % cfg.locerror -> localization error
    
    % Set up parameters for the HMM-Bayes algorithm
    
    mcmc_params.parallel = 'on'; % turn off if parallel is not available
    maxK = 4; % maximum number of hidden states to test
    
    % Create displacements from a [# dimensions x track length] trajectory
    
    steps = track(:,2:end) - track(:,1:end-1);
    
    % Run the algorithm
    % Compile the results into a structure for feeding into the plotting
    % routine and later analysis.
    
    [results.PrM, results.ML_states, results.ML_params, results.full_results, full_fitting, results.logI]...
        = hmm_process_dataset(steps,maxK,mcmc_params);
    
    results.track = track;
    results.steps = steps;
    
    % Save the results of the analysis
    %
    % full_fitting is stored in a separate .mat because it can be quite
    % large and results in long .mat load times. This file contains the
    % samples of model parameters from Markov Chain Monte Carlo.
    
    save(append('data/analytics/analysis_output_',string(c)),'cfg','results')
    save(append('data/analytics/analysis_output_samples',string(c)),'cfg','results','full_fitting')
end
% Run the plotting algorithm

% hmm_results_plot(cfg,results);