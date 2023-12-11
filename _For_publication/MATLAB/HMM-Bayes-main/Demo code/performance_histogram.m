array_ML = [];
array_GT = [];

changepoints_GT = [];
changepoints_ML = [];
for c = 4001:20000
    if mod(c,200) ~= 0
        continue
    end
    c
    append('data/analytics/analysis_output_',string(c))
    data = importdata(append('data/analytics/analysis_output_',string(c),'.mat'));
    cfg = data.cfg;
    result = data.results;
    
    test_y_path = '2022422185_SimDiff_indeptest_dim2_Drandom0.0001-0.5_dt1.0e+00_N5-600_B0.05-0.25_R5-25_subA0-0.7_superA1.3-2_Q1-16_timeresolved_y.pkl';
    filename_Y = append('data/', test_y_path);
    fid=py.open(filename_Y,'rb');
    target=py.pickle.load(fid);
    py_target = target(c);
    GT = str2num(py_target.string);
    
    % Extract variables from structures
    
    PrM = result.PrM;
    ML_params = result.ML_params; 
    
    track = result.track;  
    ML_states = result.ML_states; 
    
    cp_ML = 1;
    for i = 1:length(ML_states)-1
        if ML_states(i)~=ML_states(i+1)
            cp_ML = cp_ML + 1     
        end
    end
    changepoints_ML = [changepoints_ML, cp_ML];

    cp_GT = 1;
    for i = 1:length(GT)-1
        if GT(i)~=GT(i+1)
            cp_GT = cp_GT + 1     
        end
    end
    changepoints_GT = [changepoints_GT, cp_GT];
    
end

changepoints_ML
changepoints_GT

figure
s = scatter(changepoints_GT, log(changepoints_ML),'filled')
hold on
s.SizeData = 100;

x = [2, 3, 4];
y = [2, 3, 4];
plot(x, log(y),'-r')

xlabel('Ground Truth') 
ylabel('Prediction') 

legend({'Predictions vs True','Identity'},'Location','northwest')

ax = gca;
ax.FontSize = 13;

length(changepoints_ML)
sort(changepoints_ML)
sum(changepoints_ML==1)/length(changepoints_ML)
sum(changepoints_ML==changepoints_GT)/length(changepoints_ML)
sum(changepoints_ML<changepoints_GT)/length(changepoints_ML)