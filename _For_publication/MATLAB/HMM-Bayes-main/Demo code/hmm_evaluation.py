# %%
# load matlab files from .mat
import scipy.io

# load the data
data = scipy.io.loadmat('data/analytics/analysis_output_200.mat')
print(data['results'][0])