# %%
from deepspt_src import *

# path to example data
dim = 3
path = '_Data/Simulated_diffusion_tracks/' # path to save and load
output_name = 'tester_set'+str(dim) # name of output file - change to get new tracks if already run
print(path+output_name)

# load array of tracks saved as pickle
tracks_arr = pickle.load(open(path+output_name+'.pkl', 'rb'))

# load csv file and convert to tracks
df = read_data_csv(path+output_name+'.csv', useful_col=['x', 'y', 'z', 'particle', 'frame'])
tmp1 = Parallel(n_jobs=3)(
            delayed(
                prep_csv_tracks_track)(
                    val, identifiername='particle',
                    timename='frame', xname='x',
                    yname='y', zname='z', center=False
                    ) for val in dict(tuple(df.groupby('particle'))).values())

tracks_csv = np.array([r[0] for r in tmp1], dtype=object)
track_frames = np.array([r[1] for r in tmp1], dtype=object)
track_idx = np.array([np.unique(r[2]) for r in tmp1])

# assert equal for sanity
for i in range(len(tracks_csv)):
    assert np.all(tracks_csv[i][:, 0] == df[df['particle'] == i]['x'].values, )