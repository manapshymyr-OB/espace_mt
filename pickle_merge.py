import os
import pickle
import pandas as pd
data_dir = 'buiilding_data'
pickle_file = 0
pickle_chunk = f'ndvi_chunks/{pickle_file}.pickle'
with open(pickle_chunk, "rb") as f:
    building_ids = pickle.load(f).keys()
ndvi = []
for id in building_ids:
    try:
        ndvi_pickle_file = os.path.join(data_dir, str(id))
        with open(ndvi_pickle_file, 'rb') as f:
             ndvi_df = pickle.load(f)

        ndvi.append(ndvi_df)
    except Exception as e:
        print(e)
    # print(id)

df = pd.concat(ndvi)

df.to_pickle(f'ndvi_{pickle_file}')

# print(building_ids[0])
#
#
# pickle_files = os.listdir(data_dir)
#
# print(pickle_files)
