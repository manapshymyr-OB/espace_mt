import os
import pickle
import pandas as pd
data_dir = 'buiilding_data'
pickle_file = 0
pickle_chunk = f'ndvi_chunks/{pickle_file}.pickle'
# with open(pickle_chunk, "rb") as f:
#     building_ids = pickle.load(f).keys()
building_ids = os.listdir(data_dir)
ndvi = []
counter = 0
ndvi_mean_lst = []
ndvi_min_lst = []
ndvi_max_lst = []
b_ids = []
for id in building_ids:
    try:
        ndvi_pickle_file = os.path.join(data_dir, str(id))
        with open(ndvi_pickle_file, 'rb') as f:
             ndvi_df = pickle.load(f)

        # ndvi.append(ndvi_df)
        # print(ndvi_df)
        ndvi_mean_lst.append(ndvi_df['ndvi'].mean())
        ndvi_min_lst.append(ndvi_df['ndvi'].min())
        ndvi_max_lst.append(ndvi_df['ndvi'].max())
        b_ids.append(id)

        counter+=1
        print(counter)
    except Exception as e:
        print(e)
    # print(id)


data = {

    'building_ids': b_ids,
    'ndvi_mean': ndvi_mean_lst,
    'ndvi_min': ndvi_min_lst,
    'ndvi_max': ndvi_max_lst
}
df = pd.DataFrame.from_dict(data)
print(df)
df.to_pickle(f'ndvi_{pickle_file}')

# print(building_ids[0])
#
#
# pickle_files = os.listdir(data_dir)
#
# print(pickle_files)
