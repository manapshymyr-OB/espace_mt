import pickle

ndvi_pickle_file = 'ndvi_500/ndvi_12_500'

with open(ndvi_pickle_file, 'rb') as f:
    ndvi_df = pickle.load(f)

print(ndvi_df)

with open('ndvi_chunks/12_500.pickle', 'rb') as f:
    data = pickle.load(f)
print(data)


# List of keys to remove
ndvi_df['building_ids_new'] = ndvi_df['building_ids_new'].astype(int)
ids = ndvi_df['building_ids_new'].unique().tolist()
# print(ids)
# Using dictionary comprehension
my_dict = {k: v for k, v in data.items() if k not in ids}

print(len(my_dict))  # Output: {'b': 2, 'd': 4}

with open(f'buiilding_data/12_500_2.pickle', 'wb') as handle:
    pickle.dump(my_dict, handle)
# 32701 - 3_500
# 31132 - 10_500