"""
Downloads NDVI based on the buffered building geometry
The dataset chunked into smaller chunks to speed up the download.
Chunks can be created using the ndvi_chunk function in utils.py
The result save into building folder
"""

import os
import sys
import time

sys.path.append('..')
import pystac_client
import stackstac # !!! as of June 5, 2023, stackstac is not compatible with numpy > 1.23.5 !!!
import geopandas as gpd
from pystac.item import Item
from pystac.item_collection import ItemCollection
from typing import Dict, Any
from shapely.geometry import shape
import planetary_computer
from concurrent.futures import ThreadPoolExecutor
import concurrent
import pickle
import pandas as pd


# Set up planetary computer subscription key and STAC catalog
planetary_computer_api_key = '<KEY>'
planetary_computer.settings.set_subscription_key(planetary_computer_api_key)
catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                    modifier=planetary_computer.sign_inplace)


download_data_ = os.listdir('../buiilding_data')

download_data = []
for downloaded in download_data_:
    print(f"{downloaded} - {downloaded.split('_')[0]}")
    download_data.append(downloaded.split('_')[0])

def get_largest_polygon(multipolygon):
    # Initialize variables to keep track of the largest polygon and its area
    largest_polygon = None
    max_area = -1

    # Iterate over each polygon within the MultiPolygon
    for polygon in multipolygon.geoms:
        # Calculate the area of the current polygon
        area = polygon.area
        # print(area)

        # Check if the current polygon has a larger area than the previous largest one
        if area > max_area:
            max_area = area
            largest_polygon = polygon

    return largest_polygon

counter = 0
# engine = create_engine(f'postgresql://postgres:postgres@localhost:5432/postgres')

def get_data():
    get_building_geom = """SELECT st_transform(st_buffer(st_transform(building_geom, 3857), 100), 4326) as geom, building_id
    FROM public.nutz_building where building_id <= 50000 order by building_id"""

    gdf = gpd.read_postgis(get_building_geom, engine)
    # gdf.to_file('buildings.geojson', driver="GeoJSON")
    # dict
    object_data = dict(zip(gdf["building_id"], gdf['geom']))

    return object_data


def intersection_percent(item: Item, aoi: Dict[str, Any]) -> float:
    '''The percentage that the Item's geometry intersects the AOI. An Item that
    completely covers the AOI has a value of 100.
    '''
    geom_item = shape(item.geometry)
    geom_aoi = aoi

    intersected_geom = geom_aoi.intersection(geom_item)

    intersection_percent = (intersected_geom.area * 100) / geom_aoi.area

    return intersection_percent

# data = get_data()

def process_geom(id, geom):
    global counter
    if str(id) not in download_data:

        try:
            items = catalog.search(
               intersects=geom,
               collections=["sentinel-2-l2a"],
                datetime="2023-06-01/2023-09-01").item_collection()
        except Exception as e:
            print(e)

        new_items = []
        for item in items:
            perc = intersection_percent(item, geom)
            if perc >= 80:
                new_items.append(item)
                # print(perc)


        if geom.geom_type == 'MultiPolygon':
                # Get the largest polygon from the MultiPolygon
                geom = get_largest_polygon(geom)
                # Extract the coordinates of the original polygon
                # print(geom)
        original_coordinates = geom.exterior.coords
        item_collection = ItemCollection(items=new_items, )

        start = time.time()
        sentinel_stack = stackstac.stack(item_collection, assets=["B04", "B08", "SCL"],
                                         bounds=geom.bounds,
                                         gdal_env=stackstac.DEFAULT_GDAL_ENV.updated(
                                             {'GDAL_HTTP_MAX_RETRY': 3,
                                              'GDAL_HTTP_RETRY_DELAY': 5,
                                              }),
                                         epsg=4326, chunksize=(1, 1, 50, 50)).rename(
            {'x': 'lon', 'y': 'lat'}).to_dataset(dim='band')

        sentinel_stack['ndvi'] = (sentinel_stack['B08'] - sentinel_stack['B04'])/\
                                (sentinel_stack['B08'] + sentinel_stack['B04'])
        sentinel_stack = sentinel_stack[['ndvi', 'SCL','B04', 'B08']]
        sentinel_stack = sentinel_stack.drop_vars([c for c in sentinel_stack.coords if not (c in ['time', 'lat', 'lon'])])
        sentinel_table = sentinel_stack.to_dataframe()

        # filter by pixel classes
        sentinel_table_filtered = sentinel_table[(sentinel_table['SCL'] == 4) |
                                                (sentinel_table['SCL'] == 5) | (sentinel_table['SCL'] == 6) | (sentinel_table['SCL'] == 7)]

        print(f"Calculated in {time.time() - start} - id - {id}")
        sentinel_table_filtered = sentinel_table_filtered.reset_index()
        # print(sentinel_table_filteB04)
        sentinel_table_filtered['building_id'] = id
        # sentinel_table_filtered.to_pickle(f'buiilding_data/{id}_500')

        ndvi_dict = {
            'building_id':[id],
            'ndvi_mean': [sentinel_table_filtered['ndvi'].mean()],
            'ndvi_max': [sentinel_table_filtered['ndvi'].max()],
            'ndvi_min': [sentinel_table_filtered['ndvi'].min()],


        }
        df1 = pd.DataFrame(ndvi_dict)

        try:
            with open(f'buiilding_data/{id}', 'wb') as handle:
                pickle.dump(df1, handle)

            counter += 1

            print(f"""{counter} - {len(os.listdir('../buiilding_data'))} of {id}""")
        except Exception as e:
            print(e)
        # df1.to_pickle(f'buiilding_data/{id}_500')
        # print(df1)

    else:
        counter += 1
        print(f'already downloaded {counter}')

def main():
    with open('../ndvi_chunks/1.pickle', 'rb') as handle:
        data = pickle.load(handle)

    # print(data)
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_geom, id, geom) for id, geom in data.items()]

        for future in concurrent.futures.as_completed(futures):
            id, coverage = future.result()
            print(f"Processed building ID {id} with coverage {coverage}%")


if __name__ == "__main__":
    main()