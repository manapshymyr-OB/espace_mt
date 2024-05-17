import pystac_client
import stackstac # !!! as of June 5, 2023, stackstac is not compatible with numpy > 1.23.5 !!!
import matplotlib.pyplot as plt
import geopandas as gpd
from sqlalchemy import create_engine, text
from pystac.item import Item
from pystac.item_collection import ItemCollection
from typing import Dict, Any
from shapely.geometry import shape, Polygon
import planetary_computer
from concurrent.futures import ThreadPoolExecutor
import concurrent

import pickle
import pandas as pd
import os

planetary_computer.settings.set_subscription_key('b6d101342e1749f794a03ee36e065971')
catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                    modifier=planetary_computer.sign_inplace)

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


engine = create_engine(f'postgresql://postgres:postgres@localhost:5432/postgres')

def get_data():
    get_building_geom = """SELECT st_transform(st_buffer(st_transform(building_geom, 3857), 100), 4326) as geom, building_id
    FROM public.nutz_building where  ndvi_calc is null order by building_id"""

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


    items = catalog.search(
           intersects=geom,
           collections=["sentinel-2-l2a"],
            datetime="2023-06-01/2023-09-01").item_collection()


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
    # print(original_coordinates)
    item_collection = ItemCollection(items=new_items, )
    try:
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
        sentinel_table_filteB04 = sentinel_table[(sentinel_table['SCL'] == 4) |
                                                (sentinel_table['SCL'] == 5) | (sentinel_table['SCL'] == 6) | (sentinel_table['SCL'] == 7)]


        sentinel_table_filteB04 = sentinel_table_filteB04.reset_index()

        # sentinel_table_filteB04.to_excel('test.xlsx')

        gdf = gpd.GeoDataFrame(
            sentinel_table_filteB04,
            geometry=gpd.points_from_xy(sentinel_table_filteB04['lon'],
                                        sentinel_table_filteB04['lat']), crs="EPSG:4326"
        )

        # gdf['ndvi_mean'] =
        # gdf['ndvi_min'] =
        # gdf['ndvi_max'] =
        # gdf['building_id'] = id
        # gdf.to_postgis('ndvi_ver2', engine, if_exists='append')
        update_qry = text(
            f"UPDATE public.nutz_building SET ndvi_mean = {sentinel_table_filteB04['ndvi'].mean()}, ndvi_max = {sentinel_table_filteB04['ndvi'].max()}, ndvi_min = {sentinel_table_filteB04['ndvi'].min()},  ndvi_calc = 'done' WHERE building_id = {id};")

        # update_qry = text(
        #     f"UPDATE public.nutz_building SET ndvi_calc = 'done' WHERE building_id = {id};")

        with engine.begin() as conn:  # Ensures the connection is properly closed after operation
            conn.execute(update_qry)
    except Exception as e:
        print(e)
        print(id)

def main():
    data = get_data()
    print(len(data))
    # print(data)
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(process_geom, id, geom) for id, geom in data.items()]

        for future in concurrent.futures.as_completed(futures):
            id, coverage = future.result()
            print(f"Processed building ID {id} with coverage {coverage}%")

if __name__ == '__main__':
    main()
# def concat_pickles(folder):
#     dfs = []
#     for filename in os.listdir(folder):
#         if 'ndvi_' in filename:
#             print(filename)
#             pickle_filename = os.path.join(folder, filename)
#             with open(pickle_filename, 'rb') as handle:
#                 b = pickle.load(handle)
#                 dfs.append(b)
#
#     df = pd.concat(dfs)
#     print(df.shape)
#     print(df.columns)
#     df = df.drop_duplicates(subset=['building_ids'])
#     print(df.shape)
#     df.columns = ['building_id', 'ndvi_mean', 'ndvi_min', 'ndvi_max']
#     df.to_sql('ndvi_temp', engine, if_exists='append', index=False)
#
# concat_pickles(r'D:\New folder\MT\espace_mt\ndvi_chunks')