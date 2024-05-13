import concurrent
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any

import planetary_computer
import pystac_client
from pystac.item import Item
from pystac.item_collection import ItemCollection
from shapely.geometry import shape
from sqlalchemy import create_engine, text
import stackstac
import math

from utils import get_data

# Initialize database connection
engine = create_engine('postgresql://postgres:postgres@localhost:5432/postgres')

# Set up planetary computer subscription key and STAC catalog
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


original_crs = 'EPSG:4326'


def intersection_percent(item: Item, aoi: Dict[str, Any]) -> float:
    '''The percentage that the Item's geometry intersects the AOI. An Item that
    completely covers the AOI has a value of 100.
    '''
    geom_item = shape(item.geometry)
    geom_aoi = aoi

    intersected_geom = geom_aoi.intersection(geom_item)

    intersection_percent = (intersected_geom.area * 100) / geom_aoi.area

    return intersection_percent


def get_highest_intersection(items, geoms):
    """
    Get an item with the highest intersection area
    """
    highest_intersection_value = 0
    highest_intersection_item = None

    for item in items:
        intersection_value = intersection_percent(item, geoms)
        if intersection_value >= 60:
            if intersection_value > highest_intersection_value:
                highest_intersection_value = intersection_value
                highest_intersection_item = item

    return highest_intersection_item, highest_intersection_value


def process_geom(id, geom):
    """
    Process a single geometry: search for items and update database.
    This function is intended to be executed in a thread pool.
    """
    try:
        search = catalog.search(collections=["sentinel-1-rtc"], datetime="2021-03-01/2021-10-01",
                                query=["s1:resolution=high", 'sat:orbit_state=descending'], intersects=geom)
        items = search.item_collection()
        print(len(items))
        # print(f'here - {geom}')
        # print(items)


        new_items = []
        for item in items:
            perc = intersection_percent(item, geom)
            if perc >= 80:
                new_items.append(item)

        if geom.geom_type == 'MultiPolygon':
            # Get the largest polygon from the MultiPolygon
            geom = get_largest_polygon(geom)
            # Extract the coordinates of the original polygon
        print(f'here - {len(new_items)}')
        item_collection = ItemCollection(items=new_items, )
        sentinel_stack = stackstac.stack(item_collection, assets=["vh", "vv"],
                                         bounds=geom.bounds,
                                         gdal_env=stackstac.DEFAULT_GDAL_ENV.updated(
                                             {'GDAL_HTTP_MAX_RETRY': 3,
                                              'GDAL_HTTP_RETRY_DELAY': 5,
                                              }),
                                         epsg=4326, chunksize=(1, 1, 50, 50)).rename(
            {'x': 'lon', 'y': 'lat'}).to_dataset(dim='band')

        sentinel_stack = sentinel_stack[['vh', 'vv']]
        sentinel_stack = sentinel_stack.drop_vars(
            [c for c in sentinel_stack.coords if not (c in ['time', 'lat', 'lon'])])
        sentinel_table = sentinel_stack.to_dataframe()
        sentinel_table = sentinel_table.reset_index()
        sentinel_table = sentinel_table.sort_values(by='time', ascending=False)
        # Get the first valid index for the 'vh' column (or 'vv' column)
        first_valid_index = sentinel_table['vh'].first_valid_index()

        # Retrieve the date using the found index
        first_not_nan_date = sentinel_table.loc[first_valid_index, 'time'] if first_valid_index is not None else None
        print(first_not_nan_date)
        first_not_nan_date = sentinel_table[sentinel_table['time'] == first_not_nan_date]
        print(first_not_nan_date)

        mean_vh = first_not_nan_date['vh'].mean()
        mean_vv = first_not_nan_date['vv'].mean()
        if math.isnan(mean_vh):
            mean_vh = 'null'
        if math.isnan(mean_vv):
            mean_vv = 'null'
        update_qry = text(
            f"UPDATE public.nutz_building SET vh_desc_mean = {mean_vh}, vv_desc_mean = {mean_vv} WHERE building_id = {id};")
        # # 5424566
        with engine.begin() as conn:  # Ensures the connection is properly closed after operation
            conn.execute(update_qry)
    except Exception as e:
        print(e)
    # time.sleep(321)
#             # print(item.description)
#             vh_uri = item.assets["vh"].href
#             vv_uri = item.assets["vv"].href
#             # print(vh_uri, vv_uri)
#             vh = rioxarray.open_rasterio(vh_uri, masked=True)
#             vv = rioxarray.open_rasterio(vv_uri, masked=True)
#             target_crs = vh.rio.crs
#             # # print(target_crs)
#             #
#             # Create a PyProj transformer object to perform the CRS transformation
#             transformer = pyproj.Transformer.from_crs(original_crs, target_crs, always_xy=True)
#
#             # print(geom.geom_type)
#             if geom.geom_type == 'MultiPolygon':
#                 # Get the largest polygon from the MultiPolygon
#                 geom = get_largest_polygon(geom)
#             # Extract the coordinates of the original polygon
#             # print(geom)
#             original_coordinates = geom.exterior.coords
#
#             # Transform each coordinate of the polygon separately
#             transformed_coordinates = []
#             for x, y in original_coordinates:
#                 new_x, new_y = transformer.transform(x, y)
#                 transformed_coordinates.append((new_x, new_y))
#
#             # Create a new Polygon object with the transformed coordinates
#             transformed_polygon = Polygon(transformed_coordinates)
#             # print(transformed_polygon)
#             # print("Transformed Polygon:", transformed_polygon)
#             # Display the reprojected bounding box
#             vh_clip = vh.rio.clip_box(*transformed_polygon.bounds)
#             vv_clip = vv.rio.clip_box(*transformed_polygon.bounds)
#             mean_vh = np.mean(vh_clip.values)
#             mean_vv = np.mean(vv_clip.values)
#             print(f'Building id {id} - {mean_vh} - {mean_vv} - {vh_uri}')
#             print(mean_vh, mean_vv)
#             if math.isnan(mean_vh):
#                 mean_vh = 'null'
#             if math.isnan(mean_vv):
#                 mean_vv = 'null'
#
#         except Exception as e:
#             print(e)
#     else:
#         print('dasdsa')
    return id


def main():
    data = get_data()
    print(data)
    # print(data)
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(process_geom, id, geom) for id, geom in data.items()]

        for future in concurrent.futures.as_completed(futures):
            id  = future.result()
            print(f"Processed building ID {id} with coverage%")


if __name__ == "__main__":
    main()