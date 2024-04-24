import concurrent
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any

import planetary_computer
import pyproj
import pystac_client
import rioxarray
from pystac.item import Item
from shapely.geometry import shape, Polygon
from sqlalchemy import create_engine, text
import numpy as np
import math
import stackstac

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
    time_of_interest = "2019-06-01/2019-08-01"
    while True:
        try:

            search = catalog.search(collections=["sentinel-2-l2a"], datetime="2023-08-01/2023-09-01",
                                    query={"eo:cloud_cover": {"lt": 10}}, intersects=geom)
            items = search.item_collection()

            print(f'here - {geom}')
            # print(items)
            break
        except Exception as e:
            print(e)
            continue

    for item in items:
        print(item.id, ":", item.properties['eo:cloud_cover'])
    item, coverage = get_highest_intersection(items, geom)
    if item:
        try:
            # print(item.assets)
            print(item.id, ":", item.properties['eo:cloud_cover'])

            # print(item.assets)
            # time.sleep(3213)
            # print(item.description)
            red = item.assets["B04"].href
            nir = item.assets["B08"].href
            red = rioxarray.open_rasterio(red, masked=True)
            nir = rioxarray.open_rasterio(nir, masked=True)
            target_crs = red.rio.crs
            # # print(target_crs)
            #
            # Create a PyProj transformer object to perform the CRS transformation
            transformer = pyproj.Transformer.from_crs(original_crs, target_crs, always_xy=True)

            # print(geom.geom_type)
            if geom.geom_type == 'MultiPolygon':
                # Get the largest polygon from the MultiPolygon
                geom = get_largest_polygon(geom)
            # Extract the coordinates of the original polygon
            # print(geom)
            original_coordinates = geom.exterior.coords

            # Transform each coordinate of the polygon separately
            transformed_coordinates = []
            for x, y in original_coordinates:
                new_x, new_y = transformer.transform(x, y)
                transformed_coordinates.append((new_x, new_y))

            # Create a new Polygon object with the transformed coordinates
            transformed_polygon = Polygon(transformed_coordinates)
            print(transformed_polygon)
            # print("Transformed Polygon:", transformed_polygon)
            # Display the reprojected bounding box
            red_clip = red.rio.clip_box(*transformed_polygon.bounds)
            nir_clip = nir.rio.clip_box(*transformed_polygon.bounds)
            red_clip_matched = red_clip.rio.reproject_match(nir_clip)
            ndvi = (nir_clip - red_clip_matched) / (nir_clip + red_clip_matched)
            print(ndvi)

            max_ndvi = np.nanmax(ndvi.values)
            min_ndvi = np.nanmin(ndvi.values)
            mean_ndvi = np.nanmean(ndvi.values)
            print(max_ndvi, min_ndvi, mean_ndvi)
            time.sleep(32131)
            if math.isnan(max_ndvi):
                max_ndvi = 'null'
            if math.isnan(min_ndvi):
                min_ndvi = 'null'
            if math.isnan(mean_ndvi):
                mean_ndvi = 'null'
            update_qry = text(f"UPDATE public.nutz_building_2 SET ndvi_mean = {mean_ndvi}, ndvi_max = {max_ndvi}, ndvi_min = {min_ndvi} WHERE building_id = {id};")

            with engine.begin() as conn:  # Ensures the connection is properly closed after operation
                conn.execute(update_qry)
        except Exception as e:
            print(e)
    else:
        print('dasdsa')
    return id, coverage


def main():
    data = get_data()
    # print(data)
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(process_geom, id, geom) for id, geom in data.items()]

        for future in concurrent.futures.as_completed(futures):
            id, coverage = future.result()
            print(f"Processed building ID {id} with coverage {coverage}%")


if __name__ == "__main__":
    main()