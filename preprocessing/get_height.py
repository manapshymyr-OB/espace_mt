"""
Extracts building heights from a tiff based on the intersection of building geometric centroid
Result saved to PostgreSQL database
"""
import pandas as pd
import rasterio
from rasterstats import zonal_stats

import utils

engine = utils.engine
gdf_buildings = utils.get_all_data()

# data manually downloaded from https://geoservice.dlr.de/web/maps/eoc:wsf3d
height = rasterio.open(r'bayern_wfs_building_height.tif')
array = height.read(1)
affine = height.transform
zonal_stats_height = zonal_stats(gdf_buildings, array, affine=affine, stats=['mean'])

df = pd.DataFrame(zonal_stats_height)
df = pd.concat([df, gdf_buildings], axis=1)
del df['geom']
df.to_sql('building_height', con=engine, if_exists='replace')