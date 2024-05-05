import geopandas as gpd
from rasterstats import zonal_stats
import rioxarray
import rasterio
import utils
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine(f'postgresql://postgres:postgres@localhost:5432/postgres')
gdf_buildings = utils.get_all_data()

height = rasterio.open(r'D:\New folder\MT\bayern_wfs_building_height.tif')
array = height.read(1)
print(dir(height))
affine = height.transform

print(gdf_buildings.head())
zonal_stats_height = zonal_stats(gdf_buildings, array, affine=affine, stats=[ 'mean'])

df = pd.DataFrame(zonal_stats_height)
df = pd.concat([df, gdf_buildings], axis=1)
del df['geom']
df.to_sql('height_2', con=engine, if_exists='replace')