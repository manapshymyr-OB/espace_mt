"""
Computes the shape features based on the building geometry
"""
import geopandas as gpd
import momepy

from utils import engine

sql_query = 'select st_transform(building_geom, 25832) geom, building_id  FROM public.nutz_building ;'
buildings = gpd.read_postgis(sql_query, engine)

buildings['area'] = momepy.Area(buildings).series
buildings['perimeter'] = momepy.Perimeter(buildings).series
buildings['circularcompactness'] = momepy.CircularCompactness(buildings).series
buildings['longestaxislength'] = momepy.LongestAxisLength(buildings).series
buildings['elongation'] = momepy.Elongation(buildings).series
buildings['convexity'] = momepy.Convexity(buildings).series
buildings['orientation'] = momepy.Orientation(buildings).series
buildings['corners'] = momepy.Corners(buildings).series
buildings['sharedwall'] = momepy.SharedWalls(buildings).series

buildings.to_postgis('morphological_features', engine)