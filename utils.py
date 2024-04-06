import geopandas as gpd
from sqlalchemy import create_engine



engine = create_engine(f'postgresql://postgres:postgres@localhost:5432/postgres')

def read_shp(shp_filename):
    # Read the shapefile using geopandas

    gdf = gpd.read_file(shp_filename)

    # Create WKT representations
    wkt_list = gdf['geometry'].apply(lambda geom: geom.wkt)

    # Print or use the WKT representations as needed
    return wkt_list[0]


def get_data():
    get_building_geom = """SELECT building_geom as geom, building_id
    FROM public.nutz_building where vv_desc_mean is null ;"""

    gdf = gpd.read_postgis(get_building_geom, engine)
    # gdf.to_file('buildings.geojson', driver="GeoJSON")
    # dict
    object_data = dict(zip(gdf["building_id"], gdf['geom']))

    return object_data