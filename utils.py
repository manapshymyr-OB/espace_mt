import geopandas as gpd
from sqlalchemy import create_engine
import pickle

# example connection
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
    FROM public.nutz_building where building_id = 10061726;"""

    gdf = gpd.read_postgis(get_building_geom, engine)

    object_data = dict(zip(gdf["building_id"], gdf['geom']))

    return object_data

def get_data_ndvi_chunk(start, end):
    get_building_geom = f"""SELECT st_transform(st_buffer(building_geom_local, 500), 4326) as geom, building_id
    FROM public.nutz_building where nutz_building.ndvi_calc ='' """

    gdf = gpd.read_postgis(get_building_geom, engine)

    object_data = dict(zip(gdf["building_id"], gdf['geom']))

    return object_data

def chunk_dict(d, size):
    items = list(d.items())
    chunks = [items[i:i+size] for i in range(0, len(items), size)]
    return [dict(chunk) for chunk in chunks]

def ndvi_chunk():
    starting = 50000
    ending = 100000
    data = get_data_ndvi_chunk(starting, ending)

    split_data = chunk_dict(data, starting)
    index = 0
    for chunk in split_data:
        with open(f'ndvi_chunks/{index}_500_1.pickle', 'wb') as handle:
            pickle.dump(chunk, handle)
        index += 1


def get_all_data():
    get_building_geom = """SELECT building_centroid as geom, building_id
    FROM public.nutz_building ;"""

    gdf = gpd.read_postgis(get_building_geom, engine)

    return gdf