import geopandas as gpd
from sqlalchemy import create_engine
import pickle




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
    FROM public.nutz_building_2 where ndvi_mean is null ;"""

    gdf = gpd.read_postgis(get_building_geom, engine)
    # gdf.to_file('buildings.geojson', driver="GeoJSON")
    # dict
    object_data = dict(zip(gdf["building_id"], gdf['geom']))

    return object_data

def get_data_ndvi_chunk(start, end):
    get_building_geom = f"""SELECT st_transform(st_buffer(st_transform(building_geom, 3857), 100), 4326) as geom, building_id
    FROM public.nutz_building where building_id > {start} order by building_id;"""

    gdf = gpd.read_postgis(get_building_geom, engine)
    # gdf.to_file('buildings.geojson', driver="GeoJSON")
    # dict
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
        print(len(chunk))
        with open(f'ndvi_chunks/{index}.pickle', 'wb') as handle:
            pickle.dump(data, handle)
        index += 1
    # print(len(data))
    # while True:
    #     if ending < 650000:
    #         starting += 50000
    #         ending += 50000
    #
    #
    #
    #     else:
    #         break

def get_all_data():
    get_building_geom = """SELECT building_centroid as geom, building_id
    FROM public.nutz_building ;"""

    gdf = gpd.read_postgis(get_building_geom, engine)
    # gdf.to_file('buildings.geojson', driver="GeoJSON")
    # dict
    # object_data = dict(zip(gdf["building_id"], gdf['geom']))

    return gdf

# ndvi_chunk()
# with open('ndvi_chunks/600000_650000.pickle', 'rb') as handle:
#     b = pickle.load(handle)
#
# print(b)

