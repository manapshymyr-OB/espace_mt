import psycopg2
import base64
from sqlalchemy import create_engine, text
import geopandas as gpd
import pandas as pd
engine = create_engine(f'postgresql://postgres:postgres@localhost:5432/postgres')
connection = engine.connect()
connection.execution_options(isolation_level="AUTOCOMMIT")
sql_intersections_all = """select 
building_id, bbox
from public.nutz_building_2 nb



left join raster.rast_envelope re on st_intersects(re.bbox_rast, nb.building_envelope_geom)
where png_done is null
"""
gdf_all = pd.read_sql(sql_intersections_all, engine)
sql_intersections = """select building_id from public.nutz_building_2 nb
"""

gdf = pd.read_sql(sql_intersections, engine)
building_ids = gdf['building_id'].unique().tolist()

# print(building_ids)
all_data = len(building_ids)
for id in building_ids:

    id_data = gdf_all[gdf_all['building_id'] == id]
    id_data['bbox'] = id_data['bbox'].str.replace('.tif', '')
    names = id_data['bbox'].unique().tolist()
    # print(names)
    tiff_name = '_'.join(names)
    png_query = f"""with complete_raster as (
    select st_union(rast) rast from raster.dom d where d.filename in (select 
    bbox as filename
    from public.nutz_building_2 nb
    left join raster.rast_envelope re on st_intersects(re.bbox_rast, nb.building_envelope_geom) where nb.building_id ={id}
    )
    )
    
    select encode( ST_AsPNG(st_clip(c.rast, st_buffer(nb.building_envelope_geom, 2), TRUE)), 'base64') from complete_raster c join public.nutz_building_2 nb on ST_Intersects(c.rast, nb.building_envelope_geom ) 
    where nb.building_id = {id}"""
    result = connection.execute(text(png_query))
    png_base64 = result.scalar()

# Decode the base64 string to binary
    png_binary = base64.b64decode(png_base64)

    # Write the binary data to a file
    with open(f'png_data/{id}_{tiff_name}.png', 'wb') as file:
        file.write(png_binary)

    update_qry = f"update public.nutz_building_2 set png_done = 'done' where building_id = {id}"
    connection.execute(text(update_qry))