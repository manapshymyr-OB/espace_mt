import base64
import os
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from sqlalchemy import text

from utils import engine

# Initialize database connection
connection = engine.connect()


# Define a function to process each building ID
def process_building(id):
    local_conn = engine.connect()
    local_conn.execution_options(isolation_level="AUTOCOMMIT")

    sql_query = f"""select encode( ST_AsPNG(st_clip(c.rast, st_buffer(nb.building_envelope_geom, 2), TRUE)), 'base64') as png_base64
                    from (
                        select st_union(rast) as rast
                        from raster.dom d
                        where d.filename in (
                            select bbox as filename
                            from public.nutz_building_2 nb
                            left join raster.rast_envelope re on st_intersects(re.bbox_rast, nb.building_envelope_geom)
                            where nb.building_id = {id}
                        )
                    ) c
                    join public.nutz_building_2 nb on ST_Intersects(c.rast, nb.building_envelope_geom)
                    where nb.building_id = {id}"""

    result = local_conn.execute(text(sql_query))
    png_base64 = result.scalar()

    if png_base64:
        png_binary = base64.b64decode(png_base64)
        os.makedirs('png_data', exist_ok=True)
        with open(f'png_data/{id}.png', 'wb') as file:
            file.write(png_binary)
        print(f'done {id}')
        update_qry = f"update public.nutz_building_2 set png_done = 'done' where building_id = {id}"
        local_conn.execute(text(update_qry))
        print(f'finished {id}')

    local_conn.close()


# Get all building IDs
sql_intersections = """select distinct building_id from public.nutz_building_2 where png_done is null"""
gdf = pd.read_sql(sql_intersections, engine)
building_ids = gdf['building_id'].tolist()

# Process each building ID in parallel
with ProcessPoolExecutor(max_workers=15) as executor:
    executor.map(process_building, building_ids)

# Close the main connection
connection.close()