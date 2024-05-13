from sqlalchemy import create_engine, text

engine = create_engine('postgresql://postgres:postgres@localhost:5432/postgres')

import pandas as pd


buiilding_data_sql = 'select distinct building_id from nutz_building '

df = pd.read_sql(buiilding_data_sql, engine)
print(df)
building_ids = df['building_id'].tolist()
print(building_ids)

data_len = len(building_ids)
for building_id in building_ids:
    sql_query = f"""
    INSERT INTO tertiary (building_id, osm_id, building_centroid, geom, dist)
    select nb.building_id, highways.osm_id, nb.building_centroid_local, highways.geom_local,   st_distance(highways.geom_local,nb.building_centroid_local ) dist 
     from nutz_building nb   
     CROSS JOIN lateral (
     select bl.osm_id, bl.geom_local,  bl.geom_local <->nb.building_centroid_local as dist from bayern_line bl 
     where bl.highway = 'tertiary'
     order by dist
	limit 1     
     ) highways where nb.building_id = {building_id} """

    with engine.begin() as conn:  # Ensures the connection is properly closed after operation
        conn.execute(text(sql_query))
    # print(sql_query)
    data_len -= 1
    print(data_len)


