import os
from sqlalchemy import create_engine, text

engine = create_engine(f'postgresql://postgres:postgres@localhost:5432/postgres')
conn = engine.connect()
conn.execution_options(isolation_level="AUTOCOMMIT")
roof_dir = r'D:\New folder\MT\МАНАП\МАНАП'

nutz_dir = os.listdir(roof_dir)

for nutz in nutz_dir:
    full_path = os.path.join(roof_dir, nutz)
    roof_types = os.listdir(full_path)
    for roof_type in roof_types:
        roof_type_dir = os.path.join(full_path, roof_type)
        building_ids = os.listdir(roof_type_dir)
        print(roof_type.lower())
        for building_id in building_ids:
            building_id = building_id.split('.')[0].split('_')[0]
            print(building_id)
            print(roof_type.lower())
            lower_roof_type = roof_type.lower()
            sql_query = f"""UPDATE public.nutz_building SET roof_type = '{lower_roof_type}' WHERE building_id = {building_id}"""
            conn.execute(text(sql_query))
