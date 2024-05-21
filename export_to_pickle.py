import psycopg2
import base64
from sqlalchemy import create_engine, text
import geopandas as gpd
import pickle
import pandas as pd
engine = create_engine(f'postgresql://postgres:postgres@localhost:5432/postgres')

df = pd.read_sql_table('intersections_building', engine)
print(df)
with open(f'intersection', 'wb') as handle:
    pickle.dump(df, handle)