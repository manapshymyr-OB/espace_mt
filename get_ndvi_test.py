import pandas as pd
from sqlalchemy import create_engine

engine = create_engine(f'postgresql://postgres:postgres@localhost:5432/postgres')

df = pd.read_sql('ndvi_ver2', engine)
print(df)

grouped = df.groupby('A')


# for name, group in grouped:
