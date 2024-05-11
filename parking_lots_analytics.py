import pandas as pd
from sqlalchemy import create_engine
import jenkspy
engine = create_engine(f'postgresql://postgres:postgres@localhost:5432/postgres')

sql_query = "select  * from parkin_lots_cleaned where geom_area >20"
df = pd.read_sql(sql_query, engine)
parking_lots_area = df['geom_area'].to_list()
# print(parking_lots_area)

dd = jenkspy.jenks_breaks(parking_lots_area, n_classes=4)
print(dd)
# 5644.185838402249, 110551.18147586231, 392276.7805249393]