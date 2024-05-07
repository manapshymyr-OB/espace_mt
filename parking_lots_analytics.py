import pandas as pd
from sqlalchemy import create_engine
import jenkspy
engine = create_engine(f'postgresql://postgres:postgres@localhost:5432/postgres')


df = pd.read_sql_table('parking_lots', engine)
parking_lots_area = df['parking_area'].to_list()
# print(parking_lots_area)

dd = jenkspy.jenks_breaks(parking_lots_area, n_classes=3)
print(dd)
# 5644.185838402249, 110551.18147586231, 392276.7805249393]