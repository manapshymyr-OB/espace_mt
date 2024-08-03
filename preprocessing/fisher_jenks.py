"""
To categorize parking spaces based on their area using Fisher-Jenks algorithm
"""
import jenkspy
import pandas as pd

from utils import engine

sql_query = "select  * from parkin_lots_cleaned where geom_area >20"
df = pd.read_sql(sql_query, engine)
parking_lots_area = df['geom_area'].to_list()
# print(parking_lots_area)

thresholds = jenkspy.jenks_breaks(parking_lots_area, n_classes=4)
print(thresholds)
