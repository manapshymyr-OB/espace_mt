# -------------------------------
# author: Hao Li, leebobgiser316@gmail.com
# data: 05.10.2023
# -------------------------------

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import geopandas as gpd
from sqlalchemy import create_engine
import pandas as pd

###########  Data Import #############
engine = create_engine('postgresql://postgres:postgres@localhost:5432/postgres')

###########  Data Import #############
# input the feature and height information for all buildings in Heidelberg
buiilding_data_sql = """SELECT 
building_id, 
category_id,
vv_desc_mean, vh_desc_mean, vv_asc_mean, vh_asc_mean, 
building_area, COALESCE(building_height_wfs, -1) as building_height_wfs, ndvi_mean, ndvi_min, ndvi_max, 
 parking_large_count, parking_small_count, parking_medium_count,
  parking_large_closest_distance, parking_small_closest_distance, 
  parking_medium_closest_distance, motorway_closest_distance,
   primary_closest_distance, secondary_closest_distance, teritary_closest_distance,
    trunk_closest_distance, perimeter, circularcompactness, longestaxislength, elongation, 
    convexity, orientation, corners, sharedwall, airport_closest_distance, 
      railway_closest_distance
FROM public.nutz_building;"""

df = pd.read_sql(buiilding_data_sql, engine)
print(df)
# get features and labels
features = df.iloc[:, 2:]
category_label = df.iloc[:, 1]

# Saving feature names for later use
feature_list = list(features.columns)

print('here')

# convert features to array
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, category_label, test_size=0.7, random_state=42)

print('Training Features Shape:', X_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', X_test.shape)
print('Testing Labels Shape:', y_test.shape)
print(y_train)

## Standardization
sc = StandardScaler()
X_train_norm = sc.fit_transform(X_train)
X_test_norm = sc.transform(X_test)

#################  RF Model ###########################
# Instantiate model with 1000 decision trees
rf = SVC(C=4, kernel='rbf', gamma=0.25)
rf.fit(X_train_norm, y_train)
print("model is ready!")



#################  Evaluation ###########################
# Use the forest's predict method on the test data
y_pred = rf.predict(X_test_norm)


# Create the confusion matrix
accuracy = balanced_accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='samples')
recall = recall_score(y_test, y_pred, average='samples')
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))



