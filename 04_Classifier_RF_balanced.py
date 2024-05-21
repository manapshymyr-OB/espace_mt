# -------------------------------
# author: Hao Li, leebobgiser316@gmail.com
# data: 05.10.2023
# -------------------------------

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import geopandas as gpd
from sqlalchemy import create_engine
import pandas as pd
import shap
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# https://medium.com/analytics-vidhya/a-random-forest-classifier-with-imbalanced-data-7ef4d9ebedb8
engine = create_engine('postgresql://postgres:postgres@localhost:5432/postgres')

###########  Data Import #############
# input the feature and height information for all buildings in Heidelberg
buiilding_data_sql = """SELECT 
building_id, 
category_id,
vv_desc_mean, vh_desc_mean, vv_asc_mean, vh_asc_mean, 
building_area,  COALESCE(building_height_wfs, -1) as building_height_wfs, ndvi_mean, ndvi_min, ndvi_max, 
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


# convert features to array
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, category_label,
                                                    test_size=0.3, random_state=42, stratify=category_label)

print('Training Features Shape:', X_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', X_test.shape)
print('Testing Labels Shape:', y_test.shape)
print(y_train)

## Standardization
sc = StandardScaler()
X_train_norm = sc.fit_transform(X_train)
X_test_norm = sc.transform(X_test)
# Apply SMOTE to the training data
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_norm, y_train)
# Assuming y_train_res is a numpy array
import numpy as np
unique_categories, counts = np.unique(y_train_res, return_counts=True)

# Print the results
for category, count in zip(unique_categories, counts):
    print(f"Category {category}: {count} rows")
print(X_train_norm.shape)
print(X_train_res.shape)
print(y_train_res.shape)
print('stiop')

#################  RF Model ###########################
# Instantiate model with 1000 decision trees
rf = RandomForestClassifier(n_estimators=1000, random_state=42, verbose=1, n_jobs=20, min_samples_split=5)
rf.fit(X_train_res, y_train_res)
print("model is ready!")
# smote = SMOTE(sampling_strategy='not majority')


#################  Evaluation ###########################
# Use the forest's predict method on the test data
y_pred = rf.predict(X_test_norm)


# Create the confusion matrix
accuracy = balanced_accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(classification_report(y_test, y_pred))




# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
# Print out the feature and importances
print('Print Top10 important features:')
[print('Variable: {:5} Importance: {}'.format(*pair)) for pair in feature_importances]


# Calculate the SHAP feature importance

# DF, based on which importance is checked
X_importance = X_test_norm

shap.initjs()
# Explain model predictions using shap library:
explainer = shap.TreeExplainer(rf)
# Enable verbose output
shap_values = explainer.shap_values(X_importance, check_additivity=False)
print(shap_values)




# Plot summary_plot
shap.summary_plot(shap_values, X_importance)

# Plot summary_plot as barplot:
shap.summary_plot(shap_values, X_importance, plot_type='bar')

