# -------------------------------
# author: Hao Li, leebobgiser316@gmail.com
# data: 05.10.2023
# -------------------------------

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import geopandas as gpd


###########  Data Import #############
# input the feature and height information for all buildings in Heidelberg
attribute = "./data/korea_buildings_with_feature.geojson"

# read and shuffle all data into geopandas dataframe
df = gpd.read_file(attribute)
# df = shuffle(df)

# get features and labels
features = df.iloc[:, 3:10]
warehouse_label = df.iloc[:, 10]

# Saving feature names for later use
feature_list = list(features.columns)

print()

# convert features to array
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, warehouse_label, test_size=0.5, random_state=42)

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
rf = RandomForestClassifier(n_estimators=1000, random_state=42, verbose=1, n_jobs=6, min_samples_split=5)
rf.fit(X_train_norm, y_train)
print("model is ready!")



#################  Evaluation ###########################
# Use the forest's predict method on the test data
y_pred = rf.predict(X_test_norm)


# Create the confusion matrix
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))




# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
# Print out the feature and importances
print('Print Top10 important features:')
[print('Variable: {:5} Importance: {}'.format(*pair)) for pair in feature_importances[0:10]]


# Calculate the SHAP feature importance

# DF, based on which importance is checked
X_importance = X_test_norm

# Explain model predictions using shap library:
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_importance)
print(shap_values)
#
# # Plot summary_plot
# shap.summary_plot(shap_values, X_importance)
#
# # Plot summary_plot as barplot:
# shap.summary_plot(shap_values, X_importance, plot_type='bar')

