# -------------------------------
# author: Hao Li, leebobgiser316@gmail.com
# data: 05.10.2023
# -------------------------------

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
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
rf = SVC(C=4, kernel='rbf', gamma=0.25)
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



