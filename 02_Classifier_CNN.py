# -------------------------------
# author: Hao Li, leebobgiser316@gmail.com
# data: 05.10.2023
# -------------------------------
import time

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from keras.layers import Input
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import geopandas as gpd
import numpy as np
from tensorflow.keras.optimizers import Nadam
from sqlalchemy import create_engine
import pandas as pd

engine = create_engine('postgresql://postgres:postgres@localhost:5432/postgres')






###########  Hyperparameters #############

def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 0.0001
    if epoch > 400:
        lr *= 0.5e-3
    elif epoch > 300:
        lr *= 0.5e-3
    elif epoch > 200:
        lr *= 1e-3
    elif epoch > 50:
        lr *= 1e-2
    elif epoch > 20:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


###########  Data Import #############
# # input the feature and height information for all buildings in Heidelberg
# attribute = r"./data/korea_buildings_with_feature.geojson"
#
# # read and shuffle all data into geopandas dataframe
# df = gpd.read_file(attribute)
# # df = shuffle(df)
# nutz_division_id,
buiilding_data_sql = """SELECT 
building_id, 
category_id,
vv_desc_mean, vh_desc_mean, vv_asc_mean, vh_asc_mean, 
building_area, building_height_wfs, ndvi_mean, ndvi_min, ndvi_max, 
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
category_labels = df['category_id'].unique().tolist()
print(len(category_label))
# Saving feature names for later use
feature_list = list(features.columns)


# convert features to array
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, category_label, train_size=0.5, test_size=0.5, random_state=42)

print('Training Features Shape:', X_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', X_test.shape)
print('Testing Labels Shape:', y_test.shape)
print(y_train)

## Standardization
sc = StandardScaler()
X_train_norm = sc.fit_transform(X_train)
X_test_norm = sc.transform(X_test)

#################  CNN Model ###########################

# # Instantiate model with 1000 decision trees
n_dimension = X_train.shape[1]

print(X_train.shape)
print(n_dimension)
print(f'here - {lr_schedule(0)}')
opt = Nadam(learning_rate=lr_schedule(0), beta_1=0.9, beta_2=0.999)
# create model
model = Sequential()
model.add(Dense( Input(shape=(n_dimension,)), kernel_initializer='normal', activation='relu'))
model.add(Dense(20, kernel_initializer='normal', activation='relu'))
model.add(Dense(4, kernel_initializer='normal'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer=opt)

model.summary()

## fit the model
model.fit(X_train_norm, y_train,
          batch_size=128,
          epochs=150,
          verbose=1,
          validation_data=(X_test_norm, y_test))

print("model is ready!")

#################  Evaluation ###########################
# Use the forest's predict method on the test data
# y_pred = model.predict_classes(X_test_norm, verbose=0)
y_pred = np.argmax(model.predict(X_test_norm),axis=1)
print(y_pred)
# y_pred = np.transpose(y_pred)[0]
# print(y_pred)

# print(y_pred)
# assert 0



# Create the confusion matrix
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))


