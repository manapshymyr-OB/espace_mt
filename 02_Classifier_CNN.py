import time
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import Dropout


# Define the learning rate schedule function
def lr_schedule(epoch):
    """Learning Rate Schedule"""
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

# Database connection
engine = create_engine('postgresql://postgres:postgres@localhost:5432/postgres')

# Data Import
buiilding_data_sql = """
SELECT 
    building_id, 
    --category_id,
    nutz_division_id,
    vv_desc_mean, vh_desc_mean, vv_asc_mean, vh_asc_mean, 
    building_area, building_height_wfs, ndvi_mean, ndvi_min, ndvi_max, 
    parking_large_count, parking_small_count, parking_medium_count,
    parking_large_closest_distance, parking_small_closest_distance, 
    parking_medium_closest_distance, motorway_closest_distance,
    primary_closest_distance, secondary_closest_distance, teritary_closest_distance,
    trunk_closest_distance, perimeter, circularcompactness, longestaxislength, elongation, 
    convexity, orientation, corners, sharedwall, airport_closest_distance, 
    railway_closest_distance
FROM public.nutz_building;
"""
df = pd.read_sql(buiilding_data_sql, engine)

# Get features and labels
features = df.iloc[:, 2:]
category_label = df.iloc[:, 1]
category_labels = df['nutz_division_id'].unique().tolist()

# Print data shapes for verification
print('Training Features Shape:', features.shape)
print('Category Labels Shape:', category_label.shape)
print('Number of Unique Categories:', len(category_labels))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, category_label, train_size=0.5, test_size=0.5, random_state=42, stratify=category_label)

print('Training Features Shape:', X_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', X_test.shape)
print('Testing Labels Shape:', y_test.shape)

# Standardization
sc = StandardScaler()
X_train_norm = sc.fit_transform(X_train)
X_test_norm = sc.transform(X_test)

# One-hot encode the labels
num_classes = len(category_labels)
y_train_one_hot = to_categorical(y_train, num_classes=num_classes)
y_test_one_hot = to_categorical(y_test, num_classes=num_classes)

# Model Definition
n_dimension = X_train.shape[1]
print(f"Input dimension (n_dimension): {n_dimension}")

opt = Nadam(learning_rate=lr_schedule(0), beta_1=0.9, beta_2=0.999)

model = Sequential()
model.add(Input(shape=(n_dimension,)))
model.add(Dense(64, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

model.summary()

# class_weights = compute_class_weight('balanced', classes=np.unique(category_label), y=category_label)
# class_weight_dict = dict(enumerate(class_weights))
# print(class_weight_dict)

# Fit the model
model.fit(X_train_norm, y_train_one_hot,
          batch_size=128,
          epochs=150,
          verbose=1,
          validation_data=(X_test_norm, y_test_one_hot))
          # class_weight=class_weight_dict)

print("Model is ready!")

# Evaluation
y_pred_probs = model.predict(X_test_norm)
y_pred = np.argmax(y_pred_probs, axis=1)

# Create the confusion matrix and classification report
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
