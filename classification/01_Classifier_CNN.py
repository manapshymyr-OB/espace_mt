import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report, \
    roc_curve, auc
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.utils import to_categorical


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


# model name
model_name = 'CNN'
# version, to work with only vector, raster features
version = 'subsample_vector'
# level
level = '1'
# dir to save plots
save_dir = r'../results/SVC/'
# create dir if not exists
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

df = pd.read_pickle(r'../datasets/ml_dataset_subsample')
# Get features and labels
features = df.iloc[:, 6:]
category_label = df.iloc[:, 4]
category_labels = df['nutz_division_id'].unique().tolist()

# mapping of categories with ids
category_labels_dict = {0: "Disposal",
                        1: "Radio and telecommunications system",
                        2: "Trade and services",
                        3: "Industry and commerce",
                        4: "Wastewater treatment plant",
                        5: "Power plant",
                        6: "Storage area",
                        7: "Substation",
                        8: "Supply system",
                        9: "Waterworks"}


category_labels_list = list(category_labels_dict.values())
# Saving feature names for later use
feature_list = list(features.columns)


# Print data shapes for verification
print('Training Features Shape:', features.shape)
print('Category Labels Shape:', category_label.shape)
print('Number of Unique Categories:', len(category_labels))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, category_label, train_size=0.8, random_state=42,
                                                    stratify=category_label)

print('Training Features Shape:', X_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', X_test.shape)
print('Testing Labels Shape:', y_test.shape)

# Standardization
sc = StandardScaler()
X_train_norm = sc.fit_transform(X_train)
X_test_norm = sc.transform(X_test)

# Apply SMOTE to the training data
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_norm, y_train = smote.fit_resample(X_train_norm, y_train)
# Print category distribution after SMOTE
unique_categories, counts = np.unique(y_train, return_counts=True)
for category, count in zip(unique_categories, counts):
    print(f"Category {category}: {count} rows")

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
model.add(Dense(128, kernel_initializer='normal', activation='relu'))
model.add(Dense(64, kernel_initializer='normal', activation='relu'))
model.add(Dense(32, kernel_initializer='normal', activation='relu'))
model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

model.summary()

lr_scheduler = LearningRateScheduler(lr_schedule)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Fit the model
history = model.fit(X_train_norm, y_train_one_hot,
                    batch_size=128,
                    epochs=150,
                    verbose=1,
                    validation_data=(X_test_norm, y_test_one_hot),
                    callbacks=[lr_scheduler])

print("Model is ready!")

# Evaluation
y_pred_probs = model.predict(X_test_norm)
y_pred = np.argmax(y_pred_probs, axis=1)

# Create the confusion matrix
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average=None)
recall = recall_score(y_test, y_pred, average=None)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
conf_matrix = confusion_matrix(y_test, y_pred, normalize='true')
report_adjusted = classification_report(y_test, y_pred, target_names=category_labels_list, output_dict=True)


kappa_score = cohen_kappa_score(y_test, y_pred)
recall_avg = recall_score(y_test, y_pred, average='weighted')
precision_avg = precision_score(y_test, y_pred, average='weighted')
f1 = (2 * precision_avg * recall_avg) / (precision_avg + recall_avg)
print("Cohen's Kappa Score: {:.4f}".format(kappa_score))

# Create a DataFrame with both scores
df = pd.DataFrame({
    'Metric': ["Cohen's Kappa Score", 'Accuracy Score', 'recall_avg', 'precision_avg', 'f1'],
    'Score': [kappa_score, accuracy, recall_avg, precision_avg, f1]
})
# Save the DataFrame to an Excel file
df.to_excel(os.path.join(save_dir, f'metrics_scores {model_name} {version} {level}.xlsx'), index=False)



sns.heatmap(conf_matrix, annot=True, fmt=".2f", cmap="Blues", xticklabels=category_labels_list,
            yticklabels=category_labels_list, )
plt.title(f'Confusion Matrix of {model_name}')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.savefig(os.path.join(save_dir, f'Confusion Matrix {model_name} {version} {level}.png'))
plt.show()

# Extract class-wise metrics
class_labels = category_labels_list
class_recalls = [report_adjusted[label]['recall'] for label in class_labels]
class_precisions = [report_adjusted[label]['precision'] for label in class_labels]
class_f1_scores = [report_adjusted[label]['f1-score'] for label in class_labels]

# Plotting the metrics
fig, ax = plt.subplots(figsize=(12, 8))  # Increase figure size for better spacing
ax.grid(True, which='major', linestyle='-', linewidth=0.5)
ax.minorticks_on()

bar_width = 0.3
index = np.arange(len(class_labels))

bars1 = ax.bar(index - bar_width, class_recalls, bar_width, label='Recall', color='C0', alpha=1)
bars2 = ax.bar(index, class_precisions, bar_width, label='Precision', color='green')
bars3 = ax.bar(index + bar_width, class_f1_scores, bar_width, label='F1 Score', color='red')

# Adding text labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, round(yval, 2), ha='center', va='bottom')

# Setting title and labels
plt.title(f'Class-wise Metrics {model_name}')
plt.xlabel('Classes')
plt.ylabel('Score')
plt.ylim(0, 1.0)  # to make sure text fits

# Rotate x-axis labels if needed
ax.set_xticks(index)
ax.set_xticklabels(class_labels, rotation=45, ha='right')
plt.legend()
plt.tight_layout()  # Adjust layout to make room for the x-axis labels
plt.savefig(os.path.join(save_dir, f'Class-wise Metrics {model_name} {version} {level}.png'))
# Display the plot
plt.show()


num_classes = len(category_labels)
y_train_one_hot = to_categorical(y_train, num_classes=num_classes)
y_test_one_hot = to_categorical(y_test, num_classes=num_classes)
y_pred_probs = to_categorical(y_pred, num_classes=num_classes)

category_labels_dict_keys = list(category_labels_dict.keys())

# Compute ROC curve and ROC area for each class
fpr = {}
tpr = {}
roc_auc = {}
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_one_hot[:, i], y_pred_probs[:, i], pos_label=1)
    roc_auc[i] = auc(fpr[i], tpr[i])

# # Compute micro-average ROC curve and ROC area
# fpr["micro"], tpr["micro"], _ = roc_curve(y_test_one_hot.ravel(), y_pred_probs.ravel())
# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC curve
plt.figure(figsize=(10, 8))  # Adjust the figsize as needed
# plt.plot(fpr["micro"], tpr["micro"],
#          label='micro-average ROC curve (area = {0:0.2f})'
#                ''.format(roc_auc["micro"]))
for i in range(num_classes):
    plt.plot(fpr[i], tpr[i],
             label='ROC curve of class {0} (area = {1:0.2f})'
                   ''.format(category_labels_list[i], roc_auc[i]))
plt.grid(True)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig(os.path.join(save_dir, f'ROC {model_name} {version} {level}.png'))
# model.save_weights(f"{folder_dir}model_weights_{version}.weights.h5")
plt.show()

# Plot the precision-recall curve for each class
plt.figure(figsize=(15, 10))

# Iterate over each class
for i in range(num_classes):
    precision, recall, _ = precision_recall_curve(y_test_one_hot[:, i], y_pred_probs[:, i])
    average_precision = average_precision_score(y_test_one_hot[:, i], y_pred_probs[:, i])

    # Plot the curve
    plt.plot(recall, precision, lw=2, label=f'Class {category_labels_dict[i]} (AP = {average_precision:.2f})')
plt.grid(True)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Each Class')
plt.legend(loc='best')
plt.grid(True)
plt.savefig(os.path.join(save_dir, f'Precision-Recall Curve for Each Class {model_name} {version} {level}.png'))
plt.show()