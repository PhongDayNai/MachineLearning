import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, PowerTransformer, LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

data = pd.read_csv('sleep_data.csv')
data['Sleep Disorder'].fillna(data['Sleep Disorder'].mode()[0], inplace=True)
print(data.isnull().sum())

label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender'])
data['Occupation'] = label_encoder.fit_transform(data['Occupation'])
data['BMI Category'] = label_encoder.fit_transform(data['BMI Category'])
data[['Systolic', 'Diastolic']] = data['Blood Pressure'].str.split('/', expand=True)
data['Systolic'] = pd.to_numeric(data['Systolic'])
data['Diastolic'] = pd.to_numeric(data['Diastolic'])

power_transformer = PowerTransformer(method='yeo-johnson')
left_skewed_vars = ['Gender']
data[left_skewed_vars] = power_transformer.fit_transform(data[left_skewed_vars])
right_skewed_vars = ['Quality of Sleep',  'Heart Rate']
data[right_skewed_vars] = power_transformer.fit_transform(data[right_skewed_vars])

multimodal_vars = ['Occupation', 'Systolic', 'Diastolic', 'BMI Category']

for var in multimodal_vars:
    kmeans = KMeans(n_clusters=3, random_state=42)
    data[f'{var}_cluster'] = kmeans.fit_predict(data[[var]])

    gmm = GaussianMixture(n_components=3, random_state=42)
    data[f'{var}_gmm'] = gmm.fit_predict(data[[var]])

print("Columns in the dataset:\n", data.columns)
print("\nSample data:\n", data.head())

label_encoder = LabelEncoder()
categorical_columns = ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']
for col in categorical_columns:
    if col in data.columns:
        data[col] = label_encoder.fit_transform(data[col])

data[['Systolic', 'Diastolic']] = data['Blood Pressure'].str.split('/', expand=True).astype(float)
data = data.drop('Blood Pressure', axis=1)
data['Systolic'] = pd.to_numeric(data['Systolic'], errors='coerce')
data['Diastolic'] = pd.to_numeric(data['Diastolic'], errors='coerce')

standard_scaler = StandardScaler()
standardized_data = pd.DataFrame(standard_scaler.fit_transform(data), columns=data.columns)
print("\nStandardized Data:\n", standardized_data.describe())

label_encoders = {}
for column in ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

if 'Person ID' in data.columns:
    data = data.drop('Person ID', axis=1)

X = data.drop('Sleep Disorder', axis=1)
y = data['Sleep Disorder']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

y_train_categorical = to_categorical(y_train)
y_val_categorical = to_categorical(y_val)
y_test_categorical = to_categorical(y_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

model = Sequential()

model.add(Dense(units=64, activation='relu', input_dim=X_train_scaled.shape[1]))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=y_train_categorical.shape[1], activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train_scaled, y_train_categorical, epochs=30, batch_size=32, validation_data=(X_val_scaled, y_val_categorical))

y_train_pred_nn = model.predict(X_train_scaled)
y_val_pred_nn = model.predict(X_val_scaled)
y_test_pred_nn = model.predict(X_test_scaled)

y_train_pred_classes = y_train_pred_nn.argmax(axis=1)
y_val_pred_classes = y_val_pred_nn.argmax(axis=1)
y_test_pred_classes = y_test_pred_nn.argmax(axis=1)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

train_accuracy = accuracy_score(y_train, y_train_pred_classes)
val_accuracy = accuracy_score(y_val, y_val_pred_classes)
test_accuracy = accuracy_score(y_test, y_test_pred_classes)

train_precision = precision_score(y_train, y_train_pred_classes, average='weighted')
val_precision = precision_score(y_val, y_val_pred_classes, average='weighted')
test_precision = precision_score(y_test, y_test_pred_classes, average='weighted')

train_recall = recall_score(y_train, y_train_pred_classes, average='weighted')
val_recall = recall_score(y_val, y_val_pred_classes, average='weighted')
test_recall = recall_score(y_test, y_test_pred_classes, average='weighted')

train_f1 = f1_score(y_train, y_train_pred_classes, average='weighted')
val_f1 = f1_score(y_val, y_val_pred_classes, average='weighted')
test_f1 = f1_score(y_test, y_test_pred_classes, average='weighted')

results = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-score'],
    'Train': [train_accuracy, train_precision, train_recall, train_f1],
    'Validation': [val_accuracy, val_precision, val_recall, val_f1],
    'Test': [test_accuracy, test_precision, test_recall, test_f1]
}

results_df = pd.DataFrame(results)

print(results_df)

metrics = results_df['Metric']
train_scores = results_df['Train']
val_scores = results_df['Validation']
test_scores = results_df['Test']

plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
plt.plot(metrics, train_scores, marker='o', label='Train', color='blue')
plt.plot(metrics, val_scores, marker='o', label='Validation', color='orange')
plt.plot(metrics, test_scores, marker='o', label='Test', color='green')
plt.title('Accuracy by Dataset')
plt.xlabel('Metrics')
plt.ylabel('Scores')
plt.ylim(0, 1) 
plt.grid()
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(metrics, train_scores, marker='o', label='Train', color='blue')
plt.plot(metrics, val_scores, marker='o', label='Validation', color='orange')
plt.plot(metrics, test_scores, marker='o', label='Test', color='green')
plt.title('Precision by Dataset')
plt.xlabel('Metrics')
plt.ylabel('Scores')
plt.ylim(0, 1)
plt.grid()
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(metrics, train_scores, marker='o', label='Train', color='blue')
plt.plot(metrics, val_scores, marker='o', label='Validation', color='orange')
plt.plot(metrics, test_scores, marker='o', label='Test', color='green')
plt.title('Recall by Dataset')
plt.xlabel('Metrics')
plt.ylabel('Scores')
plt.ylim(0, 1)
plt.grid()
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(metrics, train_scores, marker='o', label='Train', color='blue')
plt.plot(metrics, val_scores, marker='o', label='Validation', color='orange')
plt.plot(metrics, test_scores, marker='o', label='Test', color='green')
plt.title('F1-score by Dataset')
plt.xlabel('Metrics')
plt.ylabel('Scores')
plt.ylim(0, 1)
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()

model.save('model_neural_network.h5')
