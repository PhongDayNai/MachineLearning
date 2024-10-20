import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, PowerTransformer, LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
import pickle

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
def create_nn_model(input_dim):
    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_dim=input_dim))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

nn_model = create_nn_model(X_train_scaled.shape[1])
nn_model.fit(X_train_scaled, y_train, epochs=30, batch_size=32, validation_data=(X_val_scaled, y_val))

y_train_nn_pred = (nn_model.predict(X_train_scaled) > 0.5).astype("int32")
y_val_nn_pred = (nn_model.predict(X_val_scaled) > 0.5).astype("int32")
y_test_nn_pred = (nn_model.predict(X_test_scaled) > 0.5).astype("int32")

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_train_lr_pred = lr_model.predict(X_train)
y_val_lr_pred = lr_model.predict(X_val)
y_test_lr_pred = lr_model.predict(X_test)

lasso_model = Lasso(alpha=0.021)
lasso_model.fit(X_train, y_train)
y_train_lasso_pred = lasso_model.predict(X_train)
y_val_lasso_pred = lasso_model.predict(X_val)
y_test_lasso_pred = lasso_model.predict(X_test)

train_preds = pd.DataFrame({
    'NN': y_train_nn_pred.flatten(),
    'Linear': y_train_lr_pred,
    'Lasso': y_train_lasso_pred
})

val_preds = pd.DataFrame({
    'NN': y_val_nn_pred.flatten(),
    'Linear': y_val_lr_pred,
    'Lasso': y_val_lasso_pred
})

test_preds = pd.DataFrame({
    'NN': y_test_nn_pred.flatten(),
    'Linear': y_test_lr_pred,
    'Lasso': y_test_lasso_pred
})

meta_model = LinearRegression()
meta_model.fit(train_preds, y_train)

final_train_pred = meta_model.predict(train_preds)
final_val_pred = meta_model.predict(val_preds)
final_test_pred = meta_model.predict(test_preds)

def evaluate_classification(y_true, y_pred):
    accuracy = accuracy_score(y_true, (y_pred > 0.5).astype("int32"))
    precision = precision_score(y_true, (y_pred > 0.5).astype("int32"))
    recall = recall_score(y_true, (y_pred > 0.5).astype("int32"))
    f1 = f1_score(y_true, (y_pred > 0.5).astype("int32"))
    return accuracy, precision, recall, f1

train_accuracy, train_precision, train_recall, train_f1 = evaluate_classification(y_train, final_train_pred)
val_accuracy, val_precision, val_recall, val_f1 = evaluate_classification(y_val, final_val_pred)
test_accuracy, test_precision, test_recall, test_f1 = evaluate_classification(y_test, final_test_pred)

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
plt.title('Accuracy, Precision, Recall, and F1-score by Dataset')
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

with open('model_stacking.pkl', 'wb') as file:
    pickle.dump(meta_model, file)