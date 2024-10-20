import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Lasso
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, PowerTransformer, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
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

def nse(y_true, y_pred):
    "Tính toán chỉ số Nash-Sutcliffe Efficiency (NSE)."
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (numerator / denominator)

def evaluate_model(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nse_value = nse(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return r2, rmse, nse_value, mae

label_encoders = {}
for column in ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

if 'Person ID' in data.columns:
    data = data.drop('Person ID', axis=1)

if 'Person ID' in data.columns:
    data = data.drop('Person ID', axis=1)

X = data.drop('Sleep Disorder', axis=1)
y = data['Sleep Disorder']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

with open('label_encoders_linear.pkl', 'rb') as file:
    label_encoders_linear = pickle.load(file)
with open('label_encoders_lasso.pkl', 'rb') as file:
    label_encoders_lasso = pickle.load(file)
with open('model_linear.pkl', 'rb') as file:
    model_linear = pickle.load(file)
with open('model_lasso.pkl', 'rb') as file:
    model_lasso = pickle.load(file)

model_neural_network = keras.models.load_model('model_neural_network.h5')

y_train_linear_pred = model_linear.predict(X_train)
y_train_lasso_pred = model_lasso.predict(X_train)
y_train_nn_pred = model_neural_network.predict(X_train)

y_val_linear_pred = model_linear.predict(X_val)
y_val_lasso_pred = model_lasso.predict(X_val)
y_val_nn_pred = model_neural_network.predict(X_val)

y_test_linear_pred = model_linear.predict(X_test)
y_test_lasso_pred = model_lasso.predict(X_test)
y_test_nn_pred = model_neural_network.predict(X_test)

X_train_stacking = np.column_stack((y_train_linear_pred, y_train_lasso_pred, y_train_nn_pred))
X_val_stacking = np.column_stack((y_val_linear_pred, y_val_lasso_pred, y_val_nn_pred))
X_test_stacking = np.column_stack((y_test_linear_pred, y_test_lasso_pred, y_test_nn_pred))

stacked_model = LinearRegression()
stacked_model.fit(X_train_stacking, y_train)

y_val_stacking_pred = stacked_model.predict(X_val_stacking)
y_test_stacking_pred = stacked_model.predict(X_test_stacking)

r2_val_stacking = r2_score(y_val, y_val_stacking_pred)
rmse_val_stacking = np.sqrt(mean_squared_error(y_val, y_val_stacking_pred))

r2_test_stacking = r2_score(y_test, y_test_stacking_pred)
rmse_test_stacking = np.sqrt(mean_squared_error(y_test, y_test_stacking_pred))

print(f'Validation R²: {r2_val_stacking}')
print(f'Validation RMSE: {rmse_val_stacking}')
print(f'Test R²: {r2_test_stacking}')
print(f'Test RMSE: {rmse_test_stacking}')

# with open('model_stacking.pkl', 'wb') as file:
#     pickle.dump(stacked_model, file)
