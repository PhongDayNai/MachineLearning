import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, PowerTransformer, LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns
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

X = data.drop('Sleep Disorder', axis=1)
y = data['Sleep Disorder']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

lr_model = LinearRegression()

lr_model.fit(X_train, y_train)

y_train_pred = lr_model.predict(X_train)
y_validation_pred = lr_model.predict(X_val)
y_test_pred = lr_model.predict(X_test)

r2_train, rmse_train, nse_train, mae_train = evaluate_model(y_train, y_train_pred)
r2_val, rmse_val, nse_val, mae_val = evaluate_model(y_val, y_validation_pred)
r2_test, rmse_test, nse_test, mae_test = evaluate_model(y_test, y_test_pred)

results = {
    'Metric': ['LinearRegression-R²', 'LinearRegression-RMSE', 'LinearRegression-NSE', 'LinearRegression-MAE'],
    'Train': [r2_train, rmse_train, nse_train, mae_train],
    'Validation': [r2_val, rmse_val, nse_val, mae_val],
    'Test': [r2_test, rmse_test, nse_test, mae_test]
}

results_df = pd.DataFrame(results)

print(results_df)

def plot_parabola(x, y_true, y_pred, title):
    poly_true = np.poly1d(np.polyfit(x, y_true, 2))
    poly_pred = np.poly1d(np.polyfit(x, y_pred, 2))

    x_smooth = np.linspace(x.min(), x.max(), 500)

    plt.figure(figsize=(12, 6))
    plt.plot(x_smooth, poly_true(x_smooth), label='True Values (Parabola)', color='blue')
    plt.plot(x_smooth, poly_pred(x_smooth), label='Predicted Values (Parabola)', color='orange')
    plt.legend()
    plt.xlabel('Sample Index')
    plt.ylabel('Price')
    plt.title(title)
    plt.show()

x_train_range = np.arange(len(y_train))
x_val_range = np.arange(len(y_val))
x_test_range = np.arange(len(y_test))

plot_parabola(x_train_range, y_train, y_train_pred, 'Train Set: True vs Predicted Values (Parabola)')
plot_parabola(x_val_range, y_val, y_validation_pred, 'Validation Set: True vs Predicted Values (Parabola)')
plot_parabola(x_test_range, y_test, y_test_pred, 'Test Set: True vs Predicted Values (Parabola)')

with open('label_encoders_linear.pkl', 'wb') as file:
    pickle.dump(label_encoders, file)
with open('model_linear.pkl', 'wb') as file:
    pickle.dump(lr_model, file)
with open('scaler_reg.pkl', 'wb') as scaler_file:
    pickle.dump(standard_scaler, scaler_file)