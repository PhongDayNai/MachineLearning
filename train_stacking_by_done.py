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

# 1. Tải dữ liệu
data = pd.read_csv('sleep_data.csv')

# Điền giá trị thiếu của cột 'Sleep Disorder' bằng mode
data['Sleep Disorder'].fillna(data['Sleep Disorder'].mode()[0], inplace=True)

# Kiểm tra lại xem còn giá trị thiếu hay không
print(data.isnull().sum())

# Chuyển đổi các biến phân loại thành số (vd: 'Gender', 'Occupation', 'BMI Category')
label_encoder = LabelEncoder()

# Giả sử dữ liệu có cột 'Gender' với các giá trị 'Male' và 'Female'
data['Gender'] = label_encoder.fit_transform(data['Gender'])

# Chuyển đổi cột 'Occupation'
data['Occupation'] = label_encoder.fit_transform(data['Occupation'])

# Chuyển đổi cột 'BMI Category'
data['BMI Category'] = label_encoder.fit_transform(data['BMI Category'])

# 1. Xử lý cột 'Blood Pressure'
# Chia 'Blood Pressure' thành 2 cột: 'Systolic' và 'Diastolic'
data[['Systolic', 'Diastolic']] = data['Blood Pressure'].str.split('/', expand=True)

# Chuyển đổi sang kiểu số
data['Systolic'] = pd.to_numeric(data['Systolic'])
data['Diastolic'] = pd.to_numeric(data['Diastolic'])

# Khởi tạo PowerTransformer
power_transformer = PowerTransformer(method='yeo-johnson')

# 1. Xử lý các biến phân phối lệch trái dùng PowerTransformer
left_skewed_vars = ['Gender']
data[left_skewed_vars] = power_transformer.fit_transform(data[left_skewed_vars])

# 2. Xử lý các biến phân phối lệch phải bằng PowerTransformer
right_skewed_vars = ['Quality of Sleep',  'Heart Rate']
data[right_skewed_vars] = power_transformer.fit_transform(data[right_skewed_vars])

# 3. Xử lý các biến phân phối đa đỉnh
# Sử dụng các cột 'Systolic' và 'Diastolic' thay vì 'Blood Pressure'
multimodal_vars = ['Occupation', 'Systolic', 'Diastolic', 'BMI Category']

# Phân cụm bằng KMeans
for var in multimodal_vars:
    # Sử dụng K-Means
    kmeans = KMeans(n_clusters=3, random_state=42)
    data[f'{var}_cluster'] = kmeans.fit_predict(data[[var]])

    # Sử dụng GMM để xác định các cụm
    gmm = GaussianMixture(n_components=3, random_state=42)
    data[f'{var}_gmm'] = gmm.fit_predict(data[[var]])

# Kiểm tra tên các cột trong dữ liệu
print("Columns in the dataset:\n", data.columns)

# Kiểm tra một vài hàng dữ liệu
print("\nSample data:\n", data.head())

# Khởi tạo LabelEncoder
label_encoder = LabelEncoder()

# Chuyển đổi các cột chuỗi thành số (thay tên cột phù hợp nếu cần)
categorical_columns = ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']  # Thay thế 'Sleep Condition' bằng tên cột đúng

for col in categorical_columns:
    if col in data.columns:
        data[col] = label_encoder.fit_transform(data[col])

# Xử lý cột 'Blood Pressure'
# Chia 'Blood Pressure' thành 2 cột: 'Systolic' và 'Diastolic'
data[['Systolic', 'Diastolic']] = data['Blood Pressure'].str.split('/', expand=True).astype(float)
data = data.drop('Blood Pressure', axis=1)

# Chuyển đổi sang kiểu số
data['Systolic'] = pd.to_numeric(data['Systolic'], errors='coerce')
data['Diastolic'] = pd.to_numeric(data['Diastolic'], errors='coerce')

# Loại bỏ cột gốc 'Blood Pressure' vì đã chia thành các cột con

# Khởi tạo StandardScaler
standard_scaler = StandardScaler()

# Tiêu chuẩn hóa dữ liệu (Standardization)
standardized_data = pd.DataFrame(standard_scaler.fit_transform(data), columns=data.columns)

# Kiểm tra kết quả
print("\nStandardized Data:\n", standardized_data.describe())

# Định nghĩa hàm tính Nash-Sutcliffe Efficiency (NSE)
def nse(y_true, y_pred):
    "Tính toán chỉ số Nash-Sutcliffe Efficiency (NSE)."
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (numerator / denominator)

# Hàm tính toán các chỉ số đánh giá trên các tập dữ liệu
def evaluate_model(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))  # Tính RMSE trực tiếp từ MSE
    nse_value = nse(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)  # Tính toán MAE
    return r2, rmse, nse_value, mae  # Trả về các chỉ số không có MSE

# Chuyển đổi các biến phân loại thành số
label_encoders = {}
for column in ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Xác minh rằng cột Person ID không có trong dữ liệu
if 'Person ID' in data.columns:
    data = data.drop('Person ID', axis=1)

# Tách dữ liệu thành các đặc trưng (features) và nhãn (target)
if 'Person ID' in data.columns:
    data = data.drop('Person ID', axis=1)

X = data.drop('Sleep Disorder', axis=1)  # Loại bỏ cột Sleep Disorder khỏi các đặc trưng
y = data['Sleep Disorder']  # Cột Sleep Disorder làm nhãn

# Chia dữ liệu thành tập huấn luyện, xác thực và tập kiểm tra
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 2. Tải LabelEncoders và các mô hình đã huấn luyện
with open('label_encoders_linear.pkl', 'rb') as file:
    label_encoders_linear = pickle.load(file)

with open('label_encoders_lasso.pkl', 'rb') as file:
    label_encoders_lasso = pickle.load(file)

with open('model_linear.pkl', 'rb') as file:
    model_linear = pickle.load(file)

with open('model_lasso.pkl', 'rb') as file:
    model_lasso = pickle.load(file)

model_neural_network = keras.models.load_model('model_neural_network.h5')

# 3. Dự đoán giá trị bằng các mô hình
y_train_linear_pred = model_linear.predict(X_train)
y_train_lasso_pred = model_lasso.predict(X_train)
y_train_nn_pred = model_neural_network.predict(X_train)

y_val_linear_pred = model_linear.predict(X_val)
y_val_lasso_pred = model_lasso.predict(X_val)
y_val_nn_pred = model_neural_network.predict(X_val)

y_test_linear_pred = model_linear.predict(X_test)
y_test_lasso_pred = model_lasso.predict(X_test)
y_test_nn_pred = model_neural_network.predict(X_test)

# 4. Tạo tập dữ liệu mới từ các dự đoán
X_train_stacking = np.column_stack((y_train_linear_pred, y_train_lasso_pred, y_train_nn_pred))
X_val_stacking = np.column_stack((y_val_linear_pred, y_val_lasso_pred, y_val_nn_pred))
X_test_stacking = np.column_stack((y_test_linear_pred, y_test_lasso_pred, y_test_nn_pred))

# 5. Huấn luyện mô hình stacking
stacked_model = LinearRegression()
stacked_model.fit(X_train_stacking, y_train)

# 6. Dự đoán và đánh giá mô hình stacking
y_val_stacking_pred = stacked_model.predict(X_val_stacking)
y_test_stacking_pred = stacked_model.predict(X_test_stacking)

r2_val_stacking = r2_score(y_val, y_val_stacking_pred)
rmse_val_stacking = np.sqrt(mean_squared_error(y_val, y_val_stacking_pred))

r2_test_stacking = r2_score(y_test, y_test_stacking_pred)
rmse_test_stacking = np.sqrt(mean_squared_error(y_test, y_test_stacking_pred))

# 7. Hiển thị kết quả
print(f'Validation R²: {r2_val_stacking}')
print(f'Validation RMSE: {rmse_val_stacking}')
print(f'Test R²: {r2_test_stacking}')
print(f'Test RMSE: {rmse_test_stacking}')

# 8. Lưu mô hình stacking
# with open('model_stacking.pkl', 'wb') as file:
#     pickle.dump(stacked_model, file)
