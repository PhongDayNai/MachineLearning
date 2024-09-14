import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Đọc dữ liệu từ file CSV
data = pd.read_csv('sleep_data.csv')

# Kiểm tra và xử lý giá trị thiếu
print("Dữ liệu thiếu:")
print(data.isna().sum())

# Loại bỏ các hàng có giá trị thiếu trong cột 'Sleep Disorder'
data = data.dropna(subset=['Sleep Disorder'])

# Tiền xử lý dữ liệu
# Chuyển đổi các biến phân loại thành số
label_encoders = {}
for column in ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Chuyển đổi cột Blood Pressure thành hai cột riêng biệt
data[['Systolic', 'Diastolic']] = data['Blood Pressure'].str.split('/', expand=True).astype(float)
data = data.drop('Blood Pressure', axis=1)

# Xác minh rằng cột Person ID không có trong dữ liệu
if 'Person ID' in data.columns:
    data = data.drop('Person ID', axis=1)

# Tách dữ liệu thành các đặc trưng (features) và nhãn (target)
X = data.drop('Sleep Disorder', axis=1)  # Loại bỏ cột Sleep Disorder khỏi các đặc trưng
y = data['Sleep Disorder']  # Cột Sleep Disorder làm nhãn

# Chia dữ liệu thành tập huấn luyện, tập xác thực và tập kiểm tra
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Tiền xử lý dữ liệu - chuẩn hóa
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Huấn luyện mô hình hồi quy tuyến tính
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Dự đoán trên tập xác thực và tập kiểm tra
y_val_pred = model.predict(X_val_scaled)
y_test_pred = model.predict(X_test_scaled)

# Hiển thị kết quả
print("Validation Set")
print(f"Mean Squared Error: {mean_squared_error(y_val, y_val_pred)}")
print(f"R^2 Score: {r2_score(y_val, y_val_pred)}")

print("\nTest Set")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_test_pred)}")
print(f"R^2 Score: {r2_score(y_test, y_test_pred)}")

# Hiển thị đồ thị kết quả huấn luyện
plt.figure(figsize=(12, 6))
plt.plot(range(len(y_val)), y_val, label='True Values - Validation Set')
plt.plot(range(len(y_val_pred)), y_val_pred, label='Predicted Values - Validation Set')
plt.legend()
plt.xlabel('Sample Index')
plt.ylabel('Sleep Disorder')
plt.title('Validation Set: True vs Predicted Values')
plt.show()

# Lưu mô hình và các bộ tiền xử lý
with open('model_reg.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
    
with open('scaler_reg.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)
    
with open('label_encoders_reg.pkl', 'wb') as le_file:
    pickle.dump(label_encoders, le_file)

# Lưu cấu trúc cột
column_names = X.columns.tolist()
with open('column_names_reg.pkl', 'wb') as file:
    pickle.dump(column_names, file)
