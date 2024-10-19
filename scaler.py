import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

# Đọc file CSV
data = pd.read_csv('sleep_data.csv')

# Điền giá trị thiếu của cột 'Sleep Disorder' bằng mode
data['Sleep Disorder'].fillna(data['Sleep Disorder'].mode()[0], inplace=True)

# Chuyển đổi các biến phân loại thành số
label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender'])
data['Occupation'] = label_encoder.fit_transform(data['Occupation'])
data['BMI Category'] = label_encoder.fit_transform(data['BMI Category'])

# Xử lý cột 'Blood Pressure'
data[['Systolic', 'Diastolic']] = data['Blood Pressure'].str.split('/', expand=True)
data['Systolic'] = pd.to_numeric(data['Systolic'], errors='coerce')
data['Diastolic'] = pd.to_numeric(data['Diastolic'], errors='coerce')

# Loại bỏ cột 'Blood Pressure' gốc
data = data.drop('Blood Pressure', axis=1)

# Khởi tạo StandardScaler
standard_scaler = StandardScaler()

# Chỉ giữ lại các cột số để tiêu chuẩn hóa
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
standardized_data = pd.DataFrame(standard_scaler.fit_transform(data[numeric_cols]), columns=numeric_cols)

# Lưu scaler vào file
with open('scaler_reg.pkl', 'wb') as scaler_file:
    pickle.dump(standard_scaler, scaler_file)

# Kiểm tra kết quả
print("\nStandardized Data:\n", standardized_data.describe())
