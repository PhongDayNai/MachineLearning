import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

# Đọc mô hình và các bộ tiền xử lý đã lưu
with open('model_reg.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler_reg.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('label_encoders_reg.pkl', 'rb') as le_file:
    label_encoders = pickle.load(le_file)

# Đọc cấu trúc cột đã lưu
with open('column_names_reg.pkl', 'rb') as file:
    column_names = pickle.load(file)

# Nhập dữ liệu từ bàn phím
def get_input():
    data = {
        'Gender': input("Enter Gender (Male/Female): "),
        'Age': float(input("Enter Age: ")),
        'Occupation': input("Enter Occupation: "),
        'Sleep Duration': float(input("Enter Sleep Duration: ")),
        'Quality of Sleep': int(input("Enter Quality of Sleep (1-10): ")),
        'Physical Activity Level': int(input("Enter Physical Activity Level (1-100): ")),
        'Stress Level': int(input("Enter Stress Level (1-10): ")),
        'BMI Category': input("Enter BMI Category (Normal/Overweight/Obese): "),
        'Blood Pressure': input("Enter Blood Pressure (e.g., 120/80): "),
        'Heart Rate': int(input("Enter Heart Rate: ")),
        'Daily Steps': int(input("Enter Daily Steps: "))
    }
    return pd.DataFrame([data])

input_data = get_input()

# Tiền xử lý dữ liệu nhập vào
# Chuyển đổi các biến phân loại thành số
for column in ['Gender', 'Occupation', 'BMI Category']:
    if column in label_encoders:
        input_data[column] = label_encoders[column].transform(input_data[column])
    else:
        raise ValueError(f"Column {column} not found in label encoders")

# Chuyển đổi cột Blood Pressure thành hai cột riêng biệt
if 'Blood Pressure' in input_data.columns:
    input_data[['Systolic', 'Diastolic']] = input_data['Blood Pressure'].str.split('/', expand=True).astype(float)
    input_data = input_data.drop('Blood Pressure', axis=1)
else:
    raise ValueError("Blood Pressure column missing in input data")

# Đảm bảo rằng tất cả các cột cần thiết đều có mặt
for col in column_names:
    if col not in input_data.columns:
        input_data[col] = 0  # Hoặc giá trị mặc định khác

# Sắp xếp cột theo đúng thứ tự
input_data = input_data[column_names]

# In ra các cột để kiểm tra
print("Columns in the input data:")
print(input_data.columns)

# Chuẩn hóa dữ liệu
input_data_scaled = scaler.transform(input_data)

# Dự đoán
prediction = model.predict(input_data_scaled)
rounded_prediction = round(prediction[0])
if rounded_prediction == 0:
    print(f"Predicted Sleep Disorder: None")
else:
    if rounded_prediction == 1:
        print(f"Predicted Sleep Disorder: Sleep Apnea")
    else:
        print(f"Predicted Sleep Disorder: Insomia")
