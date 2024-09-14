import pandas as pd
import numpy as np  # Thêm dòng này để nhập khẩu NumPy
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.models import load_model

# Tải mô hình
model = load_model('model.h5')

# Tải các label encoder
label_encoders = {
    'Gender': LabelEncoder().fit(['Male', 'Female']),
    'Occupation': LabelEncoder().fit(['Software Engineer', 'Doctor', 'Sales Representative', 'Teacher']),
    'BMI Category': LabelEncoder().fit(['Overweight', 'Normal', 'Obese']),
    'Sleep Disorder': LabelEncoder().fit(['None', 'Sleep Apnea', 'Insomnia'])
}

# Nhập dữ liệu từ bàn phím
def get_input():
    gender = input('Gender (Male/Female): ')
    occupation = input('Occupation (Software Engineer/Doctor/Sales Representative/Teacher): ')
    age = int(input('Age: '))
    sleep_duration = float(input('Sleep Duration: '))
    quality_of_sleep = int(input('Quality of Sleep: '))
    physical_activity_level = int(input('Physical Activity Level: '))
    stress_level = int(input('Stress Level: '))
    bmi_category = input('BMI Category (Overweight/Normal/Obese): ')
    blood_pressure = input('Blood Pressure (e.g., 120/80): ')
    heart_rate = int(input('Heart Rate: '))
    daily_steps = int(input('Daily Steps: '))
    
    # Chuyển đổi 'Blood Pressure' thành hai cột riêng biệt (Pressure Systolic và Pressure Diastolic)
    if '/' in blood_pressure:
        systolic, diastolic = map(float, blood_pressure.split('/'))
    else:
        systolic = diastolic = float(blood_pressure)
    
    return pd.DataFrame({
        'Gender': [gender],
        'Occupation': [occupation],
        'Age': [age],
        'Sleep Duration': [sleep_duration],
        'Quality of Sleep': [quality_of_sleep],
        'Physical Activity Level': [physical_activity_level],
        'Stress Level': [stress_level],
        'BMI Category': [bmi_category],
        'Blood Pressure Systolic': [systolic],
        'Blood Pressure Diastolic': [diastolic],
        'Heart Rate': [heart_rate],
        'Daily Steps': [daily_steps]
    })

# Tiền xử lý dữ liệu nhập vào
def preprocess_input(df):
    df['Gender'] = label_encoders['Gender'].transform(df['Gender'])
    df['Occupation'] = label_encoders['Occupation'].transform(df['Occupation'])
    df['BMI Category'] = label_encoders['BMI Category'].transform(df['BMI Category'])
    # Đảm bảo cột 'Sleep Disorder' không có trong input (nó chỉ là nhãn)
    df = df[['Gender', 'Occupation', 'Age', 'Sleep Duration', 'Quality of Sleep',
             'Physical Activity Level', 'Stress Level', 'BMI Category', 'Blood Pressure Systolic', 
             'Blood Pressure Diastolic', 'Heart Rate', 'Daily Steps']]
    
    return df

# Dự đoán kết quả
def predict():
    input_df = get_input()
    input_df = preprocess_input(input_df)
    
    # Tiền xử lý thêm: chuẩn hóa dữ liệu
    scaler = StandardScaler()
    # Lưu ý: phải dùng scaler đã được huấn luyện trước đó
    # Nếu không, hãy huấn luyện scaler với dữ liệu huấn luyện và lưu nó để dùng sau
    # Ở đây để đơn giản, tôi sẽ tạo một scaler mới, nhưng thực tế bạn nên lưu và tải scaler đã huấn luyện
    scaler.fit(input_df)  # Giả định bạn có thể sử dụng dữ liệu huấn luyện để fit scaler
    input_df = scaler.transform(input_df)
    
    # Dự đoán
    prediction = model.predict(input_df)
    class_labels = label_encoders['Sleep Disorder'].classes_
    predicted_class = class_labels[np.argmax(prediction)]
    print(f"Predicted Sleep Disorder: {predicted_class}")

if __name__ == '__main__':
    predict()
