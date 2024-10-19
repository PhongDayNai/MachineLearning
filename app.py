from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Đọc mô hình và các bộ tiền xử lý đã lưu
with open('model_linear.pkl', 'rb') as model_file:
    model_linear = pickle.load(model_file)

with open('model_lasso.pkl', 'rb') as model_file:
    model_lasso = pickle.load(model_file)

with open('model_neural_network.h5', 'rb') as model_file:
    model_neural_network = model_file  # Sử dụng Keras hoặc TensorFlow để tải mô hình này

with open('model_stacking.pkl', 'rb') as model_file:
    model_stacking = pickle.load(model_file)

with open('label_encoders_linear.pkl', 'rb') as le_file:
    label_encoders_linear = pickle.load(le_file)

with open('label_encoders_lasso.pkl', 'rb') as le_file:
    label_encoders_lasso = pickle.load(le_file)

with open('scaler_reg.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Cột đầu vào cho mô hình
input_columns = [
    'Gender', 'Age', 'Occupation', 'Sleep Duration', 
    'Quality of Sleep', 'Physical Activity Level', 
    'Stress Level', 'BMI Category', 
     'Heart Rate', 'Daily Steps', 'Systolic', 'Diastolic'
]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Lấy dữ liệu từ form
        data = {
            'Gender': request.form['gender'],
            'Age': float(request.form['age']),
            'Occupation': request.form['occupation'],
            'Sleep Duration': float(request.form['sleep_duration']),
            'Quality of Sleep': int(request.form['quality_of_sleep']),
            'Physical Activity Level': int(request.form['physical_activity_level']),
            'Stress Level': int(request.form['stress_level']),
            'BMI Category': request.form['bmi_category'],
            'Blood Pressure': request.form['blood_pressure'],
            'Heart Rate': int(request.form['heart_rate']),
            'Daily Steps': int(request.form['daily_steps']),
            'model_choice': request.form['model_choice']  # Lấy lựa chọn mô hình từ form
        }

        # Tạo DataFrame
        input_data = pd.DataFrame([data])

        # Chọn LabelEncoder phù hợp với mô hình
        if data['model_choice'] == 'linear':
            encoder = label_encoders_linear
        elif data['model_choice'] == 'lasso':
            encoder = label_encoders_lasso
        else:
            encoder = None

        # Tiền xử lý các biến phân loại
        if encoder is not None:
            for column in ['Gender', 'Occupation', 'BMI Category']:
                input_data[column] = encoder[column].transform(input_data[column])

        # Chuyển đổi Blood Pressure
        if 'Blood Pressure' in input_data.columns:
            input_data[['Systolic', 'Diastolic']] = input_data['Blood Pressure'].str.split('/', expand=True).astype(float)
            input_data = input_data.drop('Blood Pressure', axis=1)

        # Thêm các cột còn thiếu nếu cần thiết (đảm bảo dữ liệu đầu vào có đúng số cột cho mô hình)
        for col in column_names:
            if col not in input_data.columns:
                input_data[col] = 0  # Hoặc giá trị mặc định khác

        input_data = input_data[column_names]

        # Chuẩn hóa dữ liệu nếu cần (nếu bạn có scaler)
        # input_data_scaled = scaler.transform(input_data)

        # Dự đoán dựa trên mô hình đã chọn
        if data['model_choice'] == 'linear':
            prediction_value = model_linear.predict(input_data)

        elif data['model_choice'] == 'lasso':
            prediction_value = model_lasso.predict(input_data)

        elif data['model_choice'] == 'neural_network':
            prediction_value = model_neural_network.predict(input_data)

        elif data['model_choice'] == 'stacking':
            prediction_value = model_stacking.predict(input_data)

        # Xử lý giá trị dự đoán
        rounded_prediction = round(prediction_value[0])
        if rounded_prediction == 0:
            prediction = "None"
        elif rounded_prediction == 1:
            prediction = "Sleep Apnea"
        else:
            prediction = "Insomnia"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
