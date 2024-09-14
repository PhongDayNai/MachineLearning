import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Tải mô hình và encoder
model = joblib.load('linear_model.pkl')
label_encoder = joblib.load('label_encoder_linear.pkl')

# Nhập dữ liệu từ bàn phím và dự đoán
def predict_sleep_disorder():
    data = {}
    data['Gender'] = input('Gender: ')
    data['Age'] = float(input('Age: '))
    data['Occupation'] = input('Occupation: ')
    data['Sleep Duration'] = float(input('Sleep Duration: '))
    data['Quality of Sleep'] = float(input('Quality of Sleep: '))
    data['Physical Activity Level'] = float(input('Physical Activity Level: '))
    data['Stress Level'] = float(input('Stress Level: '))
    data['BMI Category'] = input('BMI Category: ')
    data['Blood Pressure'] = input('Blood Pressure: ')
    data['Heart Rate'] = float(input('Heart Rate: '))
    data['Daily Steps'] = float(input('Daily Steps: '))

    df = pd.DataFrame([data])
    
    # Dự đoán
    prediction = model.predict(df)
    
    # Chuyển đổi dự đoán thành nhãn
    prediction_rounded = round(prediction[0])
    prediction_rounded = min(max(0, prediction_rounded), len(label_encoder.classes_) - 1)
    prediction_label = label_encoder.inverse_transform([prediction_rounded])
    
    print(f"Predicted Sleep Disorder: {prediction_label[0]}")

predict_sleep_disorder()
