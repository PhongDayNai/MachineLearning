import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
import joblib

# Tải mô hình và encoder
model = joblib.load('lasso_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

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
    
    # Chuyển đổi dự đoán về nhãn văn bản
    prediction = label_encoder.inverse_transform([int(round(prediction[0]))])
    
    print(f"Predicted Sleep Disorder: {prediction[0]}")

predict_sleep_disorder()
