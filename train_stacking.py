import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, PowerTransformer, LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
import pickle

# Đọc file CSV
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
X = data.drop('Sleep Disorder', axis=1)  # Loại bỏ cột Sleep Disorder khỏi các đặc trưng
y = data['Sleep Disorder']  # Cột Sleep Disorder làm nhãn

# Chia dữ liệu thành tập huấn luyện, tập xác thực và tập kiểm tra
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

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
X = data.drop('Sleep Disorder', axis=1)  # Loại bỏ cột Sleep Disorder khỏi các đặc trưng
y = data['Sleep Disorder']  # Cột Sleep Disorder làm nhãn

# Chia dữ liệu thành tập huấn luyện, tập xác thực và tập kiểm tra
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Mô hình Neural Network
def create_nn_model(input_dim):
    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_dim=input_dim))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))  # Giả định là phân loại nhị phân
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Huấn luyện mô hình Neural Network
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

nn_model = create_nn_model(X_train_scaled.shape[1])
nn_model.fit(X_train_scaled, y_train, epochs=30, batch_size=32, validation_data=(X_val_scaled, y_val))

# Dự đoán từ Neural Network
y_train_nn_pred = (nn_model.predict(X_train_scaled) > 0.5).astype("int32")
y_val_nn_pred = (nn_model.predict(X_val_scaled) > 0.5).astype("int32")
y_test_nn_pred = (nn_model.predict(X_test_scaled) > 0.5).astype("int32")

# Mô hình Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_train_lr_pred = lr_model.predict(X_train)
y_val_lr_pred = lr_model.predict(X_val)
y_test_lr_pred = lr_model.predict(X_test)

# Mô hình Lasso Regression
lasso_model = Lasso(alpha=0.021)
lasso_model.fit(X_train, y_train)
y_train_lasso_pred = lasso_model.predict(X_train)
y_val_lasso_pred = lasso_model.predict(X_val)
y_test_lasso_pred = lasso_model.predict(X_test)

# Tạo DataFrame cho các dự đoán
train_preds = pd.DataFrame({
    'NN': y_train_nn_pred.flatten(),
    'Linear': y_train_lr_pred,
    'Lasso': y_train_lasso_pred
})

val_preds = pd.DataFrame({
    'NN': y_val_nn_pred.flatten(),
    'Linear': y_val_lr_pred,
    'Lasso': y_val_lasso_pred
})

test_preds = pd.DataFrame({
    'NN': y_test_nn_pred.flatten(),
    'Linear': y_test_lr_pred,
    'Lasso': y_test_lasso_pred
})

# Huấn luyện mô hình meta-learner (Linear Regression)
meta_model = LinearRegression()
meta_model.fit(train_preds, y_train)

# Dự đoán cuối cùng
final_train_pred = meta_model.predict(train_preds)
final_val_pred = meta_model.predict(val_preds)
final_test_pred = meta_model.predict(test_preds)

# Đánh giá mô hình
def evaluate_classification(y_true, y_pred):
    accuracy = accuracy_score(y_true, (y_pred > 0.5).astype("int32"))
    precision = precision_score(y_true, (y_pred > 0.5).astype("int32"))
    recall = recall_score(y_true, (y_pred > 0.5).astype("int32"))
    f1 = f1_score(y_true, (y_pred > 0.5).astype("int32"))
    return accuracy, precision, recall, f1

# Đánh giá trên tập train
train_accuracy, train_precision, train_recall, train_f1 = evaluate_classification(y_train, final_train_pred)

# Đánh giá trên tập validation
val_accuracy, val_precision, val_recall, val_f1 = evaluate_classification(y_val, final_val_pred)

# Đánh giá trên tập test
test_accuracy, test_precision, test_recall, test_f1 = evaluate_classification(y_test, final_test_pred)

# Tạo DataFrame để hiển thị kết quả
results = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-score'],
    'Train': [train_accuracy, train_precision, train_recall, train_f1],
    'Validation': [val_accuracy, val_precision, val_recall, val_f1],
    'Test': [test_accuracy, test_precision, test_recall, test_f1]
}

results_df = pd.DataFrame(results)

# Hiển thị DataFrame
print(results_df)

# Dữ liệu cho các chỉ số từ DataFrame
metrics = results_df['Metric']
train_scores = results_df['Train']
val_scores = results_df['Validation']
test_scores = results_df['Test']

# Thiết lập kích thước đồ thị
plt.figure(figsize=(14, 10))

# Đồ thị cho Accuracy
plt.subplot(2, 2, 1)
plt.plot(metrics, train_scores, marker='o', label='Train', color='blue')
plt.plot(metrics, val_scores, marker='o', label='Validation', color='orange')
plt.plot(metrics, test_scores, marker='o', label='Test', color='green')
plt.title('Accuracy, Precision, Recall, and F1-score by Dataset')
plt.xlabel('Metrics')
plt.ylabel('Scores')
plt.ylim(0, 1)  # Giới hạn trục y từ 0 đến 1
plt.grid()
plt.legend()

# Đồ thị cho Precision
plt.subplot(2, 2, 2)
plt.plot(metrics, train_scores, marker='o', label='Train', color='blue')
plt.plot(metrics, val_scores, marker='o', label='Validation', color='orange')
plt.plot(metrics, test_scores, marker='o', label='Test', color='green')
plt.title('Precision by Dataset')
plt.xlabel('Metrics')
plt.ylabel('Scores')
plt.ylim(0, 1)
plt.grid()
plt.legend()

# Đồ thị cho Recall
plt.subplot(2, 2, 3)
plt.plot(metrics, train_scores, marker='o', label='Train', color='blue')
plt.plot(metrics, val_scores, marker='o', label='Validation', color='orange')
plt.plot(metrics, test_scores, marker='o', label='Test', color='green')
plt.title('Recall by Dataset')
plt.xlabel('Metrics')
plt.ylabel('Scores')
plt.ylim(0, 1)
plt.grid()
plt.legend()

# Đồ thị cho F1-score
plt.subplot(2, 2, 4)
plt.plot(metrics, train_scores, marker='o', label='Train', color='blue')
plt.plot(metrics, val_scores, marker='o', label='Validation', color='orange')
plt.plot(metrics, test_scores, marker='o', label='Test', color='green')
plt.title('F1-score by Dataset')
plt.xlabel('Metrics')
plt.ylabel('Scores')
plt.ylim(0, 1)
plt.grid()
plt.legend()

# Hiển thị các đồ thị
plt.tight_layout()
plt.show()

with open('model_stacking.pkl', 'wb') as file:
    pickle.dump(meta_model, file)