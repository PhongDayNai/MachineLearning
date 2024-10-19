import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, PowerTransformer, LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.utils.np_utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical


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

# Giả sử nhãn của bạn chưa ở dạng one-hot encoding, ta cần chuyển đổi
y_train_categorical = to_categorical(y_train)
y_val_categorical = to_categorical(y_val)
y_test_categorical = to_categorical(y_test)

# Chuẩn hóa dữ liệu đầu vào
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Xây dựng mô hình mạng nơ-ron
model = Sequential()

# Thêm các lớp vào mô hình
model.add(Dense(units=64, activation='relu', input_dim=X_train_scaled.shape[1]))
model.add(Dense(units=32, activation='relu'))

# Lớp đầu ra sử dụng softmax cho phân loại nhiều lớp
model.add(Dense(units=y_train_categorical.shape[1], activation='softmax'))

# Biên dịch mô hình với hàm mất mát cho phân loại nhiều lớp
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình trên tập huấn luyện
history = model.fit(X_train_scaled, y_train_categorical, epochs=30, batch_size=32, validation_data=(X_val_scaled, y_val_categorical))

# Dự đoán trên tập train, validation, và test
y_train_pred_nn = model.predict(X_train_scaled)
y_val_pred_nn = model.predict(X_val_scaled)
y_test_pred_nn = model.predict(X_test_scaled)

# Chuyển đổi kết quả dự đoán từ xác suất sang lớp dự đoán
y_train_pred_classes = y_train_pred_nn.argmax(axis=1)
y_val_pred_classes = y_val_pred_nn.argmax(axis=1)
y_test_pred_classes = y_test_pred_nn.argmax(axis=1)

# Đánh giá mô hình neural network
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Tính toán các chỉ số đánh giá trên tập train, validation và test
train_accuracy = accuracy_score(y_train, y_train_pred_classes)
val_accuracy = accuracy_score(y_val, y_val_pred_classes)
test_accuracy = accuracy_score(y_test, y_test_pred_classes)

train_precision = precision_score(y_train, y_train_pred_classes, average='weighted')
val_precision = precision_score(y_val, y_val_pred_classes, average='weighted')
test_precision = precision_score(y_test, y_test_pred_classes, average='weighted')

train_recall = recall_score(y_train, y_train_pred_classes, average='weighted')
val_recall = recall_score(y_val, y_val_pred_classes, average='weighted')
test_recall = recall_score(y_test, y_test_pred_classes, average='weighted')

train_f1 = f1_score(y_train, y_train_pred_classes, average='weighted')
val_f1 = f1_score(y_val, y_val_pred_classes, average='weighted')
test_f1 = f1_score(y_test, y_test_pred_classes, average='weighted')

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

import matplotlib.pyplot as plt
import numpy as np

# Dữ liệu cho các chỉ số
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
plt.title('Accuracy by Dataset')
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

# Lưu toàn bộ mô hình bao gồm kiến trúc, trọng số và cấu hình optimizer
model.save('model_neural_network.h5')
