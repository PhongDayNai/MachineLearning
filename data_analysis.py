import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, PowerTransformer, LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns

# Đọc file CSV
data = pd.read_csv('sleep_data.csv')

if 'Person ID' in data.columns:
    data = data.drop('Person ID', axis=1)

print(data.head())

# Thiết lập kích thước chung cho các biểu đồ
plt.figure(figsize=(15, 10))

# Vẽ biểu đồ phân phối cho từng biến
variables = ['Gender', 'Age', 'Occupation', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level',  'Stress Level', 'BMI Category', 'Blood Pressure',  'Heart Rate',  'Daily Steps', 'Sleep Disorder']

# Tạo một grid các biểu đồ để vẽ nhiều biến trong một hình ảnh (5 hàng, 3 cột, đủ cho 13 biểu đồ)
fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(20, 20))  # 5 hàng, 3 cột là vừa đủ

# Vẽ từng biểu đồ trong grid
for i, var in enumerate(variables):
    row = i // 3
    col = i % 3
    sns.histplot(data[var], kde=True, ax=axes[row, col])
    axes[row, col].set_title(f'Distribution of {var}')
    axes[row, col].set_xlabel(var)
    axes[row, col].set_ylabel('Frequency')

# Ẩn hai ô trống thừa (vì grid 5x3 có 15 ô, nhưng chỉ cần 13 ô)
for j in range(i + 1, 15):
    fig.delaxes(axes.flat[j])

# Điều chỉnh layout để các biểu đồ không chồng lên nhau
plt.tight_layout()

# Hiển thị toàn bộ các biểu đồ
plt.show()

print(data.info())

# Đếm số lượng giá trị bị thiếu trong mỗi cột
print(data.isnull().sum())

# Chọn các cột số
numeric_data = data.select_dtypes(include=['number'])

# Tính toán ma trận tương quan
corr_matrix = numeric_data.corr()

# Vẽ biểu đồ heatmap cho ma trận tương quan
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

# Giả sử data là DataFrame chứa dữ liệu của bạn
target_column = 'Sleep Disorder'

# In ra danh sách các cột trong DataFrame
print("Các cột trong DataFrame:", data.columns)

# Tạo một figure với lưới các subplot
fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(15, 15))  # Chia thành 5 hàng, 3 cột

# Lập danh sách các biến độc lập
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
independent_vars = numeric_columns[numeric_columns != target_column]

# Kiểm tra nếu cột mục tiêu có trong danh sách biến độc lập
if target_column not in independent_vars:
    print(f"Cột '{target_column}' không có trong danh sách biến số.")

# Vẽ biểu đồ hộp cho từng biến
for ax, column in zip(axes.flatten(), independent_vars):
    sns.boxplot(x=data[target_column], y=data[column], ax=ax, palette='Set2')
    ax.set_title(f"Boxplot giữa {target_column} và {column}")
    ax.set_xlabel(target_column)
    ax.set_ylabel(column)

# Ẩn các subplot không sử dụng (nếu có)
for i in range(len(independent_vars), len(axes.flatten())):
    fig.delaxes(axes.flatten()[i])

# Điều chỉnh bố cục để không bị chồng chéo
plt.tight_layout()
plt.show()

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

# Vẽ biểu đồ để kiểm tra phân phối
plt.figure(figsize=(15, 10))
for i, var in enumerate(left_skewed_vars + right_skewed_vars + multimodal_vars):
    plt.subplot(3, 5, i + 1)
    sns.histplot(data[var], kde=True)
    plt.title(var)
plt.tight_layout()
plt.show()
