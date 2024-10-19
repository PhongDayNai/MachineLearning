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

# Định nghĩa hàm tính Nash-Sutcliffe Efficiency (NSE)
def nse(y_true, y_pred):
    "Tính toán chỉ số Nash-Sutcliffe Efficiency (NSE)."
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (numerator / denominator)

# Hàm tính toán các chỉ số đánh giá trên các tập dữ liệu
def evaluate_model(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))  # Tính RMSE trực tiếp từ MSE
    nse_value = nse(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)  # Tính toán MAE
    return r2, rmse, nse_value, mae  # Trả về các chỉ số không có MSE

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


# Khởi tạo mô hình hồi quy tuyến tính
lr_model = LinearRegression()

# Huấn luyện mô hình trên dữ liệu train
lr_model.fit(X_train, y_train)

# Dự đoán trên tập train, validation, và test
y_train_pred = lr_model.predict(X_train)
y_validation_pred = lr_model.predict(X_val)
y_test_pred = lr_model.predict(X_test)

# Đánh giá trên tập train
r2_train, rmse_train, nse_train, mae_train = evaluate_model(y_train, y_train_pred)

# Đánh giá trên tập validation
r2_val, rmse_val, nse_val, mae_val = evaluate_model(y_val, y_validation_pred)

# Đánh giá trên tập test
r2_test, rmse_test, nse_test, mae_test = evaluate_model(y_test, y_test_pred)

# Tạo DataFrame để hiển thị kết quả
results = {
    'Metric': ['LinearRegression-R²', 'LinearRegression-RMSE', 'LinearRegression-NSE', 'LinearRegression-MAE'],
    'Train': [r2_train, rmse_train, nse_train, mae_train],
    'Validation': [r2_val, rmse_val, nse_val, mae_val],
    'Test': [r2_test, rmse_test, nse_test, mae_test]
}

results_df = pd.DataFrame(results)

# Hiển thị DataFrame
print(results_df)

# Hàm để vẽ đường cong parabol từ dữ liệu thực và dự đoán
def plot_parabola(x, y_true, y_pred, title):
    # Fit đường parabol cho cả giá trị thực và dự đoán
    poly_true = np.poly1d(np.polyfit(x, y_true, 2))  # Bậc 2 (parabol)
    poly_pred = np.poly1d(np.polyfit(x, y_pred, 2))

    # Tạo dải giá trị x để vẽ đường parabol mượt mà hơn
    x_smooth = np.linspace(x.min(), x.max(), 500)

    # Vẽ đồ thị
    plt.figure(figsize=(12, 6))
    plt.plot(x_smooth, poly_true(x_smooth), label='True Values (Parabola)', color='blue')
    plt.plot(x_smooth, poly_pred(x_smooth), label='Predicted Values (Parabola)', color='orange')
    plt.legend()
    plt.xlabel('Sample Index')
    plt.ylabel('Price')
    plt.title(title)
    plt.show()

# Tạo chỉ số cho từng tập dữ liệu
x_train_range = np.arange(len(y_train))
x_val_range = np.arange(len(y_val))
x_test_range = np.arange(len(y_test))

# Vẽ đồ thị parabol cho tập train
plot_parabola(x_train_range, y_train, y_train_pred, 'Train Set: True vs Predicted Values (Parabola)')

# Vẽ đồ thị parabol cho tập validation
plot_parabola(x_val_range, y_val, y_validation_pred, 'Validation Set: True vs Predicted Values (Parabola)')

# Vẽ đồ thị parabol cho tập test
plot_parabola(x_test_range, y_test, y_test_pred, 'Test Set: True vs Predicted Values (Parabola)')

# Lưu LabelEncoder
with open('label_encoders_linear.pkl', 'wb') as file:
    pickle.dump(label_encoders, file)

# Lưu mô hình Linear
with open('model_linear.pkl', 'wb') as file:
    pickle.dump(lr_model, file)

with open('scaler_reg.pkl', 'wb') as scaler_file:
    pickle.dump(standard_scaler, scaler_file)