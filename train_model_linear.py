import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Đọc dữ liệu từ file CSV
data = pd.read_csv('sleep_data.csv')

# Kiểm tra và xử lý giá trị thiếu
print("Dữ liệu thiếu:")
print(data.isna().sum())

# Loại bỏ các hàng có giá trị thiếu trong cột 'Sleep Disorder'
data = data.dropna(subset=['Sleep Disorder'])

# Tiền xử lý dữ liệu
X = data.drop(['Person ID', 'Sleep Disorder'], axis=1)
y = data['Sleep Disorder']

# Chuyển đổi các nhãn văn bản thành số
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Chia dữ liệu thành training, validation và test set
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Xây dựng pipeline tiền xử lý dữ liệu
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 'Stress Level', 'Daily Steps']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Gender', 'Occupation', 'BMI Category', 'Blood Pressure'])
    ])

# Xây dựng mô hình Hồi Quy Tuyến Tính
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Huấn luyện mô hình
model.fit(X_train, y_train)

# Dự đoán và đánh giá mô hình
y_pred = model.predict(X_val)

# Chuyển đổi dự đoán thành nhãn
y_pred_rounded = y_pred.round().astype(int)

# Đảm bảo rằng tất cả các nhãn dự đoán đều hợp lệ
y_pred_rounded = [min(max(0, x), len(label_encoder.classes_) - 1) for x in y_pred_rounded]

# Chuyển đổi dự đoán về nhãn văn bản
try:
    y_pred_labels = label_encoder.inverse_transform(y_pred_rounded)
except ValueError as e:
    print(f"Lỗi chuyển đổi nhãn: {e}")
    y_pred_labels = ["Unknown"] * len(y_pred_rounded)

# Chuyển đổi nhãn thực tế về nhãn văn bản
y_val_labels = label_encoder.inverse_transform(y_val)

# In ra báo cáo đánh giá
print("Mean Squared Error:", mean_squared_error(y_val, y_pred))
print("R^2 Score:", r2_score(y_val, y_pred))

# Vẽ đồ thị kết quả huấn luyện
plt.figure(figsize=(10, 6))
sns.lineplot(data=pd.DataFrame(model.named_steps['regressor'].coef_).T)
plt.title('Linear Regression Coefficients Over Training')
plt.xlabel('Feature')
plt.ylabel('Coefficient Value')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Lưu mô hình và encoder
joblib.dump(model, 'linear_model.pkl')
joblib.dump(label_encoder, 'label_encoder_linear.pkl')
