import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.metrics import classification_report
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

# Xây dựng mô hình Lasso
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', Lasso())
])

# Huấn luyện mô hình
model.fit(X_train, y_train)

# Dự đoán và đánh giá mô hình
y_pred = model.predict(X_val)

# Chuyển đổi dự đoán về nhãn văn bản
y_pred = label_encoder.inverse_transform(y_pred.round().astype(int))

# Chuyển đổi nhãn thực tế về nhãn văn bản
y_val = label_encoder.inverse_transform(y_val)

print(classification_report(y_val, y_pred))

# Vẽ đồ thị kết quả huấn luyện
plt.figure(figsize=(10, 6))
sns.lineplot(data=pd.DataFrame(model.named_steps['classifier'].coef_).T)
plt.title('Lasso Coefficients Over Training')
plt.xlabel('Feature')
plt.ylabel('Coefficient Value')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Lưu mô hình và encoder
joblib.dump(model, 'lasso_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
