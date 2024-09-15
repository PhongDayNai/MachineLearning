## Phương pháp Neural Network

### Thư viện cần thiết
---
+ Những thư viện cần thiết bao gồm: pandas, numpy, matplotlib, sklearn, keras, tensorflow.

+ Để cài đặt các thư viện này, hãy mở command prompt và chạy lệnh:

  ```
  pip install pandas numpy matplotlib scikit-learn keras tensorflow
  ```

### Xử lý dữ liệu
---
#### Đọc dữ liệu
+ Sử dụng thư viện `pandas`, gọi hàm `read_csv()` để đọc file dữ liệu huấn luyện csv

  ```
  data = pd.read_csv('sleep_data.csv')
  ```

#### Tiền xử lý dữ liệu
+ Bỏ cột `'Person ID'` không huấn luyện bằng hàm drop đối tượng `DataFrame` trong thư viện pandas

  `DataFrame` *giống như một bảng dữ liệu trong CSDL hoặc bảng tính Excel, nó bao gồm các hàng và cột*

+ Các tham số của hàm drop bao gồm: `'Person ID'` là cột muốn loại bỏ, `axis=1` là loại bỏ cột (`axis=0` là loại bỏ hàng)

+ Sau khi xử lý trả về một `DataFrame` mới mà không có cột `'Person ID'`

  ```
  data = data.drop('Person ID', axis=1)
  ```

#### Chuyển biến phân loại (categorical variables) thành biến số (numeric variables)
+ *Biến phân loại (Categorical variables) là những biến mà giá trị của chúng thuộc về một tập hợp các danh mục hoặc nhóm rời rạc. Các biến phân loại không có thứ tự hoặc khoảng cách cụ thể giữa các giá trị của chúng. Chúng thường được sử dụng để đại diện cho các nhóm hoặc loại khác nhau. ** Ví dụ: Gender (Nam, Nữ), Yes/No (Có, Không); Occupation (Engineer/Doctor/...)...*

+ *Biến số (Numeric variables) là những biến mà giá trị của chúng có thể đo lường và có thứ tự hoặc khoảng cách cụ thể giữa các giá trị. Biến số thường đại diện cho các số thực hoặc số nguyên và có thể được sử dụng trong các phép toán toán học. ** Ví dụ: Age (25, 26, 27, 30, 35...), Sleep_Duration (1, 2, 3, 4, 5...)...*

```
label_encoders = {}
for column in ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le
```
+ Khai báo từ điển để lưu các nhãn
+ Dùng vòng lặp qua các cột thuộc kiểu biến phân loại
+ Dùng hàm `LabelEncoder()` tạo đối tượng `LabelEncoder` (là công cụ chuyển đổi các giá trị phân loại thành các giá trị số nguyên) từ thư viện `sklearn.processing`
+ Áp dung hàm `fit_transform()` của đối tượng `LabelEncoder` để chuyển đổi giá trị phân loại thành số nguyên, nó thực hiện 2 bước:
  + `fit`: Tìm tất cả các giá trị phân loại duy nhất trong cột và gán cho chúng các giá trị số nguyên
  + `transform`: Thay thế các giá trị phân loại trong cột bằng các số nguyên tương ứng
  + `data[column] = ...`: Cập nhật cột với các giá trị số mới
+ Lưu đối tượng `LabelEncoder` vào từ điển đã khai báo với tên cột tương ứng làm khóa cho phép bạn phục hồi hoặc sử dụng encoder trong tương lai nếu cần. Ví dụ: Giải mã giá trị số trở lại dạng phân loại

#### Chuyển đổi cột `'Blood Pressure'` thành hai cột riêng biệt (Pressure Systolic và Pressure Diastolic)
```
data[['Blood Pressure Systolic', 'Blood Pressure Diastolic']] = data['Blood Pressure'].str.split('/', expand=True).astype(float)
data = data.drop('Blood Pressure', axis=1)
```
+ Tách cột `'Blood Pressure'` ra thành 2 cột: `'Blood Pressure Systolic'` và `'Blood Pressure Diastolic'`
+ `data.drop('Blood Pressure', axis=1)`: Loại bỏ cột `'Blood Pressure'`

#### Tách dữ liệu và nhãn
```
X = data.drop('Sleep Disorder', axis=1)
y = data['Sleep Disorder']
```
+ `X`: Chứa tất cả các cột dữ liệu ngoại trừ cột `'Sleep Disorder'`, tức là các đặc trưng (features) của mô hình.
+ `y`: Chứa cột `'Sleep Disorder'`, tức là nhãn (labels) mà bạn muốn dự đoán.

#### Chia dữ liệu thành train, validation, test
```
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
```
+ `train_test_split` từ `sklearn.model_selection` được dùng để chia dữ liệu thành các tập huấn luyện, kiểm tra và validation.
+ `test_size=0.4` nghĩa là 40% dữ liệu sẽ được chia cho tập kiểm tra tạm thời (`X_temp` và `y_temp`), và 60% dữ liệu sẽ được giữ lại cho tập huấn luyện (`X_train` và `y_train`).
+ Tiếp theo, dữ liệu trong `X_temp` và `y_temp` được chia đôi để tạo ra tập validation và tập kiểm tra cuối cùng (`X_val`, `X_test`, `y_val`, `y_test`), mỗi tập sẽ chiếm 50% của dữ liệu tạm thời.

#### Tiền xử lý dữ liệu thêm: Chuẩn hóa dữ liệu
```
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
```
+ `StandardScaler` từ `sklearn.preprocessing` được dùng để chuẩn hóa dữ liệu.
+ `scaler.fit_transform(X_train)`: Tính toán các tham số chuẩn hóa (trung bình và độ lệch chuẩn) từ tập huấn luyện và áp dụng chuẩn hóa cho nó.
+ `scaler.transform(X_val)` và `scaler.transform(X_test)`: Áp dụng cùng các tham số chuẩn hóa đã tính toán từ tập huấn luyện cho tập validation và tập kiểm tra. Điều này đảm bảo rằng tất cả các dữ liệu đều được chuẩn hóa theo cùng một tiêu chuẩn.

#### Chuyển đổi nhãn thành one-hot encoding
```
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)
```
+ `to_categorical` từ `keras.utils` chuyển đổi các nhãn phân loại thành định dạng one-hot encoding.
+ Ví dụ: Nếu `y_train` có các giá trị `[0, 1, 2]`, thì sau khi chuyển đổi, chúng sẽ được mã hóa thành các vectơ như `[1, 0, 0]`, `[0, 1, 0]`, và `[0, 0, 1]`.

### Mô hình học máy Neural Network
---
#### Xây dựng mô hình
```
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(y_train.shape[1], activation='softmax')
])
```
+ `Sequential`: Đây là loại mô hình cơ bản trong Keras, nơi các lớp được xếp chồng lên nhau theo thứ tự.
+ `Dense`: Lớp fully-connected (hoặc lớp dense) trong mạng nơ-ron.
  + `Dense(64, activation='relu', input_shape=(X_train.shape[1],))`: Lớp đầu tiên có 64 nơ-ron và sử dụng hàm kích hoạt ReLU (Rectified Linear Unit). `input_shape` xác định kích thước của đầu vào là số lượng đặc trưng trong dữ liệu huấn luyện (`X_train.shape[1]`).
  + `Dense(32, activation='relu')`: Lớp thứ hai có 32 nơ-ron và cũng sử dụng hàm kích hoạt ReLU.
  + `Dense(y_train.shape[1], activation='softmax')`: Lớp đầu ra có số nơ-ron bằng số lớp phân loại trong nhãn (`y_train.shape[1]`). Hàm kích hoạt `softmax` được sử dụng để chuyển đổi đầu ra thành xác suất cho các lớp phân loại.

#### Biên dịch mô hình
```
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```
+ `optimizer='adam'`: Sử dụng thuật toán Adam để tối ưu hóa các trọng số của mô hình.
+ `loss='categorical_crossentropy'`: Hàm mất mát (loss function) được sử dụng để đánh giá độ chính xác của dự đoán, phù hợp cho bài toán phân loại nhiều lớp với định dạng one-hot encoding.
+ `metrics=['accuracy']`: Theo dõi độ chính xác của mô hình trong quá trình huấn luyện và đánh giá.

#### Huấn luyện mô hình
```
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=32)
```
+ `X_train, y_train`: Dữ liệu huấn luyện và nhãn.
+ `validation_data=(X_val, y_val)`: Dữ liệu kiểm tra (validation data) để theo dõi hiệu suất của mô hình trên tập validation trong quá trình huấn luyện.
+ `epochs=30`: Số lượng epoch, tức là số lần toàn bộ dữ liệu huấn luyện được đưa qua mô hình.
+ `batch_size=32`: Kích thước của mỗi batch trong quá trình huấn luyện.

`history` là một đối tượng chứa thông tin về quá trình huấn luyện, bao gồm độ chính xác và mất mát của mô hình trong từng epoch.

#### Đánh giá mô hình
```
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")
```
+ `model.evaluate(X_test, y_test)`: Đánh giá mô hình trên dữ liệu kiểm tra (`X_test` và `y_test`), trả về mất mát và độ chính xác.
+ `print(f"Test Accuracy: {test_accuracy:.4f}")`: In độ chính xác của mô hình trên tập kiểm tra với 4 chữ số thập phân.

#### Hiển thị đồ thị kết quả huấn luyện
```
plt.figure(figsize=(12, 5))

# Đồ thị độ chính xác
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')

# Đồ thị mất mát
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')

plt.show()
```
+ `plt.figure(figsize=(12, 5))`: Tạo một hình với kích thước 12x5 inches.
+ `plt.subplot(1, 2, 1)` và `plt.subplot(1, 2, 2)`: Tạo các đồ thị con trong cùng một hình.
  + Đồ thị độ chính xác: Hiển thị độ chính xác của mô hình trên tập huấn luyện và tập validation qua các epoch.
  + Đồ thị mất mát: Hiển thị mất mát của mô hình trên tập huấn luyện và tập validation qua các epoch.
+ `plt.show()`: Hiển thị đồ thị.

#### Lưu mô hình
```
model.save('model_neural_network.h5')
```
+ `model.save('model_neural_network.h5')`: Lưu mô hình đã huấn luyện vào một file với định dạng HDF5 (`.h5`). Bạn có thể tải mô hình này trong tương lai để sử dụng hoặc tiếp tục huấn luyện.