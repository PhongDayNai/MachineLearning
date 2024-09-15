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

### Sử dụng mô hình để dự đoán
---
#### Tải mô hình
```
model = load_model('model_neural_network.h5')
```
+ `load_model`: Hàm load_model từ keras.models được sử dụng để tải mô hình đã được lưu trước đó.
+ `'model_neural_network.h5'`: Đường dẫn đến file mô hình được lưu với định dạng HDF5 (.h5).
+ `model`: Biến chứa mô hình đã được tải, sẵn sàng để sử dụng cho dự đoán hoặc tiếp tục huấn luyện.

#### Tải các label encoder
```
label_encoders = {
    'Gender': LabelEncoder().fit(['Male', 'Female']),
    'Occupation': LabelEncoder().fit(['Software Engineer', 'Doctor', 'Sales Representative', 'Teacher']),
    'BMI Category': LabelEncoder().fit(['Overweight', 'Normal', 'Obese']),
    'Sleep Disorder': LabelEncoder().fit(['None', 'Sleep Apnea', 'Insomnia'])
}
```
+ `LabelEncoder`: Từ `sklearn.preprocessing`, dùng để mã hóa các giá trị phân loại thành số nguyên.
+ `.fit([...])`: Phương thức `fit` của `LabelEncoder` học các giá trị phân loại và ánh xạ chúng đến các số nguyên. Ví dụ, `'Male'` có thể được ánh xạ thành 0 và `'Female'` thành 1.

Giải thích từng encoder:
+ `'Gender'`:
  `LabelEncoder().fit(['Male', 'Female'])`
+ `'Occupation'`:
  `LabelEncoder().fit(['Software Engineer', 'Doctor', 'Sales Representative', 'Teacher'])`
  + Tạo một bộ mã hóa nhãn cho cột `'Occupation'` với các giá trị `'Software Engineer'`, `'Doctor'`, `'Sales Representative'`, và `'Teacher'`...
+ `'BMI Category'`:
  `LabelEncoder().fit(['Overweight', 'Normal', 'Obese'])`
  + Tạo một bộ mã hóa nhãn cho cột `'BMI Category'` với các giá trị `'Overweight'`, `'Normal'`, và `'Obese'`.
+ `'Sleep Disorder'`:
  `LabelEncoder().fit(['None', 'Sleep Apnea', 'Insomnia'])`
  + Tạo một bộ mã hóa nhãn cho cột `'Sleep Disorder'` với các giá trị `'None'`, `'Sleep Apnea'`, và `'Insomnia'`.

#### Nhập dữ liệu vào
```
def get_input():
    gender = input('Gender (Male/Female): ')
    occupation = input('Occupation (Software Engineer/Doctor/Sales Representative/Teacher): ')
    age = int(input('Age: '))
    sleep_duration = float(input('Sleep Duration: '))
    quality_of_sleep = int(input('Quality of Sleep: '))
    physical_activity_level = int(input('Physical Activity Level: '))
    stress_level = int(input('Stress Level: '))
    bmi_category = input('BMI Category (Normal/Overweight/Obese): ')
    blood_pressure = input('Blood Pressure (e.g., 120/80): ')
    heart_rate = int(input('Heart Rate: '))
    daily_steps = int(input('Daily Steps: '))
    
    # Chuyển đổi 'Blood Pressure' thành hai cột riêng biệt (Pressure Systolic và Pressure Diastolic)
    if '/' in blood_pressure:
        systolic, diastolic = map(float, blood_pressure.split('/'))
    else:
        systolic = diastolic = float(blood_pressure)
    
    return pd.DataFrame({
        'Gender': [gender],
        'Occupation': [occupation],
        'Age': [age],
        'Sleep Duration': [sleep_duration],
        'Quality of Sleep': [quality_of_sleep],
        'Physical Activity Level': [physical_activity_level],
        'Stress Level': [stress_level],
        'BMI Category': [bmi_category],
        'Blood Pressure Systolic': [systolic],
        'Blood Pressure Diastolic': [diastolic],
        'Heart Rate': [heart_rate],
        'Daily Steps': [daily_steps]
    })
```

+ Xử lý huyết áp: Kiểm tra dấu `/` trong biến `blood_pressure`:
`if '/' in blood_pressure:`
  + Nếu có `/`:
  ```
  systolic, diastolic = map(float, blood_pressure.split('/'))
  ```
    + `blood_pressure.split('/')`:
      + Phương thức `split('/')` chia chuỗi `blood_pressure` thành một danh sách các chuỗi con, tách biệt bởi dấu `/`.
      + Ví dụ: Nếu `blood_pressure` là "120/80", thì `blood_pressure.split('/')` sẽ tạo ra danh sách `['120', '80']`.
    + `map(float, ...)`:
      + Hàm `map()` áp dụng hàm `float` cho từng phần tử trong danh sách kết quả của `split('/')`.
      + `float` chuyển đổi các chuỗi số thành số thực (float). Ví dụ: `'120'` và `'80'` sẽ được chuyển thành `120.0` và `80.0`.
    + Kết quả:
      + `systolic` nhận giá trị `120.0`.
      + `diastolic` nhận giá trị `80.0`.
  + Nếu không có `/`:
  ```
  else:
    systolic = diastolic = float(blood_pressure)
  ```
    + Nếu chuỗi `blood_pressure` không chứa dấu `/`, mã sẽ thực hiện phần này.
    + Trong trường hợp này, `blood_pressure` chỉ chứa một giá trị duy nhất, ví dụ: "120".
    + `float(blood_pressure)`:
      + Chuyển đổi giá trị nhập vào (chuỗi) thành số thực (float). Ví dụ: `'120'` sẽ được chuyển thành `120.0`.
    + `systolic = diastolic = float(blood_pressure)`:
      + Gán giá trị số thực vừa chuyển đổi cho cả hai biến `systolic` và `diastolic`.
      + Kết quả là cả `systolic` và `diastolic` đều nhận giá trị `120.0`.

#### Tiền xử lý dữ liệu nhập vào
```
def preprocess_input(df):
    df['Gender'] = label_encoders['Gender'].transform(df['Gender'])
    df['Occupation'] = label_encoders['Occupation'].transform(df['Occupation'])
    df['BMI Category'] = label_encoders['BMI Category'].transform(df['BMI Category'])
    # Đảm bảo cột 'Sleep Disorder' không có trong input (nó chỉ là nhãn)
    df = df[['Gender', 'Occupation', 'Age', 'Sleep Duration', 'Quality of Sleep',
             'Physical Activity Level', 'Stress Level', 'BMI Category', 'Blood Pressure Systolic', 
             'Blood Pressure Diastolic', 'Heart Rate', 'Daily Steps']]
    
    return df
```
+ Mã hoá nhãn (Label Encoding)
```
df['Gender'] = label_encoders['Gender'].transform(df['Gender'])
df['Occupation'] = label_encoders['Occupation'].transform(df['Occupation'])
df['BMI Category'] = label_encoders['BMI Category'].transform(df['BMI Category'])
```
  + Mã hóa Nhãn: Các cột `'Gender'`, `'Occupation'`, và `'BMI Category'` chứa dữ liệu phân loại (categorical data), tức là dữ liệu được phân loại thành các nhóm cụ thể như "Male", "Female", "Teacher", "Doctor", và các phân loại BMI như "Normal", "Overweight", "Obese".
  + Label Encoders:
    + `label_encoders` là một từ điển chứa các đối tượng `LabelEncoder` từ thư viện `sklearn.preprocessing` đã được huấn luyện trước.
    + `transform()` của `LabelEncoder` chuyển đổi các giá trị phân loại thành các số nguyên. Ví dụ, "Male" có thể được mã hóa thành 0 và "Female" thành 1.
  + Đoạn mã này áp dụng mã hóa nhãn cho các cột tương ứng trong DataFrame để các giá trị phân loại có thể được sử dụng trong các mô hình học máy.
+ Loại bỏ cột không cần thiết
```
df = df[['Gender', 'Occupation', 'Age', 'Sleep Duration', 'Quality of Sleep',
         'Physical Activity Level', 'Stress Level', 'BMI Category', 'Blood Pressure Systolic', 
         'Blood Pressure Diastolic', 'Heart Rate', 'Daily Steps']]
```
  + Lọc Cột:
    + Đoạn mã này chỉ giữ lại các cột cần thiết trong `DataFrame`.
    + Cột `'Sleep Disorder'` không có trong danh sách cột này, vì nó chỉ là một nhãn hoặc mục tiêu cần dự đoán, không phải là một thuộc tính đầu vào.
    + Các cột khác như `'Gender'`, `'Occupation'`, `'Age'`, và các chỉ số sức khỏe được giữ lại vì chúng là các thuộc tính đầu vào cần thiết cho phân tích hoặc mô hình hóa.
+ Trả về DataFrame đã tiền xử lý
```
return df
```
+ Trả Về `DataFrame`:
  + Sau khi thực hiện mã hóa nhãn và loại bỏ các cột không cần thiết, `DataFrame` đã được xử lý sẵn sàng để phân tích hoặc đào tạo mô hình.
  + `DataFrame` này được trả về từ hàm để có thể tiếp tục được sử dụng trong các bước tiếp theo.

#### Thực hiện dự đoán kết quả
+ Dự đoán kết quả
```
def predict():
    input_df = get_input()
    input_df = preprocess_input(input_df)
    
    # Tiền xử lý thêm: chuẩn hóa dữ liệu
    scaler = StandardScaler()
    # Lưu ý: phải dùng scaler đã được huấn luyện trước đó
    # Nếu không, hãy huấn luyện scaler với dữ liệu huấn luyện và lưu nó để dùng sau
    # Ở đây để đơn giản, tôi sẽ tạo một scaler mới, nhưng thực tế bạn nên lưu và tải scaler đã huấn luyện
    scaler.fit(input_df)  # Giả định bạn có thể sử dụng dữ liệu huấn luyện để fit scaler
    input_df = scaler.transform(input_df)
    
    # Dự đoán
    prediction = model.predict(input_df)
    class_labels = label_encoders['Sleep Disorder'].classes_
    predicted_class = class_labels[np.argmax(prediction)]
    print(f"Predicted Sleep Disorder: {predicted_class}")
```

+ Lấy dữ liệu đầu vào và tiền xử lý nó
```
input_df = get_input()
input_df = preprocess_input(input_df)
```

+ Chuẩn hoá dữ liệu
```
scaler = StandardScaler()
# Lưu ý: phải dùng scaler đã được huấn luyện trước đó
# Nếu không, hãy huấn luyện scaler với dữ liệu huấn luyện và lưu nó để dùng sau
# Ở đây để đơn giản, tôi sẽ tạo một scaler mới, nhưng thực tế bạn nên lưu và tải scaler đã huấn luyện
scaler.fit(input_df)  # Giả định bạn có thể sử dụng dữ liệu huấn luyện để fit scaler
input_df = scaler.transform(input_df)
```
+ Khởi Tạo `StandardScaler`:
  + `StandardScaler` là một công cụ trong thư viện `scikit-learn` dùng để chuẩn hóa dữ liệu. Nó chuẩn hóa dữ liệu sao cho có trung bình bằng 0 và độ lệch chuẩn bằng 1.
+ Huấn Luyện `scaler`:
  + `scaler.fit(input_df)`:
    + `fit()` thường được sử dụng với dữ liệu huấn luyện để xác định các tham số chuẩn hóa (trung bình và độ lệch chuẩn).
    + Ở đây, mã giả định rằng bạn có thể sử dụng dữ liệu hiện tại để huấn luyện `scaler`. Thực tế, bạn nên huấn luyện `scaler` với dữ liệu huấn luyện trước đó và lưu nó để dùng lại trong dự đoán.
+ Chuyển Đổi Dữ Liệu:
  + `input_df = scaler.transform(input_df)`:
    + `transform()` chuẩn hóa dữ liệu đầu vào dựa trên các tham số đã huấn luyện từ bước `fit()`.
    + Dữ liệu được chuẩn hóa sẽ có dạng phù hợp để mô hình học máy có thể xử lý.
+ Dự đoán
```
prediction = model.predict(input_df)
```
  + Dự Đoán Bằng Mô Hình:
    + `model.predict(input_df)` gọi mô hình học máy đã được huấn luyện (giả sử đã được định nghĩa và huấn luyện trước đó) để dự đoán nhãn cho dữ liệu đầu vào.
    + Kết quả là một mảng chứa xác suất cho từng lớp (class) mà mô hình dự đoán.
+ Giải mã dự đoán
```
class_labels = label_encoders['Sleep Disorder'].classes_
predicted_class = class_labels[np.argmax(prediction)]
print(f"Predicted Sleep Disorder: {predicted_class}")
```
  + Lấy Các Nhãn Lớp:
    + `label_encoders['Sleep Disorder'].classes_`:
      + Đây là danh sách các nhãn lớp cho rối loạn giấc ngủ, mà mô hình học máy dự đoán.
  + Tìm Nhãn Dự Đoán:
    + `np.argmax(prediction)`:
      + `np.argmax()` tìm chỉ số của giá trị lớn nhất trong mảng dự đoán, chỉ ra lớp có xác suất cao nhất.
      + Chỉ số này được sử dụng để tìm nhãn tương ứng từ `class_labels`.
  + In Kết Quả:
    + `print(f"Predicted Sleep Disorder: {predicted_class}")`:
      + In ra nhãn lớp dự đoán (rối loạn giấc ngủ) mà mô hình dự đoán.
+ Chạy hàm `predict()` trả ra kết quả
```
if __name__ == '__main__':
    predict()
```
  + Kiểm tra `__name__`:
    + `if __name__ == '__main__'`: đảm bảo rằng hàm `predict()` chỉ được gọi khi script được chạy trực tiếp, không phải khi nó được nhập vào như một mô-đun trong một chương trình khác.