## Phương pháp Neural Network

### Thư viện cần thiết
---
Những thư viện cần thiết bao gồm: pandas, numpy, matplotlib, sklearn, keras.

Để cài đặt các thư viện này, hãy mở command prompt và chạy lệnh:

`pip install pandas numpy matplotlib scikit-learn keras tensorflow`

### Đọc dữ liệu
---
Sử dung thư viện `pandas, gọi hàm `read_csv()` để đọc file dữ liệu huấn luyện csv

`data = pd.read_csv('sleep_data.csv')`

### Tiền xử lý dữ liệu
---
`data = data.drop('Person ID', axis=1)`

+ Bỏ cột `'Person ID'` không huấn luyện bằng hàm drop đối tượng `DateFrame` trong thư viện pandas

  `DateFrame` *giống như một bang dữ lieu trong CSDL hoặc bang tính Excel, nó bao gồm các hàng và cột*

+ Các tham số của hàm drop bao gồm: `'Person ID'` là cột muốn loại bỏ, `axis=1` là loại bỏ cột (`axis=0` là loại bỏ hàng)

+ Sau khi xử lý trả về một `DateFrame` mới mà không có cột `'Person ID'`

### Chuyển biến phân loại (categorical variables) thành biến số (numeric variables)
---
*Biến phân loại (Categorical variables) là những biến mà giá trị của chúng thuộc về một tập hợp các danh mục hoặc nhóm rời rạc. Các biến phân loại không có thứ tự hoặc khoảng cách cụ thể giữa các giá trị của chúng. Chúng thường được sử dụng để đại diện cho các nhóm hoặc loại khác nhau. ** Ví dụ: Gender (Nam, Nữ), Yes/No (Có, Không); Occupation (Engineer/Doctor/...)...*

*Biến số (Numeric variables) là những biến mà giá trị của chúng có thể đo lường và có thứ tự hoặc khoảng cách cụ thể giữa các giá trị. Biến số thường đại diện cho các số thực hoặc số nguyên và có thể được sử dụng trong các phép toán toán học. ** Ví dụ: Age (25, 26, 27, 30, 35...), Sleep_Duration (1, 2, 3, 4, 5...)...*

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

### Chuyển đổi cột 'Blood Pressure' thành hai cột riêng biệt (Pressure Systolic và Pressure Diastolic)
---
```
data[['Blood Pressure Systolic', 'Blood Pressure Diastolic']] = data['Blood Pressure'].str.split('/', expand=True).astype(float)
data = data.drop('Blood Pressure', axis=1)
```
+ Tách cột `'Blood Pressure'` ra thành 2 cột: `'Blood Pressure Systolic'` và `'Blood Pressure Diastolic'`
+ `data.drop('Blood Pressure', axis=1)`: Loại bỏ cột `'Blood Pressure'`
