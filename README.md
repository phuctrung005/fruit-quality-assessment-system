# 🍎 Hệ thống đánh giá chất lượng hoa quả

**Fruit Quality Assessment System — CNN & Faster R-CNN**

> Đồ án nhóm môn Học máy: xây dựng và tích hợp hai mô hình học sâu để phân loại độ chín và hỗ trợ nhận diện bệnh/hư hỏng trên bề mặt hoa quả từ ảnh.

## 1. Giới thiệu

Hệ thống gồm hai bài toán:

- **Phân loại độ chín:** Mô hình CNN tự xây dựng bằng **TensorFlow/Keras**, phân loại ảnh thành ba mức: **Unripe**, **Ripe** và **Overripe**.
- **Nhận diện bệnh và hư hỏng:** Mô hình **Faster R-CNN ResNet-50 FPN** bằng **PyTorch/Torchvision**, dự đoán nhãn và vùng ảnh (bounding box) cho các lớp đối tượng được huấn luyện.

Hai mô hình được tích hợp trong ứng dụng web **Streamlit**, hỗ trợ tải ảnh và xem kết quả dự đoán. Đây là dự án học tập và thử nghiệm, không phải hệ thống kiểm định chất lượng thực phẩm đã được chứng nhận.

## 2. Công nghệ sử dụng

| Thành phần | Công nghệ |
|---|---|
| Ngôn ngữ | Python |
| Phân loại độ chín | CNN, TensorFlow/Keras |
| Phát hiện bệnh/hư hỏng | Faster R-CNN, PyTorch, Torchvision |
| Xử lý ảnh | OpenCV, Pillow, NumPy |
| Giao diện web | Streamlit |
| Huấn luyện và đánh giá | Jupyter Notebook, scikit-learn, Matplotlib |

## 3. Chức năng chính

- Tải ảnh hoa quả lên giao diện web.
- Phân loại mức độ chín: **chưa chín / chín / quá chín**.
- Hiển thị xác suất dự đoán cho các lớp độ chín.
- Phát hiện và hiển thị nhãn/vị trí vùng bệnh hoặc hư hỏng theo mô hình Faster R-CNN.
- Cho phép điều chỉnh ngưỡng tin cậy cho kết quả phát hiện.

## 4. Cấu trúc dự án

Repository này giữ code trong thư mục `fruit_classification/`:

```text
fruit-ripeness-classification-cnn/
├── .gitignore
├── README.md                     # Tài liệu GitHub này
└── fruit_classification/
    ├── App/
    │   └── unified_app.py        # Ứng dụng Streamlit
    ├── Train_PhanLoai/
    │   ├── Ripe_classification.ipynb
    │   └── ripe_classification.py
    ├── Train_NhanDien/
    │   ├── NhanDienHuHongHoaQua.ipynb
    │   └── nhandienhuhonghoaqua.py
    ├── requirements.txt
    └── BTL_HocMay_Nhom2.pdf      # Báo cáo môn học
```


## 5. Cài đặt và chạy ứng dụng

### Bước 1 — Clone repository

```bash
git clone https://github.com/phuctrung005/fruit-ripeness-classification-cnn.git
cd fruit-ripeness-classification-cnn/fruit_classification
```

### Bước 2 — Tạo môi trường Python

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Linux/macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> Hãy dùng phiên bản Python tương thích với TensorFlow, PyTorch và các phiên bản thư viện trong `requirements.txt`. Các lệnh trên là hướng dẫn cấu hình; chưa được kiểm thử độc lập với mọi hệ điều hành.

### Bước 3 — Chuẩn bị model đã huấn luyện

Ứng dụng `App/unified_app.py` đang yêu cầu **hai file trọng số** tại đúng đường dẫn:

```text
fruit_classification/
└── Models/
    ├── Model_NhanDIen/
    │   └── faster_rcnn_best_model.pth
    └── Model_PhanLoai/
        └── fruit_ripeness_model.h5
```

Các file này **chưa được đính kèm trong repository**. Bạn cần lấy trọng số từ nhóm thực hiện dự án hoặc huấn luyện lại bằng notebook/script tương ứng. Không thể chạy đầy đủ chức năng dự đoán khi thiếu chúng. Không đưa dữ liệu hoặc trọng số lên GitHub nếu chưa có quyền chia sẻ.

### Bước 4 — Chạy ứng dụng

Từ thư mục `fruit_classification/`:

```bash
streamlit run App/unified_app.py
```

Truy cập địa chỉ mà Streamlit hiển thị trong terminal (thường là `http://localhost:8501`).

## 6. Kiến trúc mô hình

### CNN — Phân loại độ chín

Theo tài liệu dự án, CNN sử dụng ảnh đầu vào kích thước **224 × 224**, ba khối **Conv2D + MaxPooling2D**, sau đó là **Flatten, Dense, Dropout** và tầng dự đoán Softmax cho ba mức độ chín. Mô hình được huấn luyện với **Adam** và **Categorical Crossentropy**.

### Faster R-CNN — Nhận diện hư hỏng

Sử dụng kiến trúc `fasterrcnn_resnet50_fpn`, thay đổi đầu dự đoán để phục vụ các lớp đối tượng của bộ dữ liệu riêng. Hệ thống trả về nhãn dự đoán, độ tin cậy và bounding box trên ảnh.

## 7. Huấn luyện, đánh giá và giới hạn

Source code bao gồm các notebook và script huấn luyện trong `Train_PhanLoai/` và `Train_NhanDien/`. Một số script được xuất từ Google Colab nên có thể chứa lệnh hoặc đường dẫn đặc thù của Colab, cần điều chỉnh trước khi chạy bằng Python thông thường.



**Tài liệu tham khảo:** `fruit_classification/BTL_HocMay_Nhom2.pdf`.
