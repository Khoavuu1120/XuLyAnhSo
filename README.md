
# NHẬP MÔN XỬ LÝ ẢNH – ĐỒ ÁN NHÓM 02  
## Đề tài: Ứng dụng xử lý hình ảnh và nhận diện khuôn mặt

### Giảng viên hướng dẫn:
TS. Đỗ Hữu Quân

### Nhóm thực hiện:
- Trần Minh Tâm – 2174802010640  
- Trần Văn Quốc Thiện – 2174802010618  
- Trần Văn Tú – 2174802010126  
- Vũ Hoàng Minh Khoa – 2174902010319  
- Chế Ngọc Linh – 2174802010363  

---

## Mục tiêu đề tài:
Xây dựng một ứng dụng sử dụng **OpenCV với Python** để:
- Phát hiện và nhận diện khuôn mặt trong ảnh.
- Căn chỉnh khuôn mặt (xoay, chỉnh hướng).
- Mã hóa đặc trưng khuôn mặt để so sánh và phân loại.

---

## Các kỹ thuật áp dụng:
- **Face Detection** bằng Haarcascade.
- **Facial Landmark Detection**: 68 điểm mốc bằng thuật toán của Kazemi & Sullivan (2014).
- **Affine Transformation**: Căn giữa mắt & miệng.
- **Face Encoding** với mạng neural – sinh vector 128 chiều cho mỗi khuôn mặt.

---

## Công cụ sử dụng:
- Python 3.x  
- Thư viện: `opencv-python`, `dlib`, `numpy`, `tkinter`  
- Visual Studio Code  
- Cơ sở dữ liệu: MySQL (lưu thông tin người dùng)

---

## Cấu trúc thư mục:
```
Doan_nhapmonXLA/
│
├── image/             # Thư mục chứa ảnh gốc
├── faces/             # Ảnh khuôn mặt đã cắt
├── haarcascades/      # Chứa file haarcascade_frontalface_default.xml
├── main.py            # File chạy chính
├── train_model.py     # File huấn luyện nhận diện khuôn mặt
├── README.md          # Giới thiệu dự án
└── Nhóm_02_NMXLA_HK243.docx  # Báo cáo chi tiết
```

---

## Hướng dẫn sử dụng:

1. **Cài đặt thư viện cần thiết:**
   ```bash
   pip install opencv-python dlib numpy
   ```

2. **Chạy chương trình chính:**
   ```bash
   python main.py
   ```

3. **Huấn luyện nhận diện:**
   - Cắt và lưu ảnh khuôn mặt vào thư mục `faces/`.
   - Chạy `train_model.py` để huấn luyện dữ liệu.

---

## Kết quả đạt được:
- Ứng dụng có khả năng cắt và lưu khuôn mặt từ ảnh.
- Định vị và căn chỉnh khuôn mặt theo chuẩn.
- Phân biệt được các khuôn mặt khác nhau bằng phương pháp học sâu.

---

## Tài liệu tham khảo:
- Peter Dauvergne, *Identified, Tracked, and Profiled*  
- Zhang Zhi Yong & Song Bin, *Facial Expression Recognition*  
- Dr. Awanit Kumar et al., *Machine Learning for Emotion & Facial Recognition*

---

## Ghi chú:
Dự án mang tính học thuật, sử dụng mô hình đơn giản để hiểu rõ về các bước xử lý ảnh và nhận dạng khuôn mặt. Không áp dụng cho hệ thống thực tế quy mô lớn.
