PHÁT HIỆN PHƯƠNG TIỆN VƯỢT ĐÈN ĐỎ
🚦 Dự án sử dụng CNN, MOG2 và OCR để phát hiện phương tiện vượt đèn đỏ.
1. Giới thiệu
Hệ thống này được phát triển nhằm phát hiện các phương tiện vi phạm luật giao thông bằng cách vượt đèn đỏ. Dự án sử dụng các kỹ thuật xử lý ảnh và học máy bao gồm:

- CNN để nhận diện trạng thái của đèn giao thông (Đỏ, Vàng, Xanh).
- MOG2 (Background Subtraction) để phát hiện chuyển động và nhận dạng phương tiện.
- OCR (Optical Character Recognition) để đọc và nhận diện biển số xe vi phạm.
2. Mục tiêu chính
- Tự động phát hiện phương tiện vượt đèn đỏ thông qua video giám sát.
- Trích xuất khung hình vi phạm, định vị biển số và nhận diện nội dung.
- Lưu trữ thông tin vi phạm vào cơ sở dữ liệu hoặc hệ thống quản lý.
3. Công nghệ sử dụng
Công nghệ	Mô tả
OpenCV	Xử lý ảnh, MOG2, phát hiện chuyển động
TensorFlow / Keras	Huấn luyện mô hình CNN nhận diện tín hiệu đèn
EasyOCR / Tesseract	Nhận diện ký tự biển số xe
Python	Ngôn ngữ lập trình chính
SQLite / CSV / JSON	Lưu trữ kết quả (tuỳ chọn)
4. Cấu trúc thư mục
```
red_light_violation_system/
├── traffic_light_detector/        # Mô hình CNN
├── vehicle_detection/             # MOG2 xử lý phát hiện phương tiện
├── ocr/                           # Nhận diện biển số bằng OCR
├── videos/                        # Video đầu vào
├── outputs/                       # Hình ảnh/video vi phạm đã xử lý
├── main.py                        # File chạy chính
└── README.md                      # File hướng dẫn này
```
5. Hướng dẫn chạy project
### 1. Cài đặt môi trường
```bash
pip install -r requirements.txt
```
### 2. Chạy project
```bash
python main.py --input videos/sample.mp4 --output outputs/
```
6. Kết quả đầu ra
- Hình ảnh các phương tiện vượt đèn đỏ được lưu lại.
- Thông tin biển số được in ra hoặc lưu file.
- Thời gian, tọa độ, khung hình vi phạm được log.
7. Hạn chế
- Môi trường ánh sáng thay đổi có thể ảnh hưởng độ chính xác.
- OCR chưa tối ưu với biển số bị che mờ hoặc nghiêng.
- Hệ thống chưa tích hợp real-time (trừ khi được nâng cấp thêm).
8. Định hướng phát triển
- Tích hợp real-time với camera IP.
- Sử dụng YOLOv8 để thay thế MOG2 cho độ chính xác cao hơn.
- Huấn luyện mô hình OCR chuyên biệt cho biển số Việt Nam.
- Triển khai trên hệ thống nhúng như Jetson Nano, Raspberry Pi.
9. Tác giả
**SSFRF**
Liên hệ: [dghuytuy@gmail.com/DongDuyNguyen]
10. Giấy phép
Dự án được phát hành dưới giấy phép MIT. Xem thêm trong LICENSE.
