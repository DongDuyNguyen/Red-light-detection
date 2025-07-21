import os
import sys

# Thêm thư mục src vào PYTHONPATH để import các module trong đó
src_path = os.path.join(os.path.dirname(__file__), 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

# Import và chạy script chính
try:
    # Sử dụng __import__ để tránh cảnh báo linting không cần thiết nếu dùng from ... import
    main_module = __import__('main_violation_detector')
    # Nếu bạn muốn gọi một hàm cụ thể trong main_violation_detector.py thay vì
    # dựa vào block if __name__ == "__main__": thì gọi hàm đó ở đây.
    # Ví dụ: main_module.run_detection_function()
    print("Đã thực thi main_violation_detector.py")
except ImportError as e:
    print(f"Lỗi import: {e}")
    print("Hãy đảm bảo bạn đang chạy script này từ thư mục gốc của project (red_light_violation_system)")
except AttributeError:
     # Xử lý nếu main_violation_detector không có hàm cần gọi (nếu bạn thay đổi cấu trúc)
     print("Lỗi: Không tìm thấy hàm thực thi chính trong main_violation_detector.py")
except Exception as e:
     print(f"Đã xảy ra lỗi không mong muốn: {e}")