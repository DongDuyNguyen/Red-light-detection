import cv2
import time
import os
import pandas as pd
import datetime
import argparse
import yaml
import pyodbc

from traffic_light_detector import TrafficLightDetector
from vehicle_detector import VehicleDetector
from license_plate_reader import read_license_plate
from utils import (
    draw_detections, select_stop_line, select_roi as select_traffic_light_roi,
    encode_violation_frame,
    detect_and_crop_plate
)

# --- Cấu hình --- (Điều chỉnh các đường dẫn và tham số nếu cần)
VIDEO_PATH = r"C:\Users\MY PC\Máy tính\red_light_violation_system/a.mp4"  # Sẽ được ghi đè bởi tham số dòng lệnh
MODEL_DIR = r"C:\Users\MY PC\Máy tính\red_light_violation_system\trained_models/"
# Sử dụng model mới nhất đã được đánh giá tốt
TRAFFIC_LIGHT_MODEL_FILENAME = "traffic_light_cnn_noaug_best.keras"
TRAFFIC_LIGHT_MODEL_PATH = os.path.join(MODEL_DIR, TRAFFIC_LIGHT_MODEL_FILENAME)

OUTPUT_EXCEL_PATH = r"C:\Users\ADMIN\Desktop\red_light_violation_system\output/violations.xlsx"
VIOLATION_FRAMES_DIR = r"C:\Users\ADMIN\Desktop\red_light_violation_system\output/violation_frames/"

# Tham số cho Background Subtraction
BGS_HISTORY = 500
BGS_VAR_THRESHOLD = 16
BGS_DETECT_SHADOWS = False

# Tham số xử lý Contour
MIN_CONTOUR_AREA = 1000 # Diện tích tối thiểu của contour để được coi là xe

# Tham số cho CNN đèn giao thông
CNN_INPUT_SIZE = (64, 64) # Phải khớp với lúc huấn luyện
TRAFFIC_LIGHT_CLASSES = {0: "RED", 1: "YELLOW", 2: "GREEN"}
PREDICTION_CONFIDENCE_THRESHOLD = 0.9 # Ngưỡng tin cậy để chấp nhận dự đoán đèn

# Cấu hình thủ công ROI và vạch dừng (Nếu không muốn chọn bằng chuột)
# TRAFFIC_LIGHT_ROI = (x, y, w, h) # Ví dụ: (650, 100, 80, 150)
#***TRAFFIC_LIGHT_ROI = None # Đặt là None để chọn bằng chuột
TRAFFIC_LIGHT_ROI = None
# STOP_LINE_Y = y # Ví dụ: 450
STOP_LINE_Y = None # Đặt là -1 để chọn bằng chuột

# Giới hạn số lượng ảnh vi phạm lưu trong mỗi chu kỳ đèn đỏ
MAX_VIOLATIONS_PER_CYCLE = 10

# --- Cấu hình Kết nối SQL Server ---
# !!! THAY THẾ CÁC GIÁ TRỊ NÀY BẰNG THÔNG TIN CỦA BẠN !!!
SQL_SERVER = 'LAPTOP-K5PS924G\DUY ' # Ví dụ: 'DESKTOP-12345\SQLEXPRESS' hoặc địa chỉ IP
SQL_DATABASE = 'TrafficViolations'
SQL_USER = '' # Bỏ trống nếu dùng Windows Authentication
SQL_PASSWORD = '' # Bỏ trống nếu dùng Windows Authentication
SQL_TABLE_NAME = 'ViolationsNew' # Tên bảng bạn đã tạo

# Chuỗi kết nối (Connection String)
# Dùng Windows Authentication (nếu SQL_USER và SQL_PASSWORD để trống)
if not SQL_USER and not SQL_PASSWORD:
    CONNECTION_STRING = (
        f"DRIVER={{ODBC Driver 17 for SQL Server}};" # Hoặc driver phù hợp khác bạn đã cài
        f"SERVER={SQL_SERVER};"
        f"DATABASE={SQL_DATABASE};"
        f"Trusted_Connection=yes;"
    )
# Dùng SQL Server Authentication
else:
     CONNECTION_STRING = (
        f"DRIVER={{ODBC Driver 17 for SQL Server}};" # Hoặc driver phù hợp khác bạn đã cài
        f"SERVER={SQL_SERVER};"
        f"DATABASE={SQL_DATABASE};"
        f"UID={SQL_USER};"
        f"PWD={SQL_PASSWORD};"
    )

# Hàm ghi log vi phạm vào SQL Server (MODIFIED)
def log_violation_to_sql(timestamp, image_data, plate_image_data, plate_text="Unknown", plate_confidence=0.0):
    """Ghi thông tin vi phạm (thời gian, ảnh frame, ảnh biển số, text biển số) vào bảng SQL Server."""
    conn = None
    cursor = None
    # Cần cả ảnh frame chính
    if image_data is None: 
        print("Lỗi: Dữ liệu ảnh frame vi phạm để ghi vào SQL là None.")
        return
    
    # plate_image_data có thể là None nếu không cắt được biển số
    
    try:
        conn = pyodbc.connect(CONNECTION_STRING, autocommit=False)
        cursor = conn.cursor()
        # Cập nhật câu lệnh INSERT để thêm cột LicensePlate và LicensePlateConfidence
        # !!! Đảm bảo bạn đã thêm các cột này vào bảng ViolationsNew !!!
        sql_query = f"INSERT INTO {SQL_TABLE_NAME} (ViolationTime, ImagePath, LicensePlateImage, LicensePlate, LicensePlateConfidence) VALUES (?, ?, ?, ?, ?)" 
        # Truyền dữ liệu biển số text và độ tin cậy vào tham số 4 và 5
        cursor.execute(sql_query, timestamp, image_data, plate_image_data, plate_text, plate_confidence) 
        conn.commit()
        # Cập nhật thông báo log
        status = "có ảnh biển số" if plate_image_data else "không có ảnh biển số"
        print(f"Đã ghi vi phạm (ảnh frame, {status}, biển số: {plate_text}, tin cậy: {plate_confidence:.2f}) lúc {timestamp} vào SQL Server.") 
    except pyodbc.Error as ex:
        print(f"Lỗi PYODBC khi ghi vào SQL Server: {ex}")
        if conn:
            try: conn.rollback()
            except Exception as rb_ex: print(f"Lỗi khi rollback: {rb_ex}")
    except Exception as e:
         print(f"Lỗi KHÔNG XÁC ĐỊNH khi ghi vào SQL Server: {e}")
    finally:
        if cursor: cursor.close()
        if conn: conn.close()

# --- Main Application ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hệ thống phát hiện xe vượt đèn đỏ.')
    parser.add_argument('--video_path', type=str, default=VIDEO_PATH,
                        help='Đường dẫn đến file video đầu vào.')
    args = parser.parse_args()

    # Sử dụng đường dẫn video từ tham số dòng lệnh
    VIDEO_PATH = args.video_path

    print("--- Hệ thống phát hiện vượt đèn đỏ ---")

    # --- Khởi tạo ---
    print("Khởi tạo hệ thống phát hiện vi phạm...")

    # 1. Bộ phát hiện đèn giao thông
    if not os.path.exists(TRAFFIC_LIGHT_MODEL_PATH):
        print(f"Lỗi: Không tìm thấy model CNN tại {TRAFFIC_LIGHT_MODEL_PATH}")
        exit()
    print(f"Tải model CNN đèn giao thông từ: {TRAFFIC_LIGHT_MODEL_PATH}")
    traffic_light_detector = TrafficLightDetector(model_path=TRAFFIC_LIGHT_MODEL_PATH)
    print(f"Đã tải model đèn giao thông từ: {TRAFFIC_LIGHT_MODEL_PATH}")

    # Kiểm tra xem model đã được tải thành công chưa
    if traffic_light_detector.model is None:
         print("Lỗi: Không thể tải model từ TrafficLightDetector. Thoát chương trình.")
         exit()

    # 2. Bộ phát hiện phương tiện
    print("Khởi tạo bộ phát hiện phương tiện (Background Subtraction)...")
    vehicle_detector = VehicleDetector(
        history=BGS_HISTORY,
        varThreshold=BGS_VAR_THRESHOLD,
        detectShadows=BGS_DETECT_SHADOWS,
        min_contour_area=MIN_CONTOUR_AREA # Truyền giá trị MIN_CONTOUR_AREA vào đây
    )

    # 3. Mở nguồn video
    print(f"Mở nguồn video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Lỗi: Không thể mở video hoặc webcam tại '{VIDEO_PATH}'")
        exit()

    # 4. Lấy frame đầu tiên để thiết lập ROI và vạch dừng (nếu cần)
    ret, first_frame = cap.read()
    if not ret:
        print("Lỗi: Không thể đọc frame đầu tiên từ video.")
        cap.release()
        exit()

    frame_height, frame_width, _ = first_frame.shape
    print(f"Kích thước video: {frame_width}x{frame_height}")

    # 4.1 Chọn ROI đèn giao thông (nếu chưa được cấu hình sẵn)
    if TRAFFIC_LIGHT_ROI is None:
        print("\n--- Chọn vùng chứa đèn giao thông ---")
        TRAFFIC_LIGHT_ROI = select_traffic_light_roi(first_frame.copy())
        if TRAFFIC_LIGHT_ROI is None:
            print("Chưa chọn ROI đèn giao thông. Sử dụng một vùng mặc định (có thể không chính xác).")
            # Cung cấp một ROI mặc định nhỏ ở góc trên bên phải làm ví dụ
            TRAFFIC_LIGHT_ROI = (int(frame_width * 0.8), int(frame_height * 0.05), int(frame_width * 0.1), int(frame_height * 0.2))
            print(f"ROI mặc định: {TRAFFIC_LIGHT_ROI}")
        else:
            print(f"Đã chọn ROI đèn: {TRAFFIC_LIGHT_ROI}")

    # 4.2 Chọn vạch dừng (nếu chưa được cấu hình sẵn)
    if STOP_LINE_Y is None:
        print("\n--- Chọn vạch dừng ---")
        STOP_LINE_Y = select_stop_line(first_frame.copy())
        if STOP_LINE_Y is None:
            print("Chưa chọn vạch dừng. Hệ thống sẽ không thể phát hiện vi phạm.")
            # Có thể thoát hoặc đặt một giá trị mặc định không hợp lý để báo hiệu lỗi
            STOP_LINE_Y = -1 # Giá trị không hợp lệ
        else:
             print(f"Đã chọn vạch dừng tại y = {STOP_LINE_Y}")

    if STOP_LINE_Y <= 0:
         print("Cảnh báo: Vạch dừng chưa được thiết lập hoặc không hợp lệ.")
         print("Hệ thống sẽ chạy nhưng không phát hiện được vi phạm.")

    # 5. Khởi tạo biến trạng thái
    current_light_state = "UNKNOWN"
    last_light_state = "UNKNOWN" # Khởi tạo last_light_state
    vehicles_violated_this_red_cycle = set()
    violations_count_this_cycle = 0 # Biến đếm số vi phạm trong chu kỳ
    
    # Biến theo dõi biển số được phát hiện gần nhất
    last_detected_plate = ""

    # 6. Vòng lặp xử lý video
    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Kết thúc video hoặc lỗi đọc frame.")
            break

        frame_copy = frame.copy() # Làm việc trên bản sao để không ảnh hưởng frame gốc
        frame_count += 1

        # ---- Xử lý đèn giao thông ----
        light_roi_img = None
        if TRAFFIC_LIGHT_ROI:
            try:
                x, y, w, h = TRAFFIC_LIGHT_ROI
                # Đảm bảo ROI nằm trong kích thước frame
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(frame_width, x + w), min(frame_height, y + h)
                if x2 > x1 and y2 > y1:
                     light_roi_img = frame[y1:y2, x1:x2]
            except Exception as e:
                 print(f"Lỗi khi crop ROI đèn: {e}")
                 light_roi_img = None

        # Chỉ dự đoán nếu có ROI hợp lệ
        if light_roi_img is not None and light_roi_img.size > 0:
            predicted_state, confidence = traffic_light_detector.predict(light_roi_img)
            # Chỉ cập nhật trạng thái nếu độ tin cậy đủ cao
            if confidence >= PREDICTION_CONFIDENCE_THRESHOLD:
                # Nếu trạng thái đèn thay đổi từ không phải đỏ sang đỏ, reset danh sách xe vi phạm
                if current_light_state != "RED" and predicted_state == "RED":
                     print("Đèn chuyển sang ĐỎ. Reset danh sách vi phạm.")
                     vehicles_violated_this_red_cycle.clear()
                current_light_state = predicted_state
            # else: # Giữ nguyên trạng thái cũ nếu không đủ tin cậy
                 # print(f"Dự đoán đèn {predicted_state} không đủ tin cậy ({confidence:.2f} < {PREDICTION_CONFIDENCE_THRESHOLD}). Giữ trạng thái {current_light_state}")
            pass # Giữ trạng thái cũ
        else:
            # Nếu không có ROI hoặc ROI không hợp lệ, không thể xác định đèn
            current_light_state = "UNKNOWN"

        # ---- Xử lý phát hiện phương tiện ----
        detected_vehicles_boxes, processed_mask = vehicle_detector.detect(frame_copy)
        # Optional: Hiển thị mask đã xử lý để debug
        # if processed_mask is not None:
        #    cv2.imshow("Processed Mask", processed_mask)

        # ---- Logic phát hiện vi phạm (MODIFIED) ----
        if current_light_state == "RED":
            for (x, y, w, h) in detected_vehicles_boxes:
                trigger_point_y = y # Cạnh trên cùng
                vehicle_center_x = x + w // 2
                center_y_for_key = y + h // 2
                vehicle_key = (round(vehicle_center_x / 10), round(center_y_for_key / 10))

                if (trigger_point_y > STOP_LINE_Y and
                    vehicle_key not in vehicles_violated_this_red_cycle and
                    violations_count_this_cycle < MAX_VIOLATIONS_PER_CYCLE):

                    # Ghi nhận vi phạm
                    timestamp = datetime.datetime.now()

                    # Mã hóa frame thành dữ liệu byte
                    image_byte_data = encode_violation_frame(frame.copy())
                    MIN_VALID_IMAGE_SIZE = 1000 # Ngưỡng kích thước tối thiểu (bytes)

                    # Thêm kiểm tra kích thước trước khi ghi DB
                    if image_byte_data and len(image_byte_data) > MIN_VALID_IMAGE_SIZE:
                        # Crop the vehicle ROI from the original frame
                        vx1, vy1 = max(0, x), max(0, y)
                        vx2, vy2 = min(frame_width, x + w), min(frame_height, y + h)
                        vehicle_roi_lpr = frame[vy1:vy2, vx1:vx2] 
                        
                        plate_image_np = None # Initialize plate image variable
                        plate_image_data = None # Initialize byte data for DB
                        
                        if vehicle_roi_lpr.size > 0:
                             # Call the new function to get the cropped plate image (color NumPy array)
                             print("\n=== BẮT ĐẦU QUÁ TRÌNH PHÁT HIỆN BIỂN SỐ & OCR ===")
                             print(f"Đã phát hiện xe vượt đèn đỏ tại ({x}, {y}, {w}, {h})")
                             plate_image_np = detect_and_crop_plate(vehicle_roi_lpr.copy()) 
                        else:
                             print("Cảnh báo: ROI xe vi phạm rỗng, không thể cắt biển số.")
                        
                        # Khởi tạo giá trị mặc định cho plate_text và confidence
                        plate_text = "Unknown"
                        plate_confidence = 0.0
                        
                        # Encode the cropped plate image if it was detected
                        if plate_image_np is not None and plate_image_np.size > 0:
                            print(f"--- Tìm thấy biển số: {plate_image_np.shape}")
                            print("--- Bắt đầu OCR trên biển số...")
                            # Thực hiện OCR trên biển số
                            try:
                                plate_text, plate_confidence = read_license_plate(plate_image_np)
                                print(f"--- KẾT QUẢ OCR: '{plate_text}' (tin cậy: {plate_confidence:.2f})")
                                
                                # Cập nhật biển số phát hiện được nếu đủ tin cậy
                                if plate_text and plate_confidence > 0.6:
                                    last_detected_plate = plate_text
                                    
                            except Exception as e:
                                print(f"--- LỖI TRONG QUÁ TRÌNH OCR: {e}")
                                import traceback
                                traceback.print_exc()
                                plate_text = "Error"
                                plate_confidence = 0.0

                            # Tiếp tục xử lý mã hóa biển số
                            success, encoded_plate = cv2.imencode('.jpg', plate_image_np) # Encode as JPG
                            if success:
                                plate_image_data = encoded_plate.tobytes()
                                print("   >> Đã cắt và mã hóa ảnh biển số.")
                            else:
                                print("   >> Cảnh báo: Không thể mã hóa ảnh biển số đã cắt.")
                        else:
                            print("   >> Không phát hiện/cắt được ảnh biển số.")
                        
                        # Log to SQL Server với đầy đủ thông tin biển số
                        log_violation_to_sql(timestamp, image_byte_data, plate_image_data, plate_text, plate_confidence)
                        
                        # Update tracking and counters
                        # Print message moved inside log_violation_to_sql
                        # print(f"VI PHẠM ({violations_count_this_cycle + 1}/{MAX_VIOLATIONS_PER_CYCLE}): ...") 
                        vehicles_violated_this_red_cycle.add(vehicle_key)
                        violations_count_this_cycle += 1
                        print("=== KẾT THÚC XỬ LÝ VI PHẠM ===\n")
                    else:
                        # Keep warning for invalid main image data
                        if not image_byte_data:
                            print(f"Cảnh báo: Không thể mã hóa ảnh vi phạm chính (encode trả về None).")
                        else:
                            print(f"Cảnh báo: Dữ liệu ảnh mã hóa chính quá nhỏ ({len(image_byte_data)} bytes).")

        elif current_light_state != last_light_state and current_light_state != "RED":
             # Nếu đèn không còn đỏ (chuyển sang xanh/vàng), reset lại set và biến đếm
             if last_light_state == "RED":
                 print("Đèn không còn đỏ, reset danh sách và số lượng vi phạm trong chu kỳ.")
                 vehicles_violated_this_red_cycle.clear()
                 violations_count_this_cycle = 0 # Reset biến đếm
        # Cập nhật trạng thái đèn cuối cùng
        last_light_state = current_light_state

        # ---- Vẽ thông tin lên frame ----
        # Vẽ thông tin lên frame với biển số
        frame_processed = draw_detections(frame_copy, detected_vehicles_boxes, current_light_state, STOP_LINE_Y, last_detected_plate)

        # Vẽ ROI đèn giao thông (nếu có)
        if TRAFFIC_LIGHT_ROI:
             x,y,w,h = TRAFFIC_LIGHT_ROI
             cv2.rectangle(frame_processed, (x,y), (x+w, y+h), (255, 255, 0), 1) # Màu vàng nhạt

        # ---- Hiển thị kết quả ----
        # Tính FPS
        end_time = time.time()
        try:
            fps = frame_count / (end_time - start_time)
            cv2.putText(frame_processed, f"FPS: {fps:.2f}", (frame_width - 150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        except ZeroDivisionError:
            pass # Tránh lỗi chia cho 0 ở frame đầu tiên

        cv2.imshow("Red Light Violation Detection", frame_processed)

        # ---- Thoát ----
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27: # Nhấn 'q' hoặc ESC để thoát
            print("Đang thoát...")
            break

    # 7. Giải phóng tài nguyên
    print("Giải phóng tài nguyên...")
    cap.release()
    cv2.destroyAllWindows()

    # Giữ lại code lưu Excel nhưng tạm thời vô hiệu hóa
    # print("Lưu danh sách vi phạm vào Excel...")
    # save_violations_to_excel(OUTPUT_EXCEL_PATH)

    print("--- Chương trình kết thúc ---")