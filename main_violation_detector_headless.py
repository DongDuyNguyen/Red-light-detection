import cv2
import time
import os
import datetime
import argparse # Import argparse
import pyodbc
import numpy as np 

# Assuming TrafficLightDetector and VehicleDetector can run without GUI
from traffic_light_detector import TrafficLightDetector
from vehicle_detector import VehicleDetector
# Import license plate reader function
from license_plate_reader import read_license_plate
# Import necessary functions from utils
from utils import (
    # draw_detections, # Not needed for headless display
    # select_stop_line, select_roi as select_traffic_light_roi, # Remove interactive selections
    encode_violation_frame,
    detect_and_crop_plate # Keep plate detection if desired
)

# --- Cấu hình --- (MODIFIED FOR HEADLESS)
# These might be overridden by args or could be removed if args handle all
# VIDEO_PATH = r"path_set_by_argparse"
MODEL_DIR = r"C:\Users\MY PC\Máy tính\red_light_violation_system\trained_models"
TRAFFIC_LIGHT_MODEL_FILENAME = "traffic_light_cnn_noaug_best.keras"
TRAFFIC_LIGHT_MODEL_PATH = os.path.join(MODEL_DIR, TRAFFIC_LIGHT_MODEL_FILENAME)

# Output paths likely not needed for headless DB logging version
# OUTPUT_EXCEL_PATH = r"..."
# VIOLATION_FRAMES_DIR = r"..."

# Background Subtraction Parameters (Keep as is or make configurable)
BGS_HISTORY = 500
BGS_VAR_THRESHOLD = 16
BGS_DETECT_SHADOWS = False

# Contour Filtering Parameter
MIN_CONTOUR_AREA = 1000 # Keep the tuned value

# CNN Parameters
CNN_INPUT_SIZE = (64, 64)
# TRAFFIC_LIGHT_CLASSES = {0: "RED", 1: "YELLOW", 2: "GREEN"} # Defined in TrafficLightDetector
PREDICTION_CONFIDENCE_THRESHOLD = 0.9 # Keep the tuned value

# --- HARDCODED ROI and STOP LINE ---
# !!! IMPORTANT: Replace these with actual values for your demo video !!!
# These MUST be set as interactive selection is removed.
TRAFFIC_LIGHT_ROI = (891, 144, 170, 43) # Example ROI (x, y, w, h) - ADJUST THIS!
STOP_LINE_Y = 495                   # Example Stop Line Y - ADJUST THIS!
#------------------------------------

MAX_VIOLATIONS_PER_CYCLE = 10

# --- Database Configuration (Keep as is) ---
SQL_SERVER = 'LAPTOP-K5PS924G\\DUY' # Ví dụ: 'DESKTOP-12345\SQLEXPRESS' hoặc địa chỉ IP
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

# Hàm ghi log vi phạm vào SQL Server (MODIFIED FOR LICENSE PLATE OCR)
def log_violation_to_sql(timestamp, image_data, plate_image_data, license_plate_text="Unknown", license_plate_confidence=0.0, video_id=""):
    """Ghi thông tin vi phạm (thời gian, ảnh frame, ảnh biển số, video_id) vào bảng SQL Server."""
    conn = None
    cursor = None
    if image_data is None:
        print(f"[LOG] Lỗi: Dữ liệu ảnh frame vi phạm (video_id: {video_id}) để ghi vào SQL là None.")
        return

    try:
        # In thông tin kết nối để debug
        print(f"[SQL] Kết nối đến: {CONNECTION_STRING}")
        print(f"[SQL] Bảng: {SQL_TABLE_NAME}")
        
        conn = pyodbc.connect(CONNECTION_STRING, autocommit=False)
        cursor = conn.cursor()
        
        # Sửa lại câu lệnh SQL để có đủ 6 trường, thêm LicensePlateConfidence
        sql_query = f"""
            INSERT INTO {SQL_TABLE_NAME} (ViolationTime, ImagePath, LicensePlate, LicensePlateImage, VideoId, LicensePlateConfidence)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        
        # Sử dụng giá trị biển số từ OCR hoặc "Unknown" nếu không có
        license_plate = license_plate_text if license_plate_text else "Unknown"
        
        print(f"[SQL] Thực thi: {sql_query}")
        print(f"[SQL] Với VideoId: {video_id}, Biển số: {license_plate}, Độ tin cậy: {license_plate_confidence:.2f}")
        
        # Truyền đủ 6 tham số
        cursor.execute(sql_query, timestamp, image_data, license_plate, plate_image_data, video_id, license_plate_confidence)
        conn.commit()
        
        status = "có ảnh biển số" if plate_image_data else "không có ảnh biển số"
        print(f"[LOG] Đã ghi vi phạm (video_id: {video_id}, biển số: {license_plate}, độ tin cậy: {license_plate_confidence:.2f}, {status}) lúc {timestamp} vào SQL Server.")
    except pyodbc.Error as ex:
        print(f"[LOG] Lỗi PYODBC khi ghi vào SQL Server (video_id: {video_id}): {ex}")
        if conn:
            try: conn.rollback()
            except Exception as rb_ex: print(f"[LOG] Lỗi khi rollback: {rb_ex}")
    except Exception as e:
         print(f"[LOG] Lỗi KHÔNG XÁC ĐỊNH khi ghi vào SQL Server (video_id: {video_id}): {e}")
    finally:
        if cursor: cursor.close()
        if conn: conn.close()

# --- Main Application (HEADLESS VERSION) ---
if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='Headless Red Light Violation Detection.')
    parser.add_argument('--video_path', type=str, required=True,
                        help='Path to the input video file.')
    parser.add_argument('--video_id', type=str, required=True,
                        help='Unique identifier for this video processing task.')
    # Optionally add args for ROI/StopLine if you want to configure via command line instead of hardcoding
    # parser.add_argument('--roi_x', type=int, default=1109) ... etc.
    args = parser.parse_args()

    VIDEO_PATH = args.video_path
    CURRENT_VIDEO_ID = args.video_id # Store the video ID for logging

    print(f"--- Starting Headless Detection for Video: {VIDEO_PATH} (ID: {CURRENT_VIDEO_ID}) ---")

    # --- Initialization (Keep relevant parts) ---
    print("[INIT] Initializing violation detection system...")

    # 1. Traffic Light Detector
    if not os.path.exists(TRAFFIC_LIGHT_MODEL_PATH):
        print(f"[ERROR] CNN model not found at {TRAFFIC_LIGHT_MODEL_PATH}")
        exit()
    print(f"[INIT] Loading CNN model from: {TRAFFIC_LIGHT_MODEL_PATH}")
    traffic_light_detector = TrafficLightDetector(model_path=TRAFFIC_LIGHT_MODEL_PATH)
    if traffic_light_detector.model is None:
         print("[ERROR] Failed to load model from TrafficLightDetector. Exiting.")
         exit()
    print("[INIT] CNN model loaded.")

    # 2. Vehicle Detector
    print("[INIT] Initializing Vehicle Detector (Background Subtraction)...")
    vehicle_detector = VehicleDetector(
        history=BGS_HISTORY,
        varThreshold=BGS_VAR_THRESHOLD,
        detectShadows=BGS_DETECT_SHADOWS,
        min_contour_area=MIN_CONTOUR_AREA
    )
    print("[INIT] Vehicle Detector initialized.")

    # 3. Open Video Source
    print(f"[VIDEO] Opening video source: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: '{VIDEO_PATH}'")
        exit()

    # 4. Get Video Properties (width/height needed for checking ROI bounds)
    ret, first_frame = cap.read()
    if not ret:
        print("[ERROR] Cannot read the first frame.")
        cap.release()
        exit()
    frame_height, frame_width, _ = first_frame.shape
    print(f"[VIDEO] Video dimensions: {frame_width}x{frame_height}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reset video to start

    # --- REMOVE Interactive ROI/Stop Line Selection ---
    # The hardcoded TRAFFIC_LIGHT_ROI and STOP_LINE_Y above will be used directly.
    # Ensure they are valid.
    if TRAFFIC_LIGHT_ROI is None or len(TRAFFIC_LIGHT_ROI) != 4:
        print(f"[ERROR] Hardcoded TRAFFIC_LIGHT_ROI is invalid or None: {TRAFFIC_LIGHT_ROI}. Exiting.")
        exit()
    if STOP_LINE_Y is None or STOP_LINE_Y <= 0:
        print(f"[ERROR] Hardcoded STOP_LINE_Y is invalid or None: {STOP_LINE_Y}. Exiting.")
        exit()

    print(f"[CONFIG] Using Traffic Light ROI: {TRAFFIC_LIGHT_ROI}")
    print(f"[CONFIG] Using Stop Line Y: {STOP_LINE_Y}")

    # 5. Initialize State Variables
    current_light_state = "UNKNOWN"
    last_light_state = "UNKNOWN"
    vehicles_violated_this_red_cycle = set()
    violations_count_this_cycle = 0

    # 6. Video Processing Loop (HEADLESS)
    frame_count = 0
    start_time = time.time()
    print("[PROC] Starting video processing loop...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[PROC] End of video or error reading frame.")
            break

        # Do not copy frame if not needed for display drawing
        # frame_copy = frame.copy()
        frame_count += 1

        # --- Traffic Light Detection ---
        light_roi_img = None
        # Use hardcoded ROI, ensure bounds check
        try:
            x, y, w, h = TRAFFIC_LIGHT_ROI
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(frame_width, x + w), min(frame_height, y + h)
            if x2 > x1 and y2 > y1:
                 light_roi_img = frame[y1:y2, x1:x2]
        except Exception as e:
             print(f"[WARN] Error cropping traffic light ROI: {e}")
             light_roi_img = None

        if light_roi_img is not None and light_roi_img.size > 0:
            predicted_state, confidence = traffic_light_detector.predict(light_roi_img)
            if confidence >= PREDICTION_CONFIDENCE_THRESHOLD:
                if current_light_state != "RED" and predicted_state == "RED":
                     # print("[PROC] Light changed to RED. Resetting violation cycle.") # Optional log
                     vehicles_violated_this_red_cycle.clear()
                current_light_state = predicted_state
            # else: pass # Keep previous state if confidence too low
        else:
            current_light_state = "UNKNOWN" # Cannot determine state

        # --- Vehicle Detection ---
        detected_vehicles_boxes, processed_mask = vehicle_detector.detect(frame)
        # Do not display mask: cv2.imshow("Processed Mask", processed_mask)

        # --- Violation Logic ---
        if current_light_state == "RED":
            for (x, y, w, h) in detected_vehicles_boxes:
                trigger_point_y = y # Use top edge for trigger
                vehicle_center_x = x + w // 2
                center_y_for_key = y + h // 2
                vehicle_key = (round(vehicle_center_x / 10), round(center_y_for_key / 10))

                # Use hardcoded STOP_LINE_Y
                if (trigger_point_y > STOP_LINE_Y and
                    vehicle_key not in vehicles_violated_this_red_cycle and
                    violations_count_this_cycle < MAX_VIOLATIONS_PER_CYCLE):

                    timestamp = datetime.datetime.now()
                    # Encode the full frame (or a relevant crop)
                    image_byte_data = encode_violation_frame(frame) # Encode the whole frame for context
                    MIN_VALID_IMAGE_SIZE = 1000

                    if image_byte_data and len(image_byte_data) > MIN_VALID_IMAGE_SIZE:
                        # --- Try to Crop Plate Image ---
                        vx1, vy1 = max(0, x), max(0, y)
                        vx2, vy2 = min(frame_width, x + w), min(frame_height, y + h)
                        vehicle_roi_lpr = frame[vy1:vy2, vx1:vx2]

                        plate_image_np = None
                        plate_image_data = None
                        license_plate_text = None
                        license_plate_confidence = 0.0

                        if vehicle_roi_lpr.size > 0:
                             plate_image_np = detect_and_crop_plate(vehicle_roi_lpr) # Don't need copy if not modifying ROI
                        # else: print("[WARN] Vehicle ROI for plate detection is empty.")

                        if plate_image_np is not None and plate_image_np.size > 0:
                            # Perform OCR on the plate image
                            license_plate_text, license_plate_confidence = read_license_plate(plate_image_np)
                            print(f"[OCR] Detected license plate: {license_plate_text}, confidence: {license_plate_confidence:.2f}")
                            
                            success, encoded_plate = cv2.imencode('.jpg', plate_image_np)
                            if success:
                                plate_image_data = encoded_plate.tobytes()
                                # print("[PROC]   >> Cropped and encoded plate image.") # Optional log
                            # else: print("[WARN]   >> Failed to encode cropped plate image.")
                        # else: print("[PROC]   >> No plate detected/cropped.")

                        # --- Log to SQL Server (with license plate info) ---
                        log_violation_to_sql(timestamp, image_byte_data, plate_image_data, 
                                            license_plate_text, license_plate_confidence, CURRENT_VIDEO_ID)
    
                        # Update tracking and counters
                        vehicles_violated_this_red_cycle.add(vehicle_key)
                        violations_count_this_cycle += 1
                    # else: # Keep warnings for invalid main image data
                         # if not image_byte_data: print("[WARN] Failed to encode main violation frame.")
                         # else: print(f"[WARN] Encoded main frame data too small ({len(image_byte_data)} bytes).")

        elif current_light_state != last_light_state and current_light_state != "RED":
             if last_light_state == "RED":
                 # print("[PROC] Light no longer RED. Resetting violation cycle.") # Optional log
                 vehicles_violated_this_red_cycle.clear()
                 violations_count_this_cycle = 0
        last_light_state = current_light_state

        # --- REMOVE Drawing and Display ---
        # frame_processed = draw_detections(...)
        # cv2.rectangle(...) # Remove ROI drawing
        # cv2.putText(...) # Remove FPS display
        # cv2.imshow("Red Light Violation Detection", frame_processed)

        # --- REMOVE Wait Key ---
        # key = cv2.waitKey(1) & 0xFF
        # if key == ord('q') or key == 27: break

        # Optional: Print progress periodically
        if frame_count % 100 == 0:
             print(f"[PROC] Processed frame {frame_count}...")


    # 7. Cleanup
    print("[PROC] Releasing resources...")
    cap.release()
    # cv2.destroyAllWindows() # No windows were created

    end_time = time.time()
    print(f"[INFO] Total frames processed: {frame_count}")
    print(f"[INFO] Elapsed time: {end_time - start_time:.2f} seconds")
    print(f"--- Headless Detection Finished for Video ID: {CURRENT_VIDEO_ID} ---") 