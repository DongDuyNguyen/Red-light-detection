import json
import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import datetime

# --- Configuration ---
# Ánh xạ tên lớp trong COCO sang ID (chỉ lấy các lớp đèn giao thông)
# Đảm bảo tên khớp chính xác với file _annotations.coco.json
TRAFFIC_LIGHT_CLASSES = {
    "den do": 0,
    "den vang": 1,
    "den xanh": 2
    # Thêm các lớp khác nếu cần, nhưng CNN chỉ tập trung vào 3 lớp này
}
# Kích thước ảnh đầu vào cho CNN
CNN_INPUT_SIZE = (64, 64)

# --- COCO Parsing ---
def load_coco_annotations(annotation_path, img_dir):
    """
    Tải dữ liệu từ file COCO JSON, chỉ trích xuất thông tin đèn giao thông.

    Args:
        annotation_path (str): Đường dẫn đến file _annotations.coco.json.
        img_dir (str): Đường dẫn đến thư mục chứa ảnh.

    Returns:
        tuple: (list of image paths, list of corresponding bboxes, list of corresponding labels (int))
               Trả về (None, None, None) nếu có lỗi.
               Labels: 0: RED, 1: YELLOW, 2: GREEN
    """
    try:
        with open(annotation_path, 'r', encoding='utf-8') as f: # Chỉ định encoding='utf-8'
            coco_data = json.load(f)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file annotation: {annotation_path}")
        return None, None, None # Đảm bảo trả về 3 giá trị
    except json.JSONDecodeError:
        print(f"Lỗi: File annotation không đúng định dạng JSON: {annotation_path}")
        return None, None, None # Đảm bảo trả về 3 giá trị
    except Exception as e:
        print(f"Lỗi không xác định khi đọc file annotation: {e}")
        return None, None, None # Đảm bảo trả về 3 giá trị

    # Kiểm tra cấu trúc file cơ bản
    if 'images' not in coco_data or 'annotations' not in coco_data or 'categories' not in coco_data:
        print("Lỗi: File COCO không có cấu trúc 'images', 'annotations', hoặc 'categories'.")
        return None, None, None # Đảm bảo trả về 3 giá trị

    image_id_to_filename = {img['id']: os.path.join(img_dir, img['file_name']) for img in coco_data['images']}
    category_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
    print(f"Các lớp trong COCO: {category_id_to_name}") # Debugging

    # Map tên lớp mong muốn sang ID số nguyên
    target_classes = {"den do": 0, "den vang": 1, "den xanh": 2}
    traffic_light_category_ids = {cat_id for cat_id, cat_name in category_id_to_name.items() if cat_name in target_classes}

    if not traffic_light_category_ids:
        print("Cảnh báo: Không tìm thấy category ID nào khớp với 'den do', 'den vang', 'den xanh' trong file annotations.")
        # return None, None, None # Có thể trả về lỗi ở đây nếu muốn

    image_paths_list = []
    bboxes_list = []
    labels_list = []
    found_traffic_light_annotations = 0

    for ann in coco_data.get('annotations', []):
        image_id = ann.get('image_id')
        category_id = ann.get('category_id')
        bbox = ann.get('bbox')

        if image_id is None or category_id is None or bbox is None:
            continue

        if image_id in image_id_to_filename and category_id in traffic_light_category_ids:
            label_name = category_id_to_name[category_id]
            label_int = target_classes[label_name]
            image_path = image_id_to_filename[image_id]

            if os.path.exists(image_path):
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4 and all(isinstance(n, (int, float)) for n in bbox):
                    image_paths_list.append(image_path)
                    bboxes_list.append(bbox)
                    labels_list.append(label_int)
                    found_traffic_light_annotations += 1
                # else: (Bỏ qua cảnh báo bbox không hợp lệ)
            # else: (Bỏ qua cảnh báo ảnh không tồn tại)

    if not image_paths_list:
        print("Cảnh báo: Không tìm thấy annotation đèn giao thông hợp lệ nào có file ảnh tương ứng.")
        return None, None, None # Đảm bảo trả về 3 giá trị

    print(f"Tìm thấy {found_traffic_light_annotations} annotations đèn giao thông hợp lệ với ảnh tồn tại.")
    return image_paths_list, bboxes_list, labels_list # Đảm bảo trả về 3 giá trị

# --- Image Processing ---
def preprocess_image_for_cnn(image, target_size=CNN_INPUT_SIZE):
    """Chuẩn bị ảnh patch cho đầu vào CNN."""
    if image is None or image.size == 0:
        print("Cảnh báo: Ảnh đầu vào cho preprocess_image_for_cnn bị None hoặc rỗng.")
        return None # Trả về None nếu ảnh không hợp lệ
    try:
        img_resized = cv2.resize(image, target_size)
        img_normalized = img_resized / 255.0 # Chuẩn hóa về [0, 1]
        return img_normalized
    except Exception as e:
        print(f"Lỗi khi resize ảnh trong preprocess_image_for_cnn: {e}")
        return None

def extract_traffic_light_patches(image_paths, bboxes, labels):
    """Trích xuất các patch ảnh đèn giao thông từ danh sách ảnh, bbox và label."""
    patches = []
    patch_labels = []

    if not (len(image_paths) == len(bboxes) == len(labels)):
        print("Lỗi: Số lượng image_paths, bboxes, và labels không khớp!")
        return np.array([]), np.array([]) # Trả về array rỗng

    num_items = len(image_paths)
    print(f"Bắt đầu trích xuất patches từ {num_items} annotations...")

    processed_count = 0
    error_count = 0

    for i in range(num_items):
        image_path = image_paths[i]
        bbox = bboxes[i]
        label = labels[i]

        img = cv2.imread(image_path)
        if img is None:
            error_count += 1
            continue

        height, width, _ = img.shape
        x, y, w, h = map(int, bbox) # Chuyển bbox thành int

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(width, x + w)
        y2 = min(height, y + h)

        if x2 > x1 and y2 > y1: # Bbox hợp lệ
            patch = img[y1:y2, x1:x2]
            if patch.size > 0:
                # Sử dụng target_size từ cấu hình toàn cục hoặc truyền vào
                processed_patch = preprocess_image_for_cnn(patch, target_size=CNN_INPUT_SIZE)
                if processed_patch is not None: # Kiểm tra kết quả từ preprocess
                    patches.append(processed_patch)
                    patch_labels.append(label)
                    processed_count += 1
                else:
                    error_count += 1 # Lỗi xảy ra trong preprocess
            else:
                error_count += 1
        else:
            error_count += 1

        # In tiến độ thưa thớt
        if (i + 1) % 500 == 0 or (i + 1) == num_items:
             print(f"  Đã xử lý {i + 1}/{num_items} annotations... (Thành công: {processed_count}, Lỗi/Bỏ qua: {error_count})", end='\r')

    print(f"\nHoàn tất trích xuất: {processed_count} patches thành công, {error_count} lỗi/bỏ qua.")

    if not patches:
        print("Cảnh báo: Không trích xuất thành công patch nào.")
        return np.array([]), np.array([])

    return np.array(patches), np.array(patch_labels)

# --- Drawing ---
def draw_detections(frame, detected_vehicles_boxes, current_light_state, stop_line_y=-1, license_plate_text=None):
    """Vẽ các phát hiện lên frame."""
    # 1. Vẽ trạng thái đèn
    light_color_map = {"RED": (0, 0, 255), "YELLOW": (0, 255, 255), "GREEN": (0, 255, 0), "UNKNOWN": (128, 128, 128)}
    status_color = light_color_map.get(current_light_state, (128, 128, 128))
    cv2.putText(frame, f"Light: {current_light_state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

    # 2. Vẽ bounding box cho xe
    for (x, y, w, h) in detected_vehicles_boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Vẽ điểm kiểm tra (dưới cùng)
        center_x = x + w // 2
        bottom_y = y + h
        cv2.circle(frame, (center_x, bottom_y), 3, (0, 0, 255), -1)

    # 3. Vẽ vạch dừng (nếu hợp lệ)
    if stop_line_y > 0:
        frame_width = frame.shape[1]
        # --- Tính toán để vẽ đoạn kẻ ngắn hơn --- #
        line_proportion = 0.5 # Tỷ lệ chiều dài đoạn kẻ so với chiều rộng ảnh (vd: 60%)
        line_length = int(frame_width * line_proportion)
        center_x = frame_width // 2
        start_x = max(0, center_x - line_length // 2)
        end_x = min(frame_width, center_x + line_length // 2)
        # --- Vẽ đoạn kẻ ngắn --- #
        cv2.line(frame, (start_x, stop_line_y), (end_x, stop_line_y), (0, 255, 255), 2) # Giữ màu vàng nhạt
        # Vẽ chữ gần giữa đoạn kẻ
        text_x = start_x + 10 # Hoặc tính vị trí khác nếu muốn
        cv2.putText(frame, "Stop Line", (text_x, stop_line_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    
    # 4. Hiển thị thông tin biển số nếu có
    if license_plate_text:
        # Vẽ nền đen để chữ dễ đọc
        cv2.rectangle(frame, (10, frame.shape[0] - 50), (400, frame.shape[0] - 10), (0, 0, 0), -1)
        # Vẽ text biển số
        cv2.putText(frame, f"PLATE: {license_plate_text}", (20, frame.shape[0] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return frame

# --- Violation Saving ---
# violation_data = [] # Không còn dùng danh sách tạm này nữa
# violation_count = 0 # Không cần biến đếm này nữa

def encode_violation_frame(frame):
    """Mã hóa frame ảnh vi phạm thành mảng byte JPEG."""
    if frame is None:
        print("Lỗi: Frame đầu vào để mã hóa là None.")
        return None
    try:
        # Mã hóa frame thành định dạng JPEG trong bộ nhớ
        # Tham số thứ 2 là danh sách các tham số mã hóa (ví dụ chất lượng JPEG, 0-100)
        success, encoded_image = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if success:
            return encoded_image.tobytes() # Trả về dữ liệu dưới dạng bytes
        else:
            print("Lỗi: cv2.imencode không thành công.")
            return None
    except Exception as e:
        print(f"Lỗi nghiêm trọng khi mã hóa ảnh: {e}")
        return None

# Hàm save_violation_frame cũ không còn cần thiết nếu chỉ lưu vào DB
# Bạn có thể xóa hoặc giữ lại nếu muốn tùy chọn lưu cả file
# def save_violation_frame(frame, output_dir, timestamp):
#     ...

# Hàm log_violation cũ không còn cần thiết
# def log_violation(frame_path, timestamp):
#     ...

# Hàm save_violations_to_excel cũ không còn cần thiết
# def save_violations_to_excel(output_excel_path):
#    ...

# --- Interactive Line Selection ---
drawing = False # true if mouse is pressed
stop_line_y = -1

def select_line_callback(event, x, y, flags, param):
    """Callback function for mouse events to draw line."""
    global stop_line_y, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        stop_line_y = y # Chỉ cần y cho đường ngang

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            stop_line_y = y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        print(f"Đã chọn vạch dừng tại y = {stop_line_y}")

def select_stop_line(first_frame):
    """Hiển thị frame đầu tiên và cho phép người dùng vẽ vạch dừng."""
    global stop_line_y
    stop_line_y = -1 # Reset
    clone = first_frame.copy()
    cv2.namedWindow("Select Stop Line - Draw a horizontal line and press ENTER")
    cv2.setMouseCallback("Select Stop Line - Draw a horizontal line and press ENTER", select_line_callback)

    while True:
        temp_frame = clone.copy()
        if stop_line_y != -1:
            cv2.line(temp_frame, (0, stop_line_y), (temp_frame.shape[1], stop_line_y), (0, 0, 255), 2)
        cv2.imshow("Select Stop Line - Draw a horizontal line and press ENTER", temp_frame)
        key = cv2.waitKey(1) & 0xFF

        # Nhấn ENTER để xác nhận, ESC để hủy
        if key == 13: # Enter key
            if stop_line_y != -1:
                break
            else:
                print("Bạn chưa vẽ đường nào. Hãy click và kéo để vẽ đường ngang.")
        elif key == 27: # Esc key
             stop_line_y = None # Hủy bỏ
             print("Đã hủy chọn vạch dừng.")
             break

    cv2.destroyWindow("Select Stop Line - Draw a horizontal line and press ENTER")
    return stop_line_y

# --- Interactive ROI Selection ---
roi_coords = None
drawing_roi = False
roi_start_point = (-1, -1)

def mouse_callback_roi(event, x, y, flags, param):
    """Callback function để xử lý sự kiện click chuột chọn ROI."""
    global points, drawing, selected_roi, temp_frame_roi # Thêm temp_frame_roi
    window_name = param['window_name']
    frame = param['frame']

    if event == cv2.EVENT_LBUTTONDOWN:
        points = [(x, y)]
        drawing = True
        # print(f"Điểm bắt đầu ROI: ({x}, {y})") # Giảm bớt thông báo
        temp_frame_roi = frame.copy() # Lưu frame gốc khi bắt đầu vẽ
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Vẽ hình chữ nhật tạm thời trên bản sao
            img_copy = temp_frame_roi.copy()
            cv2.rectangle(img_copy, points[0], (x, y), (0, 255, 0), 1)
            cv2.imshow(window_name, img_copy)
    elif event == cv2.EVENT_LBUTTONUP:
        if drawing:
            points.append((x, y))
            drawing = False
            x1, y1 = points[0]
            x2, y2 = points[1]
            # Đảm bảo tọa độ đúng thứ tự và có kích thước > 0
            start_x, start_y = min(x1, x2), min(y1, y2)
            end_x, end_y = max(x1, x2), max(y1, y2)
            if end_x > start_x and end_y > start_y:
                selected_roi = (start_x, start_y, end_x - start_x, end_y - start_y) # (x, y, w, h)
                print(f"Callback: Đã chọn ROI: {selected_roi}")
                # Vẽ ROI cuối cùng lên frame gốc tạm thời để hiển thị
                final_display_frame = temp_frame_roi.copy()
                cv2.rectangle(final_display_frame, (selected_roi[0], selected_roi[1]),
                              (selected_roi[0] + selected_roi[2], selected_roi[1] + selected_roi[3]),
                              (0, 255, 0), 2)
                cv2.imshow(window_name, final_display_frame)
                # Không cần yêu cầu nhấn phím ở đây nữa
            else:
                print("Callback: ROI không hợp lệ (kích thước bằng 0). Vui lòng vẽ lại.")
                # Reset để người dùng có thể vẽ lại
                points = []
                selected_roi = None
                cv2.imshow(window_name, frame) # Hiển thị lại frame gốc

# Biến tạm thời để callback sử dụng
temp_frame_roi = None

def select_roi(frame, window_name="Chon ROI den giao thong"):
    """Hiển thị frame và cho phép người dùng kéo chuột để chọn ROI."""
    global points, drawing, selected_roi, temp_frame_roi
    points = []
    drawing = False
    selected_roi = None # Reset ROI
    temp_frame_roi = frame.copy() # Chuẩn bị frame cho callback

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback_roi, {'frame': frame, 'window_name': window_name})
    print("Kéo chuột trên cửa sổ để chọn vùng chứa đèn giao thông. Nhấn ESC để hủy.")

    while selected_roi is None:
        # Chỉ hiển thị frame gốc trong vòng lặp chính
        # Việc hiển thị hình chữ nhật tạm thời đã được xử lý trong callback MOUSEMOVE
        cv2.imshow(window_name, temp_frame_roi)
        key = cv2.waitKey(20) & 0xFF # Tăng nhẹ waitKey để giảm CPU usage
        if key == 27: # ESC key
            cv2.destroyWindow(window_name)
            print("Đã hủy chọn ROI.")
            return None
        # Vòng lặp sẽ tự thoát khi selected_roi được callback gán giá trị khác None

    # Xác nhận lần cuối và đóng cửa sổ
    print("Hàm select_roi: Đã ghi nhận ROI.")
    # Có thể thêm delay nhỏ để người dùng thấy ROI cuối cùng trước khi cửa sổ đóng
    # cv2.waitKey(500)
    cv2.destroyWindow(window_name)
    return selected_roi

# --- New Function to Detect and Crop License Plate Image --- 
def detect_and_crop_plate(vehicle_roi):
    """Detects potential license plate contour in the vehicle ROI and returns the cropped COLOR image.

    Args:
        vehicle_roi: The COLOR image (NumPy array) of the vehicle ROI.

    Returns:
        The cropped COLOR license plate image (NumPy array) or None if not found.
    """
    if vehicle_roi is None or vehicle_roi.size == 0:
        return None

    # Tạo thư mục debug nếu chưa tồn tại
    debug_dir = r"C:\Users\MY PC\Máy tính\red_light_violation_system\debug_crops"
    os.makedirs(debug_dir, exist_ok=True)
    
    # Cấu hình debug
    DEBUG_CROP = False  # Đặt thành True để lưu ảnh debug
    
    # Lưu ảnh xe gốc cho debug
    if DEBUG_CROP:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        vehicle_path = os.path.join(debug_dir, f"vehicle_{timestamp}.jpg")
        cv2.imwrite(vehicle_path, vehicle_roi)
        print(f"DEBUG CROP: Đã lưu ảnh xe tại: {vehicle_path}")

    detected_plate_image = None
    try:
        # 1. Preprocessing for contour detection
        gray = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.bilateralFilter(gray, 9, 75, 75) 
        # Adjust Canny thresholds if needed (e.g., 30, 120 or 70, 200)
        edged = cv2.Canny(blurred, 50, 150) 
        
        # (Optional) Display edged image for debugging
        if DEBUG_CROP:
            cv2.imshow("Edged Vehicle ROI for Plate Detection", edged)

        # --- ADD Morphological Closing --- 
        # Use a rectangular kernel, size might need tuning
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)) # Wider than tall
        closed_edges = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=1)
        # (Optional) Display closed edges for debugging
        if DEBUG_CROP:
            cv2.imshow("Closed Edges", closed_edges)

        # 2. Find Contours (on the closed edges image)
        contours, _ = cv2.findContours(closed_edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15] # Check slightly more contours

        # 3. Filter for Plate Contour
        plate_contour = None
        for c in contours:
            perimeter = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.018 * perimeter, True) # Approximation accuracy

            if len(approx) == 4: # Check if it's a quadrilateral
                (x, y, w, h) = cv2.boundingRect(approx)
                
                # Calculate additional properties for filtering
                area = w * h
                contour_area = cv2.contourArea(c)
                if area == 0: continue # Avoid division by zero
                
                aspect_ratio = float(w) / h
                extent = float(contour_area) / area
                hull = cv2.convexHull(c)
                hull_area = cv2.contourArea(hull)
                solidity = float(contour_area) / hull_area if hull_area > 0 else 0
                
                # --- ADJUST Filtering Criteria --- 
                min_area_threshold = 100 # Minimum pixel area
                max_area_threshold = vehicle_roi.shape[0] * vehicle_roi.shape[1] * 0.3 # Max 30% of ROI area
                min_aspect = 2.0
                max_aspect = 5.5
                min_solidity = 0.90 # Contour should be mostly solid
                min_extent = 0.65  # Area vs Bounding Box ratio
                
                # DEBUG: In thông số của contour ứng viên
                if DEBUG_CROP:
                    print(f"DEBUG CROP: Contour: AR={aspect_ratio:.2f}, Area={area}, Solidity={solidity:.2f}, Extent={extent:.2f}")
                
                # Apply all filters
                if (min_area_threshold < area < max_area_threshold and
                    min_aspect < aspect_ratio < max_aspect and
                    solidity > min_solidity and
                    extent > min_extent):
                    
                    plate_contour = approx
                    if DEBUG_CROP:
                        print(f"DEBUG CROP: Phát hiện biển số - AR={aspect_ratio:.2f}, Area={area}, Solidity={solidity:.2f}, Extent={extent:.2f}")
                    break # Found a likely plate

        # 4. Crop the plate from the original COLOR vehicle ROI if found
        if plate_contour is not None:
            x, y, w, h = cv2.boundingRect(plate_contour)
            y1, y2 = max(0, y), min(vehicle_roi.shape[0], y + h)
            x1, x2 = max(0, x), min(vehicle_roi.shape[1], x + w)
            
            if x2 > x1 and y2 > y1: 
                detected_plate_image = vehicle_roi[y1:y2, x1:x2] 
                
                # Lưu ảnh biển số đã crop
                if DEBUG_CROP:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    plate_path = os.path.join(debug_dir, f"plate_{timestamp}.jpg")
                    cv2.imwrite(plate_path, detected_plate_image)
                    print(f"DEBUG CROP: Đã lưu ảnh biển số tại: {plate_path}")
                    cv2.imshow("Detected Plate Image", detected_plate_image)
            else:
                if DEBUG_CROP:
                    print("Warning: Bounding rect invalid after boundary check.")
        else:
            if DEBUG_CROP:
                print("DEBUG CROP: Không tìm thấy biển số phù hợp sau khi lọc contour.")

    except Exception as e:
        if DEBUG_CROP:
            print(f"Error during plate detection/cropping: {e}")
            import traceback
            traceback.print_exc()
        detected_plate_image = None

    return detected_plate_image
