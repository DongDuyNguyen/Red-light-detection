import cv2
import numpy as np
import re
import os
import logging
import datetime

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('license_plate_reader')

# Thêm thư mục debug để lưu ảnh
DEBUG_DIR = r"C:\Users\MY PC\Máy tính\red_light_violation_system\debug_ocr"
os.makedirs(DEBUG_DIR, exist_ok=True)
DEBUG_ENABLED = False  # Tắt debug - đặt lại thành True nếu muốn debug

# In ra vị trí thư mục debug để dễ tìm
if DEBUG_ENABLED:
    print(f"DEBUG OCR: Tạo thư mục debug tại: {os.path.abspath(DEBUG_DIR)}")
    # Tạo file test để xác nhận thư mục hoạt động
    try:
        with open(os.path.join(DEBUG_DIR, 'readme.txt'), 'w') as f:
            f.write('Thư mục debug cho OCR biển số xe')
        print(f"DEBUG OCR: Đã tạo file test trong thư mục debug")
    except Exception as e:
        print(f"DEBUG OCR: Lỗi khi tạo file test: {e}")
else:
    print("Debug OCR đã bị vô hiệu hóa. Đặt DEBUG_ENABLED = True để kích hoạt lại khi cần.")

# Thử import pytesseract, xử lý lỗi nếu không có
try:
    import pytesseract
    # Tự động tìm kiếm đường dẫn Tesseract trên Windows
    if os.name == 'nt':  # Windows
        tesseract_paths = [
            r'C:\Users\MY PC\AppData\Local\Programs\Tesseract-OCR\tesseract.exe',  # Đường dẫn mới người dùng đã cài đặt
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        ]
        for path in tesseract_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                logger.info(f"Đã tìm thấy Tesseract tại: {path}")
                break
    HAS_TESSERACT = True
except ImportError:
    logger.error("Không thể import pytesseract. Cài đặt với lệnh: pip install pytesseract")
    logger.error("Đồng thời cài đặt Tesseract OCR từ: https://github.com/UB-Mannheim/tesseract/wiki")
    HAS_TESSERACT = False

# Mẫu biển số phổ biến để đối chiếu
COMMON_PLATE_PATTERNS = [
    r'^[A-Z]\d{3}[A-Z]{2}$',      # L605HZ
    r'^[A-Z]-\d{3}-[A-Z]{2}$',     # L-605-HZ
    r'^[A-Z]\d{2}[A-Z]\d{3}$',     # AB12C345
    r'^\d{2}[A-Z]\d{4,5}$'         # 12A34567
]

def preprocess_plate_for_ocr(plate_image):
    """Tiền xử lý ảnh biển số để nâng cao độ chính xác OCR"""
    if plate_image is None or plate_image.size == 0:
        logger.warning("Ảnh biển số rỗng hoặc không hợp lệ")
        return []
    
    try:
        # Đảm bảo kích thước đủ lớn
        h, w = plate_image.shape[:2]
        min_width = 200
        if w < min_width:
            ratio = min_width / w
            plate_image = cv2.resize(plate_image, (min_width, int(h * ratio)), interpolation=cv2.INTER_CUBIC)
        
        # Chuyển sang ảnh xám nếu là ảnh màu
        if len(plate_image.shape) == 3:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_image.copy()
        
        # Tạo nhiều phiên bản tiền xử lý
        results = []
        
        # Thêm ảnh gốc đã chuyển xám
        results.append(gray)
        
        # Phiên bản 1: Cân bằng histogram + Bilateral Filter + Adaptive Threshold
        img1 = cv2.equalizeHist(gray)
        img1 = cv2.bilateralFilter(img1, 11, 17, 17)
        img1 = cv2.adaptiveThreshold(img1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 11, 2)
        results.append(img1)
        
        # Phiên bản 2: CLAHE + Otsu
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img2 = clahe.apply(gray)
        _, img2 = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results.append(img2)
        
        # Phiên bản 3: Sharpen + Threshold
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        img3 = cv2.filter2D(gray, -1, kernel)
        _, img3 = cv2.threshold(img3, 100, 255, cv2.THRESH_BINARY)
        results.append(img3)
        
        # Phiên bản 4: Bilateral Filter + Threshold trung bình
        img4 = cv2.bilateralFilter(gray, 11, 17, 17)
        mean_val = np.mean(img4)
        _, img4 = cv2.threshold(img4, mean_val, 255, cv2.THRESH_BINARY)
        results.append(img4)
        
        # Phiên bản 5: Phủ định - hữu ích cho biển nền sáng chữ tối
        img5 = cv2.bitwise_not(img1)
        results.append(img5)
        
        return results
    except Exception as e:
        logger.error(f"Lỗi khi tiền xử lý ảnh biển số: {e}")
        import traceback
        traceback.print_exc()
        return []

def clean_plate_text(text):
    """Làm sạch và định dạng lại text OCR"""
    if not text:
        return ""
    
    # Loại bỏ ký tự đặc biệt và khoảng trắng
    text = re.sub(r'[^\w\-]', '', text)
    
    # Chuyển đổi về chữ in hoa
    text = text.upper()
    
    # Sửa ký tự dễ nhầm lẫn
    replacements = {
        'O': '0', 'D': '0', 'Q': '0',
        'I': '1', 'L': '1',
        'Z': '2',
        'A': '4',
        'S': '5',
        'B': '8'
    }
    
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    
    # Áp dụng định dạng mẫu biển số nếu gần khớp
    # Ví dụ: L605HZ -> L-605-HZ
    if len(text) >= 6 and '-' not in text:
        if re.match(r'^[A-Z]\d{3}[A-Z]{2}$', text):
            text = f"{text[0]}-{text[1:4]}-{text[4:]}"
    
    return text

def evaluate_plate_text(text):
    """Đánh giá độ phù hợp của text biển số"""
    if not text or len(text) < 3:
        return 0
    
    score = 0
    
    # Điểm cho độ dài phù hợp
    if 5 <= len(text) <= 9:
        score += 30
    
    # Điểm cho tỷ lệ chữ/số hợp lý
    letters = sum(1 for c in text if c.isalpha())
    digits = sum(1 for c in text if c.isdigit())
    if digits >= 2 and letters >= 1:
        score += 30
    
    # Điểm cho định dạng
    if '-' in text:
        score += 20
    
    # Điểm cho mẫu phổ biến
    for pattern in COMMON_PLATE_PATTERNS:
        if re.match(pattern, text):
            score += 20
            break
    
    return min(score, 100)

def read_license_plate_fast(plate_image):
    """Đọc biển số xe từ ảnh đã cắt - phiên bản nhanh"""
    if not HAS_TESSERACT:
        logger.error("Tesseract OCR không khả dụng")
        return "ERROR_NO_TESSERACT", 0.0
    
    if plate_image is None or plate_image.size == 0:
        logger.warning("Ảnh biển số rỗng")
        return "", 0.0
    
    try:
        # Tạo timestamp cho tên file debug
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        debug_prefix = os.path.join(DEBUG_DIR, f"plate_{timestamp}")
        
        # Lưu ảnh gốc để kiểm tra
        if DEBUG_ENABLED:
            original_path = f"{debug_prefix}_1_original.jpg"
            cv2.imwrite(original_path, plate_image)
            logger.info(f"Đã lưu ảnh biển số gốc: {original_path}")
            cv2.imshow("Original Plate", plate_image)
            cv2.waitKey(100)  # Hiển thị trong 100ms
        
        # Tiền xử lý ảnh với phương pháp đơn giản nhất
        if len(plate_image.shape) == 3:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_image.copy()
        
        # Lưu ảnh grayscale
        if DEBUG_ENABLED:
            gray_path = f"{debug_prefix}_2_gray.jpg"
            cv2.imwrite(gray_path, gray)
            cv2.imshow("Grayscale Plate", gray)
            cv2.waitKey(100)
        
        # Tăng kích thước nếu ảnh quá nhỏ
        h, w = gray.shape
        min_width = 200
        if w < min_width:
            ratio = min_width / w
            gray = cv2.resize(gray, (min_width, int(h * ratio)), interpolation=cv2.INTER_CUBIC)
            
            if DEBUG_ENABLED:
                resized_path = f"{debug_prefix}_3_resized.jpg"
                cv2.imwrite(resized_path, gray)
                cv2.imshow("Resized Plate", gray)
                cv2.waitKey(100)
        
        # --- PHƯƠNG PHÁP MỚI: TƯƠNG TỰ PHƯƠNG PHÁP TÌM CONTOUR ---
        # Phương pháp này bắt chước tiền xử lý trong detect_and_crop_plate()
        
        # 1. Áp dụng bilateral filter (giảm nhiễu, giữ cạnh)
        blurred = cv2.bilateralFilter(gray, 11, 90, 90)
        if DEBUG_ENABLED:
            blurred_path = f"{debug_prefix}_4_bilateral.jpg"
            cv2.imwrite(blurred_path, blurred)
            cv2.imshow("Bilateral Filtered", blurred)
            cv2.waitKey(100)
        
        # 2. Áp dụng Canny edge detection
        edges = cv2.Canny(blurred, 30, 200)  # Thử cả ngưỡng thấp và cao
        if DEBUG_ENABLED:
            edges_path = f"{debug_prefix}_5_canny.jpg"
            cv2.imwrite(edges_path, edges)
            cv2.imshow("Canny Edges", edges)
            cv2.waitKey(100)
        
        # 3. Áp dụng Morphological Closing để kết nối các cạnh
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        if DEBUG_ENABLED:
            closed_path = f"{debug_prefix}_6_closed.jpg"
            cv2.imwrite(closed_path, closed_edges)
            cv2.imshow("Closed Edges", closed_edges)
            cv2.waitKey(100)
        
        # 4. Tạo mask từ closed edges
        # Ngược với bước phát hiện contour: 
        # thay vì tìm contour, ta dùng trực tiếp ảnh này để làm rõ ký tự
        mask = closed_edges.copy()
        
        # 5. Tạo ảnh chỉ có ký tự đen nền trắng và ngược lại
        # Tạo 2 phiên bản: bình thường và đảo ngược
        text_binary = cv2.bitwise_not(mask)  # Ký tự trắng trên nền đen
        text_binary_inv = mask.copy()  # Ký tự đen trên nền trắng
        
        if DEBUG_ENABLED:
            binary_path = f"{debug_prefix}_7_binary.jpg"
            binary_inv_path = f"{debug_prefix}_7_binary_inv.jpg"
            cv2.imwrite(binary_path, text_binary)
            cv2.imwrite(binary_inv_path, text_binary_inv)
            cv2.imshow("Binary Text (white on black)", text_binary)
            cv2.imshow("Binary Text (black on white)", text_binary_inv)
            cv2.waitKey(100)
        
        # --- KẾT THÚC PHƯƠNG PHÁP MỚI ---
        
        # Áp dụng tiền xử lý cơ bản (phương pháp cũ, giữ lại để so sánh)
        # Phương pháp 1: Cân bằng histogram
        img1 = cv2.equalizeHist(gray)
        if DEBUG_ENABLED:
            equalized_path = f"{debug_prefix}_8_equalized.jpg"
            cv2.imwrite(equalized_path, img1)
            cv2.imshow("Equalized Plate", img1)
            cv2.waitKey(100)
        
        # Phương pháp 2: Lọc bilateral
        img2 = cv2.bilateralFilter(img1, 11, 17, 17)
        if DEBUG_ENABLED:
            bilateral_path = f"{debug_prefix}_9_bilateral_old.jpg"
            cv2.imwrite(bilateral_path, img2)
            cv2.imshow("Bilateral Filtered (old)", img2)
            cv2.waitKey(100)
        
        # Phương pháp 3: Thresholding
        _, img3 = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if DEBUG_ENABLED:
            threshold_path = f"{debug_prefix}_10_threshold.jpg"
            cv2.imwrite(threshold_path, img3)
            cv2.imshow("Thresholded Plate", img3)
            cv2.waitKey(100)
        
        # Phương pháp 4: Thử nghiệm với ảnh âm bản
        img4 = cv2.bitwise_not(img3)
        if DEBUG_ENABLED:
            inverted_path = f"{debug_prefix}_11_inverted.jpg"
            cv2.imwrite(inverted_path, img4)
            cv2.imshow("Inverted Plate", img4)
            cv2.waitKey(100)
        
        # Whitelist các ký tự cho biển số
        whitelist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
        
        # Thử OCR với nhiều cấu hình khác nhau
        results = []
        
        # Cấu hình OCR và thực hiện OCR
        configs = [
            ('7', text_binary, "Canny Binary"),     # Phương pháp mới: chữ trắng nền đen
            ('7', text_binary_inv, "Canny Binary Inv"), # Phương pháp mới: chữ đen nền trắng
            ('7', img3, "Thresh Binary"),     # Phương pháp cũ: chữ đen nền trắng
            ('7', img4, "Thresh Binary Inv"), # Phương pháp cũ: chữ trắng nền đen
            ('6', text_binary, "PSM6 Canny Binary"), # Block of text với phương pháp mới
            ('8', text_binary, "PSM8 Canny Binary"), # Single word với phương pháp mới
            ('10', text_binary, "PSM10 Canny Binary") # Ký tự đơn với phương pháp mới
        ]
        
        for psm, img, desc in configs:
            config = f'--oem 3 --psm {psm} -c tessedit_char_whitelist={whitelist}'
            try:
                text = pytesseract.image_to_string(img, config=config).strip()
                clean_text = clean_plate_text(text)
                
                if clean_text:
                    # Đánh giá độ tin cậy cơ bản
                    confidence = 0.5  # Giá trị mặc định
                    
                    # Tăng điểm tin cậy dựa vào độ dài hợp lý
                    if 5 <= len(clean_text) <= 9:
                        confidence = 0.7
                    
                    # Tăng điểm tin cậy nếu có cả chữ và số
                    letters = sum(1 for c in clean_text if c.isalpha())
                    digits = sum(1 for c in clean_text if c.isdigit())
                    if digits >= 2 and letters >= 1:
                        confidence = min(confidence + 0.1, 1.0)
                    
                    # Tăng điểm nếu có dấu gạch ngang
                    if '-' in clean_text:
                        confidence = min(confidence + 0.1, 1.0)
                    
                    results.append((clean_text, confidence, desc))
                    logger.info(f"OCR ({desc}): '{clean_text}' (độ tin cậy: {confidence:.2f})")
            except Exception as e:
                logger.error(f"Lỗi OCR ({desc}): {e}")
        
        # Nếu có kết quả, chọn kết quả tốt nhất
        if results:
            # Sắp xếp theo độ tin cậy
            results.sort(key=lambda x: x[1], reverse=True)
            best_result = results[0]
            
            # Lưu ảnh kết quả với text OCR
            if DEBUG_ENABLED:
                result_img = plate_image.copy() if len(plate_image.shape) == 3 else cv2.cvtColor(plate_image, cv2.COLOR_GRAY2BGR)
                cv2.putText(result_img, f"{best_result[0]} ({best_result[1]:.2f})", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                result_path = f"{debug_prefix}_12_result_{best_result[0]}.jpg"
                cv2.imwrite(result_path, result_img)
                
                # Hiển thị kết quả
                cv2.imshow("OCR Result", result_img)
                cv2.waitKey(500)  # Hiển thị lâu hơn
            
            return best_result[0], best_result[1]
        
        logger.warning(f"Không đọc được biển số từ ảnh biển")
        return "", 0.0
        
    except Exception as e:
        logger.error(f"Lỗi không xác định khi đọc biển số: {e}")
        import traceback
        traceback.print_exc()
        return "", 0.0

# Sử dụng hàm nhanh thay thế hàm chi tiết
read_license_plate = read_license_plate_fast

# Test trực tiếp
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if os.path.exists(image_path):
            # Đọc ảnh
            plate_img = cv2.imread(image_path)
            
            # Xử lý OCR
            text, confidence = read_license_plate(plate_img)
            
            print(f"Biển số: {text}")
            print(f"Độ tin cậy: {confidence:.2f}")
            
            # Hiển thị kết quả
            if plate_img is not None:
                result_img = plate_img.copy()
                cv2.putText(result_img, f"{text} ({confidence:.2f})", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("License Plate", result_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            print(f"Lỗi: Không tìm thấy file ảnh {image_path}")
    else:
        print("Sử dụng: python license_plate_reader.py [đường_dẫn_ảnh_biển_số]") 