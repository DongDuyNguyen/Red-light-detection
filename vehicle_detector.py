import cv2
import numpy as np

class VehicleDetector:
    """Phát hiện phương tiện đang di chuyển bằng Background Subtraction."""

    def __init__(self, history=500, varThreshold=16, detectShadows=True, min_contour_area=500):
        """
        Khởi tạo bộ phát hiện phương tiện.

        Args:
            history (int): Số frame dùng để xây dựng background model.
            varThreshold (int): Ngưỡng phương sai để xác định pixel thuộc foreground.
                                Giá trị cao hơn sẽ ít nhạy cảm hơn.
            detectShadows (bool): Có phát hiện và đánh dấu bóng hay không.
            min_contour_area (int): Ngưỡng diện tích tối thiểu để lọc contour.
        """
        # Sử dụng MOG2 là một thuật toán phổ biến và hiệu quả
        self.backSub = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=varThreshold,
            detectShadows=detectShadows
        )
        # Kernel cho morphological operations
        self.kernel_open = np.ones((5, 5), np.uint8) # Kernel lớn hơn cho Opening (5x5)
        self.kernel_close = np.ones((7, 7), np.uint8) # Giữ nguyên kernel Closing

        # Ngưỡng diện tích tối thiểu để coi là một phương tiện
        self.min_contour_area = min_contour_area # Lấy giá trị từ tham số

    def detect(self, frame):
        """
        Phát hiện phương tiện trong một frame video.

        Args:
            frame (numpy.ndarray): Frame ảnh đầu vào (BGR).

        Returns:
            tuple: (list of bounding boxes [(x, y, w, h)], processed_mask)
                   processed_mask là mask sau khi xử lý hình thái học (để debug nếu cần).
        """
        if frame is None:
            return [], None

        # 1. Áp dụng Background Subtraction
        fg_mask = self.backSub.apply(frame)

        # 2. Loại bỏ bóng (shadows - giá trị 127)
        # Chỉ giữ lại các pixel chắc chắn là foreground (giá trị 255)
        fg_mask_no_shadows = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)[1]

        # 3. Áp dụng Morphological Operations để làm sạch mask
        # Opening: Loại bỏ nhiễu trắng nhỏ
        mask_opened = cv2.morphologyEx(fg_mask_no_shadows, cv2.MORPH_OPEN, self.kernel_open)
        # Closing: Lấp đầy lỗ đen nhỏ, nối liền vùng trắng
        mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, self.kernel_close)

        # Lưu mask đã xử lý để trả về (hữu ích cho việc debug)
        processed_mask = mask_closed

        # 4. Tìm contours trên mask đã xử lý
        contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 5. Lọc contours và lấy bounding boxes
        detected_boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Lọc theo diện tích tối thiểu
            if area > self.min_contour_area:
                x, y, w, h = cv2.boundingRect(cnt)
                # (Tùy chọn) Thêm các bộ lọc khác ở đây (aspect ratio, solidity)
                detected_boxes.append((x, y, w, h))

        return detected_boxes, processed_mask # Trả về cả mask đã xử lý

# Ví dụ sử dụng (chạy khi file này được thực thi trực tiếp)
if __name__ == '__main__':
    # Đường dẫn đến video test
    video_path = r"C:\Users\MY PC\Máy tính\red_light_violation_system/tr.mp4" # Thay bằng video của bạn

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Lỗi: Không thể mở video: {video_path}")
        exit()

    # Khởi tạo detector
    vehicle_detector = VehicleDetector(varThreshold=50, detectShadows=False) # Thử nghiệm với tham số khác

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Kết thúc video hoặc lỗi đọc frame.")
            break

        # Resize frame cho dễ nhìn (tùy chọn)
        frame_resized = cv2.resize(frame, (960, 540))

        # Phát hiện phương tiện
        vehicles, fg_mask = vehicle_detector.detect(frame_resized)

        # Vẽ bounding box lên frame
        for (x, y, w, h) in vehicles:
            cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Hiển thị kết quả
        cv2.imshow("Detected Vehicles", frame_resized)
        if fg_mask is not None:
            cv2.imshow("Foreground Mask", fg_mask)

        # Nhấn 'q' để thoát
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()