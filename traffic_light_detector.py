import tensorflow as tf
import numpy as np
import cv2
from utils import preprocess_image_for_cnn, CNN_INPUT_SIZE, TRAFFIC_LIGHT_CLASSES

class TrafficLightDetector:
    """Phát hiện trạng thái đèn giao thông sử dụng mô hình CNN đã huấn luyện."""

    def __init__(self, model_path):
        """
        Khởi tạo bộ phát hiện đèn giao thông.

        Args:
            model_path (str): Đường dẫn đến file model .h5 đã huấn luyện.
        """
        self.model_path = model_path
        self.model = None
        self.input_size = CNN_INPUT_SIZE
        # Tạo map ngược từ ID sang tên lớp (Red, Yellow, Green)
        self.class_names = {v: k for k, v in TRAFFIC_LIGHT_CLASSES.items()}
        self.load_model()

    def load_model(self):
        """Tải mô hình Keras từ file."""
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            print(f"Đã tải thành công mô hình CNN từ: {self.model_path}")
        except Exception as e:
            print(f"Lỗi: Không thể tải mô hình CNN từ {self.model_path}. Lỗi: {e}")
            print("Hãy đảm bảo bạn đã huấn luyện và lưu model vào đúng đường dẫn.")
            self.model = None # Đặt lại thành None nếu lỗi

    def predict(self, image_roi):
        """
        Dự đoán trạng thái đèn giao thông từ một vùng ảnh (ROI).

        Args:
            image_roi (numpy.ndarray): Ảnh crop chứa đèn giao thông.

        Returns:
            str: Tên trạng thái đèn ('RED', 'YELLOW', 'GREEN', hoặc 'UNKNOWN').
            float: Độ tin cậy (confidence score) của dự đoán.
        """
        if self.model is None or image_roi is None or image_roi.size == 0:
            return "UNKNOWN", 0.0

        try:
            # 1. Tiền xử lý ảnh ROI
            processed_roi = preprocess_image_for_cnn(image_roi) # Resize và chuẩn hóa

            # 2. Mở rộng chiều batch (model mong đợi batch ảnh)
            input_batch = np.expand_dims(processed_roi, axis=0)

            # 3. Dự đoán bằng model
            predictions = self.model.predict(input_batch)
            # predictions shape sẽ là (1, num_classes), ví dụ: [[0.1, 0.8, 0.1]]

            # 4. Lấy kết quả
            predicted_class_id = np.argmax(predictions[0])
            confidence = np.max(predictions[0])

            # 5. Chuyển ID sang tên lớp
            # Dùng get để tránh lỗi nếu ID không có trong map
            predicted_class_name = self.class_names.get(predicted_class_id, "UNKNOWN").upper()

            # Ánh xạ tên lớp tiếng Việt sang tiếng Anh (hoặc chuẩn hóa)
            if predicted_class_name == "DEN DO":
                predicted_class_name = "RED"
            elif predicted_class_name == "DEN VANG":
                predicted_class_name = "YELLOW"
            elif predicted_class_name == "DEN XANH":
                predicted_class_name = "GREEN"

            return predicted_class_name, float(confidence)

        except Exception as e:
            print(f"Lỗi trong quá trình dự đoán đèn giao thông: {e}")
            return "UNKNOWN", 0.0

# Ví dụ sử dụng (chạy khi file này được thực thi trực tiếp)
if __name__ == '__main__':
    # Giả sử bạn có một ảnh và ROI của đèn
    test_image_path = '../data/images/road1.png' # Thay bằng ảnh thực tế có đèn
    # Tọa độ ROI (x, y, w, h) - Cần chỉnh lại cho đúng với ảnh của bạn
    roi_x, roi_y, roi_w, roi_h = 500, 50, 50, 150

    image = cv2.imread(test_image_path)
    if image is None:
        print(f"Không đọc được ảnh test: {test_image_path}")
    else:
        # Crop ROI
        traffic_light_roi = image[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]

        if traffic_light_roi.size > 0:
            cv2.imshow("Test Traffic Light ROI", traffic_light_roi)
            cv2.waitKey(100) # Chờ để hiển thị

            # Đường dẫn model đã train
            model_file = '../trained_models/traffic_light_cnn.h5'

            # Khởi tạo detector
            detector = TrafficLightDetector(model_file)

            # Dự đoán
            if detector.model: # Chỉ dự đoán nếu model load thành công
                light_state, confidence = detector.predict(traffic_light_roi)
                print(f"Trạng thái đèn dự đoán: {light_state} (Confidence: {confidence:.2f})")

                # Vẽ kết quả lên ảnh gốc
                cv2.rectangle(image, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 255, 0), 2)
                cv2.putText(image, f"{light_state} ({confidence:.2f})", (roi_x, roi_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.imshow("Test Image with Prediction", image)
                print("Nhấn phím bất kỳ để thoát.")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                 print("Không thể thực hiện dự đoán do model chưa được tải.")
        else:
            print("ROI không hợp lệ (kích thước bằng 0).")