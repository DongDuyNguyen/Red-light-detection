# Script để đánh giá model CNN trên tập dữ liệu test 
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import các hàm tiện ích
from utils import load_coco_annotations, extract_traffic_light_patches, preprocess_image_for_cnn

# --- Cấu hình --- (Phải khớp với cấu hình lúc huấn luyện)
IMG_WIDTH = 64
IMG_HEIGHT = 64
IMG_CHANNELS = 3
NUM_CLASSES = 3 # (0: RED, 1: YELLOW, 2: GREEN)
CLASS_NAMES = ["RED", "YELLOW", "GREEN"]

# Xác định đường dẫn tuyệt đối
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

# Đường dẫn đến model đã huấn luyện (phiên bản dùng train_test_split)
MODEL_PATH = os.path.join(PROJECT_ROOT, 'trained_models', 'traffic_light_cnn_noaug_best.keras')

# Đường dẫn đến dữ liệu test
TEST_ANNOTATION_PATH = os.path.join(PROJECT_ROOT, 'dataset', 'test', '_annotations.coco.json')
TEST_IMG_DIR = os.path.join(PROJECT_ROOT, 'dataset', 'test', 'images')

# Đường dẫn lưu confusion matrix
CONFUSION_MATRIX_SAVE_PATH = os.path.join(PROJECT_ROOT, 'test_confusion_matrix_percent.png')
# --- Kết thúc Cấu hình ---

def load_and_preprocess_test_data(annotation_path, img_dir, img_width, img_height):
    """Tải và tiền xử lý dữ liệu test (tương tự train nhưng không có augmentation)."""
    print("--- Đang xử lý tập test ---")
    print(f"Annotation file: {annotation_path}")
    print(f"Image directory: {img_dir}")

    if not os.path.exists(annotation_path) or not os.path.isdir(img_dir):
        print("Lỗi: Không tìm thấy file annotation hoặc thư mục ảnh của tập test.")
        return None, None

    # Sử dụng hàm load_coco_annotations đã được sửa đổi
    image_paths, bboxes, labels = load_coco_annotations(annotation_path, img_dir)
    if image_paths is None:
        print("Lỗi khi tải dữ liệu test.")
        return None, None

    print(f"Đang trích xuất và tiền xử lý ảnh test...")
    light_patches, patch_labels = extract_traffic_light_patches(image_paths, bboxes, labels)

    if light_patches.size == 0:
        print("Lỗi: Không trích xuất được patch ảnh nào từ tập test.")
        return None, None

    print(f"Đang chuẩn hóa ảnh test...")
    processed_images_list = [preprocess_image_for_cnn(patch, target_size=(img_width, img_height)) for patch in light_patches]
    valid_indices = [i for i, img in enumerate(processed_images_list) if img is not None]

    if not valid_indices:
        print("Lỗi: Không có ảnh test nào được tiền xử lý thành công.")
        return None, None

    X_test = np.array([processed_images_list[i] for i in valid_indices])
    y_test = np.array([patch_labels[i] for i in valid_indices])

    print(f"Hoàn tất xử lý tập test thành công.")
    return X_test, y_test

# === Hàm chính thực thi ===
if __name__ == "__main__":
    print(f"--- Bắt đầu đánh giá model {os.path.basename(MODEL_PATH)} trên tập test ---")

    # --- 1. Kiểm tra sự tồn tại của model ---
    if not os.path.exists(MODEL_PATH):
        print(f"Lỗi: Không tìm thấy file model đã huấn luyện tại: {MODEL_PATH}")
        print("Vui lòng chạy script huấn luyện trước (ví dụ: train_cnn.py).")
        exit()

    # --- 2. Tải model đã huấn luyện ---
    print(f"Đang tải model từ {MODEL_PATH}...")
    try:
        model = load_model(MODEL_PATH)
        print("Tải model thành công.")
        # Ẩn model.summary() để không hiển thị thông tin chi tiết
    except Exception as e:
        print(f"Lỗi khi tải model: {e}")
        exit()

    # --- 3. Tải và chuẩn bị dữ liệu test ---
    X_test, y_test = load_and_preprocess_test_data(TEST_ANNOTATION_PATH, TEST_IMG_DIR, IMG_WIDTH, IMG_HEIGHT)

    if X_test is None or y_test is None:
        print("Không thể tải hoặc xử lý dữ liệu test. Dừng đánh giá.")
        exit()

    # Không hiển thị kích thước dữ liệu test
    print("\nDữ liệu test đã được tải thành công.")

    # --- 4. Đánh giá model trên tập test ---
    print("\n--- Đánh giá hiệu suất model ---")
    try:
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"\nKết quả đánh giá tổng thể:")
        print(f"  - Accuracy: {accuracy:.2%}")
    except Exception as e:
        print(f"Lỗi trong quá trình đánh giá: {e}")

    # --- 5. Đánh giá chi tiết (Classification Report và Confusion Matrix) ---
    print("\n--- Đánh giá chi tiết ---")
    try:
        # Dự đoán trên tập test
        print("Đang thực hiện dự đoán...")
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1) # Lấy lớp có xác suất cao nhất

        # Tính các chỉ số hiệu suất cho từng lớp
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, labels=[0, 1, 2], average=None)

        # Hiển thị kết quả theo tỷ lệ phần trăm
        print("\nHiệu suất model theo từng nhãn:")
        print(f"{'Nhãn':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-" * 46)
        for i, class_name in enumerate(CLASS_NAMES):
            print(f"{class_name:<10} {precision[i]:.2%} {' ' * 4} {recall[i]:.2%} {' ' * 4} {f1[i]:.2%}")

        # In ra accuracy trung bình
        print(f"\nAccuracy trung bình: {accuracy:.2%}")

        # Tính và vẽ Confusion Matrix (dạng chuẩn hóa theo hàng)
        cm = confusion_matrix(y_test, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Vẽ confusion matrix đẹp hơn bằng seaborn
        try:
            plt.figure(figsize=(10, 8))
            
            # Vẽ ma trận đã chuẩn hóa với định dạng phần trăm
            sns.heatmap(cm_normalized, annot=True, fmt='.1%', cmap='Blues', 
                       xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, linewidths=1)
            plt.xlabel('Predicted Label', fontsize=14)
            plt.ylabel('True Label', fontsize=14)
            plt.title('Confusion Matrix (Normalized)', fontsize=16)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            
            plt.savefig(CONFUSION_MATRIX_SAVE_PATH, dpi=300, bbox_inches='tight')
            print(f"\nĐã lưu Confusion Matrix (dạng tỷ lệ phần trăm) vào: {CONFUSION_MATRIX_SAVE_PATH}")
            
            # Hiển thị bảng
            plt.show()
        except ImportError:
            print("Không tìm thấy thư viện seaborn hoặc matplotlib. Bỏ qua vẽ Confusion Matrix.")
        except Exception as e:
            print(f"Lỗi khi vẽ Confusion Matrix: {e}")

    except Exception as e:
        print(f"Lỗi trong quá trình dự đoán hoặc tính toán metrics: {e}")

    print("\n--- Đánh giá hoàn tất ---") 