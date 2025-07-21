import os
import numpy as np
import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator # Tạm thời không dùng
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# Import các hàm tiện ích và mô hình
# Đảm bảo các hàm này trong utils.py là phiên bản mới nhất hỗ trợ trả về 3 giá trị
from utils import load_coco_annotations, extract_traffic_light_patches, preprocess_image_for_cnn
from cnn_model import build_traffic_light_cnn

# --- Cấu hình --- (Điều chỉnh nếu cần)
IMG_WIDTH = 64
IMG_HEIGHT = 64
IMG_CHANNELS = 3
NUM_CLASSES = 3 # (0: RED, 1: YELLOW, 2: GREEN)
CLASS_NAMES = ["RED", "YELLOW", "GREEN"]

BATCH_SIZE = 32
EPOCHS = 75 # Giữ nguyên số epoch, EarlyStopping sẽ dừng sớm
PATIENCE = 10

# Sử dụng đường dẫn tuyệt đối
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

BASE_DATASET_DIR = os.path.join(PROJECT_ROOT, 'dataset')

TRAIN_ANNOTATION_PATH = os.path.join(BASE_DATASET_DIR, 'train', '_annotations.coco.json')
TRAIN_IMG_DIR = os.path.join(BASE_DATASET_DIR, 'train', 'images')

VALID_ANNOTATION_PATH = os.path.join(BASE_DATASET_DIR, 'valid', '_annotations.coco.json')
VALID_IMG_DIR = os.path.join(BASE_DATASET_DIR, 'valid', 'images')

TEST_ANNOTATION_PATH = os.path.join(BASE_DATASET_DIR, 'test', '_annotations.coco.json')
TEST_IMG_DIR = os.path.join(BASE_DATASET_DIR, 'test', 'images')

MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, 'trained_models')
# Đổi tên model để phản ánh không có augmentation
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'traffic_light_cnn_noaug_best.keras')
TRAINING_HISTORY_PLOT_PATH = os.path.join(PROJECT_ROOT, 'training_history_noaug.png')
# --- Kết thúc Cấu hình ---

def load_and_preprocess_data(annotation_path, img_dir, img_width, img_height, split_name=""):
    """Tải, trích xuất và tiền xử lý dữ liệu đèn giao thông từ COCO annotations."""
    print(f"\n--- Đang xử lý tập {split_name} ---")
    print(f"Annotation file: {annotation_path}")
    print(f"Image directory: {img_dir}")

    if not os.path.exists(annotation_path) or not os.path.isdir(img_dir):
        print(f"Lỗi: Không tìm thấy file annotation hoặc thư mục ảnh của tập {split_name}.")
        return None, None

    image_paths, bboxes, labels = load_coco_annotations(annotation_path, img_dir)
    if image_paths is None:
        print(f"Lỗi khi tải dữ liệu {split_name}.")
        return None, None

    print(f"Đang trích xuất và tiền xử lý {len(image_paths)} ảnh {split_name}...")
    light_patches, patch_labels = extract_traffic_light_patches(image_paths, bboxes, labels)

    if light_patches.size == 0:
        print(f"Lỗi: Không trích xuất được patch ảnh nào từ tập {split_name}.")
        return None, None

    print(f"Đang chuẩn hóa {len(light_patches)} ảnh {split_name}...")
    processed_images_list = [preprocess_image_for_cnn(patch, target_size=(img_width, img_height)) for patch in light_patches]
    valid_indices = [i for i, img in enumerate(processed_images_list) if img is not None]

    if not valid_indices:
        print(f"Lỗi: Không có ảnh {split_name} nào được tiền xử lý thành công.")
        return None, None

    processed_images = np.array([processed_images_list[i] for i in valid_indices])
    processed_labels = np.array([patch_labels[i] for i in valid_indices])

    print(f"Hoàn tất xử lý tập {split_name}: {len(processed_images)} ảnh, {len(processed_labels)} nhãn.")
    return processed_images, processed_labels

# === Hàm chính thực thi ===
if __name__ == "__main__":
    print("--- Bắt đầu quá trình huấn luyện CNN đèn giao thông (KHÔNG Augmentation, tập Valid riêng) ---") # Cập nhật tiêu đề

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # --- 1. Tải và tiền xử lý dữ liệu --- (Vẫn giữ nguyên)
    X_train, y_train = load_and_preprocess_data(TRAIN_ANNOTATION_PATH, TRAIN_IMG_DIR, IMG_WIDTH, IMG_HEIGHT, "train")
    if X_train is None:
        print("Lỗi nghiêm trọng: Không thể tải dữ liệu huấn luyện. Dừng chương trình.")
        exit()

    X_val, y_val = load_and_preprocess_data(VALID_ANNOTATION_PATH, VALID_IMG_DIR, IMG_WIDTH, IMG_HEIGHT, "validation")
    if X_val is None:
        print("Lỗi nghiêm trọng: Không thể tải dữ liệu validation. Dừng chương trình.")
        exit()

    X_test, y_test = load_and_preprocess_data(TEST_ANNOTATION_PATH, TEST_IMG_DIR, IMG_WIDTH, IMG_HEIGHT, "test")

    print(f"\nKích thước dữ liệu:")
    print(f"Train:      {X_train.shape if X_train is not None else 'N/A'}, Nhãn: {y_train.shape if y_train is not None else 'N/A'}")
    print(f"Validation: {X_val.shape if X_val is not None else 'N/A'}, Nhãn: {y_val.shape if y_val is not None else 'N/A'}")
    print(f"Test:       {X_test.shape if X_test is not None else 'N/A'}, Nhãn: {y_test.shape if y_test is not None else 'N/A'}")

    # Kiểm tra nhãn tập Train
    unique_train_labels = np.unique(y_train)
    print(f"\nCác nhãn duy nhất trong tập train: {unique_train_labels}")
    if len(unique_train_labels) != NUM_CLASSES:
        print(f"Cảnh báo: Số lớp tìm thấy ({len(unique_train_labels)}) khác với NUM_CLASSES ({NUM_CLASSES}).")

    # >>> THÊM KIỂM TRA NHÃN TẬP VALIDATION <<<
    if X_val is not None and y_val is not None:
        unique_val_labels, val_counts = np.unique(y_val, return_counts=True)
        print(f"\nCác nhãn duy nhất và số lượng trong tập validation:")
        if len(unique_val_labels) > 0:
            for label_id, count in zip(unique_val_labels, val_counts):
                # Đảm bảo label_id hợp lệ trước khi truy cập CLASS_NAMES
                class_name = CLASS_NAMES[label_id] if 0 <= label_id < len(CLASS_NAMES) else f"Unknown ID: {label_id}"
                print(f"  - Lớp {label_id} ({class_name}): {count}")
            if len(unique_val_labels) < NUM_CLASSES:
                 print(f"Cảnh báo: Tập validation chỉ chứa {len(unique_val_labels)}/{NUM_CLASSES} lớp.")
        else:
            print("  (Không tìm thấy nhãn nào trong tập validation)")
    # >>> KẾT THÚC KIỂM TRA <<<

    # --- 3. Xây dựng mô hình ---
    print("\n--- Xây dựng mô hình CNN ---")
    # Cân nhắc biên dịch lại với learning rate cụ thể nếu cần thử nghiệm sau
    # optimizer=tf.keras.optimizers.Adam(learning_rate=0.001) # Giữ mặc định trước
    model = build_traffic_light_cnn(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), num_classes=NUM_CLASSES)

    # --- 4. Định nghĩa Callbacks ---
    early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=1, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True, verbose=1)

    # --- 5. Huấn luyện mô hình KHÔNG sử dụng Generator ---
    print("\n--- Bắt đầu huấn luyện (Không có Data Augmentation) ---")

    history = model.fit(
        X_train, y_train,         # Sử dụng dữ liệu train trực tiếp
        batch_size=BATCH_SIZE,     # Thêm batch_size ở đây
        epochs=EPOCHS,
        # steps_per_epoch=steps_per_epoch, # Không cần khi không dùng generator
        validation_data=(X_val, y_val), # Dữ liệu validation giữ nguyên
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )

    # --- 6. Đánh giá trên tập Validation cuối cùng (model tốt nhất) ---
    print("\n--- Đánh giá model tốt nhất trên tập Validation ---")
    loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Loss (best model): {loss:.4f}")
    print(f"Validation Accuracy (best model): {accuracy:.4f}")

    # --- 7. Đánh giá cuối cùng trên tập Test ---
    if X_test is not None and y_test is not None:
        print("\n--- Đánh giá model tốt nhất trên tập Test ---")
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
    else:
        print("\nBỏ qua đánh giá trên tập Test do không tải được dữ liệu.")

    print(f"\n--- Huấn luyện hoàn tất ---")
    print(f"Model tốt nhất (không augmentation) đã được lưu tại: {MODEL_SAVE_PATH}")

    # --- 8. Vẽ đồ thị learning curve ---
    try:
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs_range = range(len(acc))

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy (No Augmentation)') # Đổi tiêu đề

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss (No Augmentation)') # Đổi tiêu đề

        plt.savefig(TRAINING_HISTORY_PLOT_PATH) # Lưu file plot mới
        print(f"Đã lưu đồ thị lịch sử huấn luyện vào {TRAINING_HISTORY_PLOT_PATH}")
        # plt.show()
    except KeyError as e:
        print(f"Lỗi KeyError khi truy cập history: {e}. Có thể tên metric bị sai.")
        print(f"Các key có sẵn: {history.history.keys()}")
    except ImportError:
        print("Không tìm thấy thư viện matplotlib. Bỏ qua bước vẽ đồ thị.")
    except Exception as e:
         print(f"Lỗi khi vẽ đồ thị: {e}")