import tensorflow as tf
from tensorflow.keras import layers, models

def build_traffic_light_cnn(input_shape, num_classes):
    """
    Xây dựng mô hình CNN đơn giản cho việc phân loại đèn giao thông.

    Args:
        input_shape (tuple): Kích thước ảnh đầu vào (height, width, channels).
                             Ví dụ: (64, 64, 3).
        num_classes (int): Số lượng lớp đầu ra (ví dụ: 3 cho đỏ, vàng, xanh).

    Returns:
        tensorflow.keras.models.Model: Mô hình CNN đã được biên dịch.
    """
    model = models.Sequential(name="TrafficLightCNN")

    # Lớp tích chập 1
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Lớp tích chập 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Lớp tích chập 3 (Thêm để tăng độ sâu nếu cần)
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten lớp đặc trưng
    model.add(layers.Flatten())

    # Lớp Fully Connected (Dense) 1
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5)) # Giảm overfitting

    # Lớp đầu ra
    model.add(layers.Dense(num_classes, activation='softmax')) # Softmax cho phân loại đa lớp

    # Biên dịch mô hình với learning rate thấp hơn
    print("Biên dịch model với Adam optimizer và learning_rate=0.0001")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy', # Vì nhãn là số nguyên (0, 1, 2)
                  metrics=['accuracy'])

    model.summary() # In cấu trúc mô hình
    return model

if __name__ == '__main__':
    # Ví dụ cách sử dụng hàm build
    img_height, img_width = 64, 64
    num_traffic_light_classes = 3 # Đỏ, Vàng, Xanh
    input_shape = (img_height, img_width, 3)

    cnn_model = build_traffic_light_cnn(input_shape, num_traffic_light_classes)
    print("Đã xây dựng mô hình CNN.")
    # tf.keras.utils.plot_model(cnn_model, to_file='cnn_model_plot.png', show_shapes=True)