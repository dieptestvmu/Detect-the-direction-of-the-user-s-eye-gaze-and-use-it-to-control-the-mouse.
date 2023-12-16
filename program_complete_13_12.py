import cv2
import tensorflow as tf
import numpy as np
import mediapipe as mp
import os
import pyautogui
import time

# Khởi tạo biến đếm cho mỗi hướng và tốc độ di chuyển chuột ban đầu
direction_counts = {'up': 0, 'down': 0, 'left': 0, 'right': 0, 'center': 0, 'blink': 0}
mouse_speeds = {'up': 10, 'down': 10, 'left': 10, 'right': 10, 'center': 10, 'blink': 10}

# Đặt biến đếm khung hình và thời gian bắt đầu
frame_count = 0
start_time = time.time()

# Khởi tạo biến thời gian bắt đầu
start_time = time.time()

# Load mô hình đã được huấn luyện
model_path = 'C:/tgmt_ytb1/model_13_12.h5'

model = tf.keras.models.load_model(model_path)

# Định nghĩa các nhãn của hướng nhìn
labels = ['left', 'right', 'up', 'down', 'center', 'blink']

# Khởi tạo bộ phân loại landmask
face_mesh = mp.solutions.face_mesh.FaceMesh()

# Mở webcam
cap = cv2.VideoCapture(0)

# Khởi tạo biến lưu trữ số lần blink
blink_count = 0

while True:
    preprocessed_data = []

    # Đọc khung hình từ webcam
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # Chuyển đổi khung hình sang không gian màu RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Dò tìm landmask trên khung hình
    results = face_mesh.process(frame_rgb)

    # Trích xuất danh sách các điểm ảnh landmask
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            x1 = int(face_landmarks.landmark[27].x * frame.shape[1])
            y1 = int(face_landmarks.landmark[27].y * frame.shape[0])

            x2 = int(face_landmarks.landmark[23].x * frame.shape[1])
            y2 = int(face_landmarks.landmark[23].y * frame.shape[0])

            x3 = int(face_landmarks.landmark[243].x * frame.shape[1])
            y3 = int(face_landmarks.landmark[243].y * frame.shape[0])

            x4 = int(face_landmarks.landmark[130].x * frame.shape[1])
            y4 = int(face_landmarks.landmark[130].y * frame.shape[0])

            x_min = min(x1, x2, x3, x4)
            x_max = max(x1, x2, x3, x4)
            y_min = min(y1, y2, y3, y4)
            y_max = max(y1, y2, y3, y4)

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cropped_image = frame[y_min:y_max, x_min:x_max]
            # Kiểm tra xem ảnh có kích thước hợp lệ không
            if cropped_image is not None and cropped_image.size > 0:
                resized_image = cv2.resize(cropped_image, (64, 64))

                gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
                image = gray_image.astype('float32') / 255.0
                input_image = np.expand_dims(image, axis=-1)
                input_image = np.reshape(input_image, (1, 64, 64, 1))

                # Dự đoán hướng nhìn từ ảnh cắt
                predictions = model.predict(input_image)
                predicted_label = labels[np.argmax(predictions)]

                # Hiển thị hướng nhìn dự đoán lên khung hình
                cv2.putText(frame, predicted_label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                ######################################################################
                # Cập nhật biến đếm và tốc độ di chuyển dựa trên hướng nhìn
                for direction in direction_counts:
                    if predicted_label == direction:
                        direction_counts[direction] += 1

                        # Nếu biến đếm đạt đến ngưỡng nhất định, tăng tốc độ di chuyển chuột
                        if direction_counts[direction] >= 2:
                            mouse_speeds[direction] += 10
                            direction_counts[direction] = 0
                    else:
                        # Nếu hướng nhìn không liên tục là 'right', reset biến đếm và tốc độ di chuyển chuột
                        direction_counts[direction] = 0
                        mouse_speeds[direction] = 10
                ##################################################################################
                # Điều khiển chuột dựa trên hướng nhìn
                if predicted_label == 'up':
                    pyautogui.moveRel(0, -mouse_speeds['up'], duration=0.0001)
                elif predicted_label == 'down':
                    pyautogui.moveRel(0, mouse_speeds['down'], duration=0.0001)
                elif predicted_label == 'left':
                    pyautogui.moveRel(-mouse_speeds['left'], 0, duration=0.0001)
                elif predicted_label == 'right':
                    pyautogui.moveRel(mouse_speeds['right'], 0, duration=0.0001)
                elif predicted_label == 'center':
                    pyautogui.moveRel(0, 0, duration=0.0001)
                elif predicted_label == 'blink':
                    blink_count += 1
                    if blink_count >= 3:
                        pyautogui.click()
                        blink_count = 0
#########################################
        # Tính toán FPS
        frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - start_time
        fps = frame_count / elapsed_time
        #########################################
        # Hiển thị FPS lên khung hình
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Hiển thị khung hình
    cv2.imshow('Frame', frame)

    # Thoát khỏi vòng lặp nếu nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()