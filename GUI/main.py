import cv2
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np


class VideoCaptureApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Video Stream with YOLO")

        # Инициализируем видеопоток
        self.cap = cv2.VideoCapture(0)

        # Загрузка YOLO
        self.net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

        with open("coco.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

        # Создаем метку для отображения видео
        self.label = tk.Label(master)
        self.label.pack()

        # Запускаем обновление кадров
        self.update()

        # Кнопка выхода
        self.exit_button = tk.Button(master, text="Exit", command=self.exit)
        self.exit_button.pack()

    def update(self):
        # Считываем кадр из камеры
        ret, frame = self.cap.read()
        if ret:
            height, width, channels = frame.shape

            # Подготовка изображения для YOLO
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            self.net.setInput(blob)
            outs = self.net.forward(self.output_layers)

            # Обработка выходных данных
            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:  # Порог уверенности
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        # Прямоугольник координаты
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            # Наносим прямоугольники на изображение
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(self.classes[class_ids[i]])
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Преобразуем BGR в RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Преобразуем кадр в Image и затем в PhotoImage
            img = Image.fromarray(frame)

            imgtk = ImageTk.PhotoImage(image=img)

            # Обновляем метку с новым изображением
            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)

        # Запускаем метод снова через 10 мс
        self.master.after(10, self.update)

    def exit(self):
        # Освобождаем ресурсы и закрываем приложение
        self.cap.release()
        self.master.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoCaptureApp(root)
    root.mainloop()

