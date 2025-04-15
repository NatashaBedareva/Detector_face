from ultralytics import YOLO
import time

ov_model = YOLO("OpenVINO/")
t = time.time()
metrics = ov_model.val(data="/content/drive/MyDrive/dataset/dataset/data.yaml", split="test")
print(f"mAP@0.5: {metrics.box.map}")  # mAP при IoU=0.5
print(time.time() - t)
# Тест на одном изображении
results = ov_model.predict("test.jpg", save=True, conf=0.5)
