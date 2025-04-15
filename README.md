
# Детектор лица человека 👋

В данном репозитории содержатся файлы, которые направлены на улучшение работы детектора. 

## 🚀Apskaler 
 Apskaler.py содержит функции для увеличения размера изображения при помощи дифузионной сети SwinIR

### Использование 

Скачайте веса модели и тестовую картинку

```commandline
wget -O swin-ir.pth https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth
wget -O butterfly.png https://raw.githubusercontent.com/JingyunLiang/SwinIR/main/testsets/RealSRSet%2B5images/butterfly.png
```
Замените "butterfly60.png" на называние своего изображения
```commandline
img_lq = cv2.imread("butterfly60.png", cv2.IMREAD_COLOR).astype(np.float32) / 255
```
Задайте значение переменной во сколько хотите увеличить изображение 
```commandline
SCALE = 4
```

## 🚀 GUI
Выводит видео с вэб камеры и отрисовка bbox-сов лиц с графическим интерфейсом на tkinter

### Использование 

В директорию с кодом поместите веса и конфигураци модели 
задайте параметры 

```commandline
self.net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
```

## 🚀 Pt2openVINO

Для увелеяения скорости модели перевели веса из .pt в openVINO.

Тестирование модели при помощи testOpenVINO.py


## ⚡️ Authors

- [chelbaev](https://github.com/chelbaev)
- [TsyrenovMergen](https://github.com/TsyrenovMergen)
- [NatashaBedareva](https://github.com/NatashaBedareva)



## 🔗 Links

 - [YOLO](https://github.com/ultralytics/ultralytics)
 - [Awesome README](https://github.com/matiassingers/awesome-readme)
 - [How to write a Good readme](https://bulldogjob.com/news/449-how-to-write-a-good-readme-for-your-github-project)

