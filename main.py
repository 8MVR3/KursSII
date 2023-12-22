import os
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import torch
import tkinter as tk
from tkinter import filedialog

IMG_PATH = "data/img"  # Путь к папке с изображениями
OUT_PATH = "data/out"  # Путь к папке для сохранения результатов
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # Использовать GPU, если доступно

# Функция для оценки кадра с помощью модели YOLO
def score_frame(model, frame):
    model.to(DEVICE)
    frame = [frame]
    results = model(frame)
    labels = []
    cord = np.empty((0,5))
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for i, box in enumerate(boxes):
            r = box.xyxy[0].astype(int)
            r_full = np.append(r, box.conf[0])
            cord = np.vstack((cord, r_full))
            labels.append(result.names[int(box.cls[0])])
    return labels, cord

# Функция для отображения обнаруженных объектов на кадре
def plot_boxes(results, frame):
    labels, cord = results
    n = len(labels)
    for i in range(n):
        row = cord[i]
        if row[4] >= 0.5:
            x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            bgr = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
            x_center = int(x1 + (x2 - x1)/2)
            y_center = int(y1 + (y2 - y1)/2)
            y_сustom = int(y1 + (y_center - y1)/2)
            bgr_2 = (0, 0, 0)
            cv2.rectangle(frame, (x_center, y_сustom), (x_center, y_сustom), bgr_2, 6)
            label_name = labels[i]
            cv2.putText(frame, f"{label_name} / {row[4]}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
    return frame

def select_img_folder():
    global img_folder_path
    img_folder_path = filedialog.askdirectory()
    lbl_img_folder.config(text=img_folder_path)

def select_out_folder():
    global out_folder_path
    out_folder_path = filedialog.askdirectory()
    lbl_out_folder.config(text=out_folder_path)

def start_processing():
    if img_folder_path and out_folder_path:
        IMG_PATH = img_folder_path
        OUT_PATH = out_folder_path
        model = YOLO('yolov8n.pt')  # Загрузка предобученной модели YOLO
        # print(model.names)
        for f in os.listdir(IMG_PATH):  # Итерация по изображениям в папке
            img = Image.open(f'{IMG_PATH}/{f}')
            frame = np.array(img)
            results = score_frame(model, frame)  # Оценка кадра с помощью модели YOLO
            frame = plot_boxes(results, frame)   # Отображение результатов на кадре
            im = Image.fromarray(frame)
            if not os.path.exists(OUT_PATH):  # Создание папки для сохранения результатов, если её нет
                os.makedirs(OUT_PATH)
            im.save(f'{OUT_PATH}/{f}')  # Сохранение кадра с обозначенными объектами
    else:
        print("Выберите папки с изображениями и для сохранения результатов")

# Создание основного окна
root = tk.Tk()
root.title("Интерфейс для обработки изображений")

# Кнопка для выбора папки с изображениями
btn_select_img = tk.Button(root, text="Выбрать папку с изображениями", command=select_img_folder)
btn_select_img.pack()

# Метка для отображения выбранной папки с изображениями
lbl_img_folder = tk.Label(root, text="")
lbl_img_folder.pack()

# Кнопка для выбора папки для сохранения результатов
btn_select_out = tk.Button(root, text="Выбрать папку для сохранения результатов", command=select_out_folder)
btn_select_out.pack()

# Метка для отображения выбранной папки для сохранения результатов
lbl_out_folder = tk.Label(root, text="")
lbl_out_folder.pack()

# Кнопка для запуска обработки изображений
btn_start = tk.Button(root, text="Начать обработку", command=start_processing)
btn_start.pack()

# Запуск основного цикла обработки событий
root.mainloop()
