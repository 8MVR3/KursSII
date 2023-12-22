# Обработка изображений с помощью YOLO

Этот код представляет собой программу для обработки изображений с использованием нейросетевой модели YOLO (You Only Look Once) для обнаружения и распознавания объектов. Код предоставляет возможность выбора папки с изображениями, папки для сохранения результатов и запуска обработки изображений с использованием предобученной модели YOLO.

## Использование

### Установка зависимостей
Для запуска этой программы убедитесь, что у вас установлены следующие зависимости:
- Python 3
- Библиотеки: `ultralytics`, `PIL`, `numpy`, `opencv-python`, `torch`, `tkinter`

### Запуск программы
1. Запустите программу, выполнив файл `main.py`.
2. Выберите папку с изображениями, а затем выберите папку для сохранения результатов.
3. Нажмите кнопку "Начать обработку" для запуска обработки изображений.

## Структура проекта

- `main.py` - основной файл программы для обработки изображений.
- `yolov8n.pt` - файл предобученной модели YOLOv8n.
- `data/` - папка для хранения изображений и результатов.

## Лицензия
Этот код распространяется под лицензией MIT. См. файл LICENSE для получения дополнительной информации.

## Автор
Этот проект был создан Михолап Вячеславом Владиславовичем
