# Установка библиотеки Ultralytics для работы с YOLO
!pip install -q ultralytics

# Импорт библиотек
from ultralytics import YOLO          # Загрузка и запуск модели YOLO
from google.colab import files        # Загрузка пользовательского изображения
from PIL import Image                 # Открытие изображения
import io                             # Преобразование байтов в изображение
import cv2                            # Отрисовка боксов на изображении
import matplotlib.patches as patches  # Отображение названия дефекта
import matplotlib.pyplot as plt       # Отображение изображение в ноутбуке
import numpy as np                    # Получение изображения в виде массива

# Загрузка дообученной модели YOLO11.
def load_model(model_path="BestModel.pt"):
    model = YOLO(model_path)
    return model

# Загрузка пользовательского изображения
def upload_image():
    uploaded = files.upload()
    for filename in uploaded:
        image = Image.open(io.BytesIO(uploaded[filename])).convert("RGB")
        return image, filename

# Предсказание модели
def predict(model, image):
    result = model.predict(image, save=False, conf=0.5)
    return result[0]

# Постобработка и визуализация
def postprocess_and_visualize(result, class_names=None):
    # Преобразование изображения в np.ndarray с правильным цветом для OpenCV
    img = np.array(result.orig_img)
    if img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # Получение результатов (xyxy - координаты, cls - индекс класса, conf - точность)
    boxes = result.boxes
    names = result.names
    # Создание рисунка для отображения результата
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.axis('off')
    # Перебор всех предсказанных дефектов
    for box in boxes:
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = f"{names[cls]} ({conf:.2f})"
        # Отрисовка рамки
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
        # Добавление текста
        ax.text(x1, y1 - 5, label,
                fontsize=10, color='black',
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))
    plt.show()

# Главная функция
def main(model_path="DiplomaBestModel.pt"):
    model = load_model(model_path)
    image, filename = upload_image()
    result = predict(model, image)
    postprocess_and_visualize(result)

main()
