import os
import numpy as np
from PIL import Image


# Путь к папке, содержащей изображения
path = 'in/'

# Изменение размера изображений до 28x28 (На вход лучше подавать изображение 28 на 28)
new_size = (28, 28)

# Список для хранения массивов "сплющенных" изображений
image_array_list = []

# Обход всех изображений в папке и открытие их с помощью библиотеки Pillow
for file_name in os.listdir(path):
    try:
        with Image.open(path + file_name) as img:
            # Преобразование изображения в RGB-формат и изменение размера
            img = img.convert('RGB').resize(new_size)
            # "Сплющивание" изображения и преобразование его в массив numpy
            img_array = np.array(img).flatten()
            # Добавление "сплющенного" изображения в список
            image_array_list.append(img_array)
    except Exception as E:
        print(file_name, E)

# Конкатенация всех "сплющенных" изображений в один массив numpy
dataset = np.concatenate(image_array_list)

# Изменение размерности массива для соответствия количеству элементов и размеру каждого элемента
dataset = dataset.reshape(-1, 3*new_size[0]*new_size[1])

# Сохранение массива в файл numpy
np.save('dataset.npy', dataset)