import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
from statistics import mean

# Определим константы
np.random.seed(None)
latent_dim = 2 # Размерность скрытого пространства
epochs = 1000  # Количество эпох для обучения

# Загрузим набор данных и предобработаем его
dataset = np.load('dataset.npy')
dataset = dataset.astype('float32') / 255.
dataset = dataset.reshape(len(dataset), -1)

# Определим входной тензор
input_img = Input(shape=(2352,))

# Определим слои энкодера
h1 = Dense(512, activation='relu')(input_img)
h2 = Dense(256, activation='relu')(h1)

# Определим слои мат. ожидания и дисперсии для выборки из скрытого пространства
z_mean = Dense(latent_dim)(h2)
z_log_var = Dense(latent_dim)(h2)

# Определим функцию выборки из скрытого пространства
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=1.)
    return z_mean + K.exp(z_log_var / 2) * epsilon

# Определим слой выборки из скрытого пространства
z = Lambda(sampling)([z_mean, z_log_var])

# Определим слои декодера
decoder_h1 = Dense(256, activation='relu')
decoder_h2 = Dense(512, activation='relu')
decoder_output = Dense(2352, activation='sigmoid')

# Создадим модель декодера
decoder_input = Input(shape=(latent_dim,))
h_decoded = decoder_h1(decoder_input)
h_decoded = decoder_h2(h_decoded)
output_img = decoder_output(h_decoded)
decoder = Model(decoder_input, output_img)

# Создадим модель автокодировщика
autoencoder_output = decoder(z)
autoencoder = Model(input_img, autoencoder_output)

# Определим функцию потерь для VAE
reconstruction_loss = K.sum(K.binary_crossentropy(input_img, autoencoder_output), axis=-1)
kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(reconstruction_loss + kl_loss)
autoencoder.add_loss(vae_loss)
autoencoder.compile(optimizer='adam')

# Обучим модель VAE на наборе данных
history = autoencoder.fit(dataset, epochs=epochs, batch_size=128)

# Нарисуем кривую обучения и сохраняем её
plt.plot(history.history['loss'])
plt.title('Training Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.savefig('history.png')

# Сгенерируем новые изображения из скрытого пространства с помощью декодера
n = 5 # Количество изображений в одной строке
digit_size = 28 # Размер изображения
figure = np.zeros((digit_size * n, digit_size * n, 3))
grid_x = np.linspace(-4, 4, n)
grid_y = np.linspace(-4, 4, n)[::-1]
count = 0

if not os.path.exists('out'):
    os.mkdir('out')

for i, yi in enumerate(grid_y):
    for j, xi in enumerate(grid_x):
        z_sample = np.array([[xi, yi] + [0] * (latent_dim - 2)])
        x_decoded = decoder.predict(z_sample)
        # Генерируем изображение из декодированного изображения
        digit = x_decoded[0].reshape(digit_size, digit_size, 3)

        # Конвертируем numpy-массив в изображение PIL
        digit_image = Image.fromarray(np.uint8(digit*255))

        # Создаем составное изображение, которое состоит из всех сгенерированных изображений
        figure[i*digit_size:(i+1)*digit_size, j*digit_size:(j+1)*digit_size] = np.array(digit_image)

        # Сохраняем каждое изображение
        digit_image.save(f'out/gen_img_{count}.png')
        count += 1

# Сохраняем составное и последнее изобрежение для превью
figure_image = Image.fromarray(np.uint8(figure*255))
figure_image.save('gen_img_composite.png')
digit_image.save('gen_img.png')
plt.show()

def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)


# Выводим и сохраняем статистику обучения
print("Max loss:", max(history.history['loss']))
print("Min loss:", min(history.history['loss']))
print("Avg loss:", mean(history.history['loss']))

max_loss = max(history.history['loss'])
min_loss = min(history.history['loss'])
avg_loss = mean(history.history['loss'])

data = {
    "Max_loss": max_loss,
    "Min_loss": min_loss,
    "Avg_loss": avg_loss
}


with open('stat.json', 'w+') as file:
    json.dump(data, file)

#autoencoder.save('autoencoder_model.h5')
