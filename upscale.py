from PIL import Image
import os

# Создаем папку, если она не существует
if not os.path.exists('out_upscale'):
    os.mkdir('out_upscale')

# Открываем каждый файл в папке out/ и изменяем его размер до 896x896 без сглаживания
for filename in os.listdir('out/'):
    if filename.endswith('.png'):
        with Image.open(os.path.join('out/', filename)) as im:
            im = im.resize((im.size[0] * 32, im.size[1] * 32), resample=Image.NEAREST)
            im.save(os.path.join('out_upscale/', filename))


with Image.open('gen_img.png') as im:
    im = im.resize((im.size[0] * 32, im.size[1] * 32), resample=Image.NEAREST)
    im.save('gen_img_upscale.png')