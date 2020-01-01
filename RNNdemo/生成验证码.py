import random
from PIL import Image, ImageDraw, ImageFilter, ImageFont


# 生成随机数字、大小写字母
def randChar():
    li1 = [chr(i) for i in range(48, 58)]
    li2 = [chr(i) for i in range(65, 91)]
    li3 = [chr(i) for i in range(97, 123)]
    li = li1 + li2 + li3
    strs = li[random.randint(0, len(li) - 1)]
    return strs


# 随机背景色
def randBgcolor():
    return (random.randint(0, 180),
            random.randint(0, 180),
            random.randint(0, 180))


# 随机文字颜色
def randTextColor():
    return (random.randint(125, 255),
            random.randint(125, 255),
            random.randint(125, 255))


w = 30 * 4
h = 60

font = ImageFont.truetype('arial.ttf', size=36)

for i in range(10000):
    # 生成一张新的白色图片
    image = Image.new('RGB', (w, h), (255, 255, 255))

    draw = ImageDraw.Draw(image)
    # 对白色图片进行随机色填充
    for x in range(w):
        for y in range(h):
            draw.point((x, y), fill=randBgcolor())
    filename = []
    # 将文字填充至图片
    for t in range(4):
        ch = randChar()
        filename.append(ch)
        draw.text((30 * t, 10), ch, font=font, fill=randTextColor())
    # image = image.filter(ImageFilter.BLUR)
    image_path = r'data'
    image.save('{0}/{1}.jpg'.format(image_path, ''.join(filename)))
    print(i)