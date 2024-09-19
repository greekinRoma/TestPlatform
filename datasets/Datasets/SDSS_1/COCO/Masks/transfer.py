from PIL import Image
import os


def remove_red_channel(input_path, output_path):
    # 打开图片
    img = Image.open(input_path)
    # 将图片转换为RGB模式
    img = img.convert("RGB")

    # 加载图片数据
    pixels = img.load()

    # 遍历图片的每一个像素
    for i in range(img.width):
        for j in range(img.height):
            r, g, b = pixels[i, j]
            # 设置红色通道的值为0
            pixels[i, j] = (0, g, b)

    # 保存修改后的图片到新的路径
    img.save(output_path)


# 设置原始文件夹路径和目标文件夹路径
input_folder = './'
output_folder = './save'

# 如果目标文件夹不存在，创建它
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历文件夹中的所有文件
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # 检查文件扩展名
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        remove_red_channel(input_path, output_path)
        print(f"Processed and saved {filename} to {output_folder}")
