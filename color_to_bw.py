from PIL import Image
for i in range(1, 12):
    img_path = "./color_images/img" + str(i) + ".jpeg"
    img = Image.open(img_path).convert('LA')
    img.save("./bw_images/bw_img" + str(i) + ".png")