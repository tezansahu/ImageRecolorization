import myModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from skimage import io, color, transform
import numpy as np
import matplotlib.pyplot as plt

####################
# Dataset Creation #
####################

HEIGHT = 256
WIDTH = 256
img_l = []  # Luminance [b/w image]
img_a_b = [] # Chrominance

for i in range(1,12):
    img = io.imread("../color_images/color_img" + str(i) + ".jpeg")
    new_img = color.rgb2lab(img)
    l = torch.Tensor(new_img[:, :, 0]/100)
    a = torch.Tensor((new_img[:, :, 1] + 110)/220)
    b = torch.Tensor((new_img[:, :, 2] + 110)/220)
    # img_l_tensor = torch.Tensor(l)
    img_a_b_tensor = torch.cat((a.unsqueeze(0), b.unsqueeze(0)), 0)
    img_l.append(l)
    img_a_b.append(img_a_b_tensor)

img_l = torch.stack(img_l)
img_a_b = torch.stack(img_a_b)
# img_a_b = img_a_b.permute(0, 3, 1, 2)
# print(img_l.size(), img_a_b.size())


##################################
# Model Paraeters Initialization #
##################################

model = myModel.ColorRestorationNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

############
# Training #
############

for epoch in range(501):
    running_loss = 0.0
    for i in range(0, 10):
        optimizer.zero_grad()
        output = model(img_l[i].unsqueeze(0).unsqueeze(0))
        loss = criterion(output, img_a_b[i].unsqueeze(0))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # print(running_loss)
        # if i%2 == 0:
        #     print("[Epoch: %d, Image: %2d] Loss = %.4f" %(epoch, i+1, running_loss/2) )
        #     running_loss = 0.0
    if epoch%20 == 0:
        print("Epoch: %d, Loss: %.8f" %(epoch, running_loss/10))


#########################################
# Reconstruct RGB Image from LAB Format #
#########################################

def reconstruct_img(l, a, b):
    lab_img = np.stack((l.numpy()*100, a.numpy()*220 - 110, b.numpy()*220 - 110), axis=2)
    rgb_img = color.lab2rgb(lab_img)
    return rgb_img

#####################################################
# Looking at recolored images from the training set #
#####################################################

with torch.no_grad():
    for i in range(0, 10):
        output = model(img_l[i].unsqueeze(0).unsqueeze(0))
        colored_img = reconstruct_img(img_l[i], output[0][0], output[0][1])
        original_img = reconstruct_img(img_l[i], img_a_b[i][0], img_a_b[i][1])

        fig = plt.figure(figsize=(15, 45))
        fig.add_subplot(1, 3, 1)
        plt.imshow(img_l[i], cmap='gray')
        plt.title("Input Image (B/W)")
        fig.add_subplot(1, 3, 2)
        plt.imshow(colored_img)
        plt.title("Output Image (Colored)")
        fig.add_subplot(1, 3, 3)
        plt.imshow(original_img)
        plt.title("Original Image (Colored)")
        plt.show()