from unet import UNet
from vgg16 import VGG16

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

import numpy as np
import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Net(nn.Module):
    def __init__(self):
        super().__init__super(Net, self).__init__()
        self.vgg = VGG16()
        self.unet = UNet()

    def forward(self, x, y):
        """
        x is the watermarked image, y is the ground truth 
        """
        y_hat = self.unet(x)
        f_y = self.vgg(y)
        f_y_hat = self.vgg(y_hat)
        return y_hat, f_y, f_y_hat


def calc_loss(y_hat, y, f_y_hat, f_y):
    l1_loss = nn.L1Loss()(y_hat, y)
    l2_loss = nn.MSELoss()(f_y_hat, f_y)
    loss = l1_loss + (1.0/(128 * 112 * 112)) * l2_loss * l2_loss
    return loss


def training_loop(n_epochs, optimizer, model, loss_fn, train_loader, val_loader, train_gtruth, val_gtruth):
    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        model.train()
        for imgs in train_loader:

            y_hat, f_y, f_y_hat = model(imgs[0], imgs[1])
            loss = loss_fn(y_hat, imgs, f_y_hat, f_y)

            optimizer.zero_grad()  
            loss.backward() 
            optimizer.step()

        model.eval()
        for imgs in val_loader:
            pred = model(imgs)
            val_loss = loss_fn(pred, imgs)
        
        if epoch == 1 or epoch % 1 == 0:
            print('Epoch {}, Training loss {}, Validation loss {}'.format(
                epoch,
                loss_train / len(train_loader),
                val_loss / len(val_loader)))


# Make sure to concatenate the input image and the target image before training
class WaterDataSet(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir
    
    def __len__(self):
        return len(self.list_files)
    
    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = np.array(Image.open(img_path).convert("RGB"))
        split_half = int(image.shape[1]/2)
        input_image = image[:, :split_half, :]
        target_image = image[:, split_half:, :]

        return input_image, target_image

