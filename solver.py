import os
import numpy as np
import time
import datetime
import torch
import torch.nn as nn
import torchvision
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from evaluation import *
from network import *
import csv
from dataset import *
from data_loader import get_loader
from torchvision import transforms as T


class Solver(object):
    def __init__(self, config, train_loader, valid_loader, test_loader):

        self.config = config
        # Data loader
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        # Models
        self.unet = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.img_ch = config.img_ch
        self.output_ch = config.output_ch
        self.criterion_type = config.criterion_type
        self.optimizer_type = config.optimizer_type

        # self.criterion = DiceLoss()

        # Hyper-parameters
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.k = 1

        # Training settings
        self.num_epochs = config.num_epochs
        self.num_epochs_decay = config.num_epochs_decay
        self.batch_size = config.batch_size

        # Path
        self.model_path = config.model_path
        self.result_path = config.result_path
        self.mode = config.mode
        self.bestmodel_path = config.bestmodel_path
        self.save = config.save

        self.device_ids = config.cuda_idx  # multi-GPU
        torch.cuda.set_device(self.device_ids[0])
        self.model_type = config.model_type
        self.t = config.t
        self.build_model()

    def build_model(self):
        """Build generator and discriminator."""
        if self.model_type == 'U_Net':
            self.unet = U_Net(img_ch=1, output_ch=1)
        elif self.model_type == 'AttU_Net':
            self.unet = AttU_Net(img_ch=1, output_ch=1)
        elif self.model_type == 'FFNet':
            self.unet = FFNet(img_ch=1, output_ch=1)
        elif self.model_type == 'ResU-net':
            self.unet = Resnet_Unet()
        elif self.model_type == 'SegNet':
            self.unet = SegNet(1, 1)

        if self.optimizer_type == 'Adam':
            self.optimizer = optim.Adam(list(self.unet.parameters()), self.lr, [self.beta1, self.beta2])
        elif self.optimizer_type == 'RAdam':
            self.optimizer = RAdam(list(self.unet.parameters()), self.lr, [self.beta1, self.beta2])

        if self.criterion_type == 'Dice':
            self.criterion = DiceLoss()
        elif self.criterion_type == 'Tversky':
            self.criterion = TverskyLoss()
        elif self.criterion_type == 'GHMC':
            self.criterion = GHMC()
        elif self.criterion_type == 'mix':
            self.criterion = mix()
        elif self.criterion_type == 'BCE':
            self.criterion = nn.BCELoss()
        elif self.criterion_type == 'Focal':
            self.criterion = FocalLoss()

        cuda_available = torch.cuda.is_available()
        if cuda_available:
            self.unet = self.unet.cuda()
            self.unet = nn.DataParallel(self.unet, device_ids=self.device_ids)

    # self.print_network(self.unet, self.model_type)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def to_data(self, x):
        """Convert variable to tensor."""
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data

    def update_lr(self, g_lr, d_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def reset_grad(self):
        """Zero the gradient buffers."""
        self.unet.zero_grad()

    def compute_accuracy(self, SR, GT):
        SR_flat = SR.view(-1)
        GT_flat = GT.view(-1)

        acc = GT_flat.data.cpu() == (SR_flat.data.cpu() > 0.5)

    def tensor2img(self, x):
        img = (x[:, 0, :, :] > x[:, 1, :, :]).float()
        img = img * 255
        return img

    def train(self):
        """Train encoder, generator and discriminator."""

        # ====================================== Training ===========================================#
        # ===========================================================================================#

        unet_path = "/home/shixi/users_all/zhy/Medical_image/Image_Segmentation/models/R2A2U_Net-epoch29-score0.8193-lr0.0000."

        # U-Net Train
        if os.path.isfile(unet_path):
            # Load the pretrained Encoder
            self.unet.load_state_dict(torch.load(unet_path))
            print('%s is Successfully Loaded from %s' % (self.model_type, unet_path))

        # Train for Encoder
        lr = self.lr
        best_unet_score = 0.
        for epoch in range(self.num_epochs):
            since = time.time()
            self.unet.train(True)

            epoch_loss = 0

            acc = 0.  # Accuracy
            SE = 0.  # Sensitivity (Recall)
            SP = 0.  # Specificity
            FP = 0.  # sr=1,gt=0
            FN = 0.  # sr=0,gt=1
            TP = 0.  # sr=1,gt=1
            DC = 0.  # Dice Coefficient
            length = 0

            for i, (oimage, images, GT, fn) in enumerate(self.train_loader):
                # GT : Ground Truth

                images = images.cuda()
                GT = GT.cuda()

                # SR : Segmentation Result
                SR = self.unet(images)
                SR_probs = torch.sigmoid(SR)
                SR_flat = SR_probs.view(SR_probs.size(0), -1)
                GT_flat = GT.view(GT.size(0), -1)
                loss = self.criterion(SR_flat, GT_flat)
                epoch_loss += loss.item()

                # Backprop + optimize
                self.reset_grad()
                loss.backward()
                self.optimizer.step()

                acc += get_accuracy(SR, GT)
                se, fn = get_sensitivity(SR, GT)
                sp, fp = get_specificity(SR, GT)
                SE += se
                SP += sp
                FP += fp
                FN += fn
                TP += (GT * SR).sum()
                DC += get_DC(SR, GT)
                length += 1

            acc = acc / length
            SE = SE / length
            SP = SP / length
            FP = FP / length
            FN = FN / length
            TP = TP / length
            DC = DC / length
            epoch_loss = epoch_loss / length
            # Print the log info
            print(
                'Epoch [%d/%d] \n[Training Loss]: %.4f \n[Training]  Acc: %.4f, SE: %.4f, SP: %.4f, FP: %.1f, FN: %.1f, TP: %.1f, DC: %.4f' % (
                    epoch + 1, self.num_epochs, \
                    epoch_loss, \
                    acc, SE, SP, FP, FN, TP, DC))

            # Decay learning rate
            if (epoch + 1) >= self.num_epochs_decay and (epoch + 1) % 10 == 0:
                lr = self.lr / (2 * self.k)
                self.k += 1
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                print('Decay learning rate to lr: {}.'.format(lr))

                # self.scheduler.step(epoch_loss)

            # ===================================== Validation ====================================#
            with torch.no_grad():

                self.unet.train(False)
                self.unet.eval()

                acc = 0.  # Accuracy
                SE = 0.  # Sensitivity (Recall)
                SP = 0.  # Specificity
                FP = 0.  # sr=1,gt=0
                FN = 0.  # sr=0,gt=1
                TP = 0.  # sr=1,gt=1
                DC = 0.  # Dice Coefficient
                length = 0
                valid_loss = 0.
                for i, (oimage, images, GT, fn) in enumerate(self.valid_loader):
                    images = images.cuda()
                    GT = GT.cuda()
                    # SR : Segmentation Result
                    SR = torch.sigmoid(self.unet(images))
                    SR_flat = SR.view(SR.size(0), -1)
                    GT_flat = GT.view(GT.size(0), -1)
                    loss = self.criterion(SR_flat, GT_flat)
                    valid_loss += loss.item()
                    acc += get_accuracy(SR, GT)
                    se, fn = get_sensitivity(SR, GT)
                    sp, fp = get_specificity(SR, GT)
                    SE += se
                    SP += sp
                    FP += fp
                    FN += fn
                    TP += (GT * SR).sum()
                    DC += get_DC(SR, GT)

                    length += 1

                acc = acc / length
                SE = SE / length
                SP = SP / length
                FP = FP / length
                FN = FN / length
                TP = TP / length
                DC = DC / length
                valid_loss = valid_loss / length
                unet_score = DC

                time_elapsed = time.time() - since

                print('[Validation loss]: %.4f' % valid_loss, 'Epoch complete in {:.0f}m {:.0f}s'.format(
                    time_elapsed // 60, time_elapsed % 60))
                print('[Validation] Acc: %.4f, SE: %.4f, SP: %.4f, FP: %.1f, FN: %.1f, TP: %.1f, DC: %.4f' % (
                    acc, SE, SP, FP, FN, TP, DC))

                # Save Best U-Net model
                if unet_score > best_unet_score:
                    best_unet_score = unet_score
                    best_epoch = epoch
                    best_unet = self.unet.state_dict()
                    print('Best %s model score : %.4f' % (self.model_type, best_unet_score))
                    if epoch > 10:
                        torch.save(best_unet, os.path.join(self.model_path, '%s-epoch%d-score%.4f-lr%.6f.pkl' % (
                            self.model_type, epoch + 1, best_unet_score, self.optimizer.param_groups[0]['lr'])))

            # ===================================== Test ====================================#

    def test(self):
        del self.unet
        # del best_unet
        self.build_model()
        self.unet.load_state_dict(torch.load(self.bestmodel_path))

        self.unet.train(False)
        self.unet.eval()

        with torch.no_grad():

            acc = 0.  # Accuracy
            SE = 0.  # Sensitivity (Recall)
            SP = 0.  # Specificity
            PC = 0.
            FP = 0.  # sr=1,gt=0
            FN = 0.  # sr=0,gt=1
            TP = 0.  # sr=1,gt=1
            DC = 0.  # Dice Coefficient
            length = 0

            for i, (oimage, images, GT, fn) in enumerate(self.test_loader):

                images = images.cuda()
                GT = GT.cuda()
                SR = torch.sigmoid(self.unet(images))
                # save concat image,sr,gt

                if self.save:
                    output = torch.zeros((320, 320 * 3))  # H, W

                    # torchvision.utils.save_image(GT.data.cpu(), os.path.join(self.result_path, '%sgt.png' % (
                    # fn[0].replace('_', '-'))))
                    # torchvision.utils.save_image(SR.data.cpu(), os.path.join(self.result_path, '%ssr.png' % (
                    # fn[0].replace('_', '-'))))
                    output[:, 0: 320] = images.data.cpu()
                    output[:, 320: 320 * 2] = GT.data.cpu()
                    output[:, 320 * 2: 320 * 3] = SR.data.cpu()
                    perpath = os.path.join(self.result_path, fn[0].split('_')[0])
                    if not os.path.exists(perpath):
                        os.makedirs(perpath)
                    torchvision.utils.save_image(output, os.path.join(perpath, '%scmp.png' % (fn[0].replace('_', '-'))))

                acc += get_accuracy(SR, GT)
                se, fn = get_sensitivity(SR, GT)
                sp, fp = get_specificity(SR, GT)
                SE += se
                SP += sp
                PC += get_precision(SR, GT)
                FP += fp
                FN += fn
                TP += (GT * SR).sum()
                DC += get_DC(SR, GT)
                length += 1

            acc = acc / length
            SE = SE / length
            SP = SP / length
            PC = PC / length
            FP = FP / length
            FN = FN / length
            TP = TP / length
            DC = DC / length

            print('[Test] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, FP: %.1f, FN: %.1f, TP: %.1f, DC: %.4f' % (
                acc, SE, SP, PC, FP, FN, TP, DC))
            if self.save:
                print("result saved!")









