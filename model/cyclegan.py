import torch
import itertools
import torch.nn as nn
from model.G_D import Generator, Discriminator
class CycleGAN(nn.Module):
    def __init__(self, opt):
        # G:A -> B, F: B -> A
        # D_X: discriminate realA vs generated from F(B)
        # D_Y: discriminate realB vs generated from G(A)
        super(CycleGAN, self).__init__()
        self.G = Generator(opt)
        self.F = Generator(opt)
        self.D_X = Discriminator(opt)
        self.D_Y = Discriminator(opt)
        self.adv_loss = nn.MSELoss()
        self.cycle_loss = nn.L1Loss()
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.G.parameters(), self.F.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.D_X.parameters(), self.D_Y.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
    def input(self, input):
        self.real_A = input['A']
        self.real_B = input['B']
    def forward(self):
        self.fake_B = self.G(self.real_A) # G(A)
        self.rec_A = self.F(self.fake_B) # F(G(A))
        self.fake_A = self.F(self.real_B) # F(B)
        self.rec_B = self.G(self.fake_A) # G(F(B))
    def optimize(self):
        self.forward()
        f_label = torch.tensor(0.0).to(self.real_A.device)
        r_label = torch.tensor(1.0).to(self.real_A.device)
        # update G and F
        for param in self.D_X.parameters():
            param.requires_grad = False
        for param in self.D_Y.parameters():
            param.requires_grad = False
        self.optimizer_G.zero_grad()
        self.loss_G = self.adv_loss(self.D_Y(self.fake_B), f_label) + self.adv_loss(self.D_X(self.fake_A), f_label)
        self.loss_G += self.cycle_loss(self.rec_A, self.real_A) + self.cycle_loss(self.rec_B, self.real_B)
        self.loss_G.backward()
        self.optimizer_G.step()
        
        # update D_X and D_Y
        for param in self.D_X.parameters():
            param.requires_grad = True
        for param in self.D_Y.parameters():
            param.requires_grad = True
        self.optimizer_D.zero_grad()
        self.loss_D_X = self.adv_loss(self.D_X(self.fake_A), f_label) + self.adv_loss(self.D_X(self.real_A), r_label)
        self.loss_D_X.backward()
        self.loss_D_Y = self.adv_loss(self.D_Y(self.fake_B), f_label) + self.adv_loss(self.D_Y(self.real_B), r_label)
        self.loss_D_Y.backward()
        self.optimizer_D.step()