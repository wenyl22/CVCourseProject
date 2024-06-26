import torch
import itertools
import torch.nn as nn
from model.G_D import Generator, Discriminator, get_scheduler
import os
from collections import OrderedDict
from utils.image_pool import ImagePool
import torch.nn.functional as F
class CycleGAN(nn.Module):
    def __init__(self, opt):
        super(CycleGAN, self).__init__()
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0]))
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        # G: A -> B, F: B -> A
        # D_X: discriminate realB vs generated from G(A)
        # D_Y: discriminate realA vs generated from F(B)
        self.model_names = ['G', 'F', 'D_X', 'D_Y']
        self.visual_names = ['real_A', 'fake_B', 'rec_A', 'idt_B', 'real_B', 'fake_A', 'rec_B', 'idt_A', 'real_A_canny', 'real_B_canny']
        self.loss_names = ['D_X', 'G', 'cycle_A', 'idt_A',  'D_Y', 'F','cycle_B', 'idt_B', 'gpB', 'gpA']
        self.fake_A_pool = ImagePool(opt.pool_size)
        self.fake_B_pool = ImagePool(opt.pool_size)
        self.G = Generator(opt, opt.input_nc + self.opt.use_canny, opt.output_nc)
        self.F = Generator(opt, opt.output_nc, opt.input_nc)
        self.D_X = Discriminator(opt, opt.output_nc)
        self.D_Y = Discriminator(opt, opt.input_nc)
        self.adv_loss = nn.MSELoss()
        self.cycle_loss = nn.L1Loss()
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.G.parameters(), self.F.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.D_X.parameters(), self.D_Y.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizers = [self.optimizer_G, self.optimizer_D]
        self.schedulers = [get_scheduler(optimizer, opt) for optimizer in self.optimizers]

    def input(self, input):
        self.real_A = input['A'].to(self.device)
        self.real_A_blur = input['A_blur'].to(self.device)
        self.real_A_canny = input['A_canny'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.real_B_blur = input['B_blur'].to(self.device)
        self.real_B_canny = input['B_canny'].to(self.device)
    def forward(self):
        if self.opt.use_canny:
            self.fake_B = self.G(torch.cat((self.real_A, self.real_A_canny), dim = 1)) # G(A)
        else:
            self.fake_B = self.G(self.real_A)
        self.rec_A = self.F(self.fake_B) # F(G(A))
        self.fake_A = self.F(self.real_B) # F(B)
        if self.opt.use_canny:
            self.rec_B = self.G(torch.cat((self.fake_A, self.real_B_canny), 1)) # G(F(B))
            self.idt_A = self.G(torch.cat((self.real_B, self.real_B_canny), 1)) # G(B)
        else:
            self.rec_B = self.G(self.fake_A)
            self.idt_A = self.G(self.real_B)
        self.idt_B = self.F(self.real_A) # F(A)
    def updateG(self):
        tmp = self.D_X(self.fake_A)
        r_label = torch.tensor(1.0).to(self.real_A.device).expand_as(tmp)
        f_label = torch.tensor(0.0).to(self.real_A.device).expand_as(tmp)
        for param in self.D_X.parameters():
            param.requires_grad = False
        for param in self.D_Y.parameters():
            param.requires_grad = False
        self.optimizer_G.zero_grad()

        self.loss_idt_A = self.cycle_loss(self.idt_A, self.real_B) * self.opt.lambda_B * self.opt.lambda_identity
        self.loss_idt_B = self.cycle_loss(self.idt_B, self.real_A) * self.opt.lambda_A * self.opt.lambda_identity
        self.loss_G = self.adv_loss(self.D_X(self.fake_B), r_label)
        self.loss_F = self.adv_loss(self.D_Y(self.fake_A), r_label)
        self.loss_cycle_A = self.cycle_loss(self.rec_A, self.real_A) * self.opt.lambda_A
        self.loss_cycle_B = self.cycle_loss(self.rec_B, self.real_B) * self.opt.lambda_B
        
        self.loss_totG = self.loss_G + self.loss_F + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_totG.backward()
        self.optimizer_G.step()
    def updateD(self):
        tmp = self.D_X(self.fake_A)
        f_label = torch.tensor(0.0).to(self.real_A.device).expand_as(tmp)
        r_label = torch.tensor(1.0).to(self.real_A.device).expand_as(tmp)
        for param in self.D_X.parameters():
            param.requires_grad = True
        for param in self.D_Y.parameters():
            param.requires_grad = True
        self.optimizer_D.zero_grad()
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_gpB = self.adv_loss(self.D_X(self.real_B_blur), r_label) 
        self.loss_D_X = (self.adv_loss(self.D_X(self.real_B), r_label) + self.adv_loss(self.D_X(fake_B.detach()), f_label))*0.5
        tot = self.loss_gpB * 0.5 + self.loss_D_X
        tot.backward()

        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_gpA = self.adv_loss(self.D_Y(self.real_A_blur), f_label) 
        self.loss_D_Y = (self.adv_loss(self.D_Y(self.real_A), r_label) + self.adv_loss(self.D_Y(fake_A.detach()), f_label))*0.5
        tot = self.loss_gpA * 0.5 + self.loss_D_Y
        tot.backward()
        self.optimizer_D.step()

    def optimize(self, totstep = 0):
        self.forward()
        self.updateG()
        if totstep % 3 == 0:
            self.updateD()
    
    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate %.7f -> %.7f' % (old_lr, lr))

    def get_current_visuals(self):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            errors_ret[name] = float(getattr(self, 'loss_' + name))
        return errors_ret

    def save_networks(self, epoch):
        for name in self.model_names:
            save_filename = '%s_%s.pth' % (epoch, name)
            save_path = os.path.join(self.save_dir, save_filename)
            net = getattr(self, name)
            torch.save(net.module.cpu().state_dict(), save_path)
            net.cuda(self.gpu_ids[0])

    def load_networks(self, epoch):
        for name in self.model_names:
            load_filename = '%s_%s.pth' % (epoch, name)
            load_path = os.path.join(self.save_dir, load_filename)
            net = getattr(self, name)
            if isinstance(net, torch.nn.DataParallel):
                net = net.module
            print('loading the model from %s' % load_path)
            state_dict = torch.load(load_path, map_location=str(self.device))
            net.load_state_dict(state_dict)