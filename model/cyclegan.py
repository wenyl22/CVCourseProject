import torch
import itertools
import torch.nn as nn
from model.G_D import Generator, Discriminator, get_scheduler
import os
from collections import OrderedDict
class CycleGAN(nn.Module):
    def __init__(self, opt):
        super(CycleGAN, self).__init__()
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0]))
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        # G: A -> B, F: B -> A
        # D_X: discriminate realA vs generated from F(B)
        # D_Y: discriminate realB vs generated from G(A)
        self.model_names = ['G', 'F', 'D_X', 'D_Y']
        self.visual_names = ['real_A', 'fake_B', 'rec_A', 'idt_B', 'real_B', 'fake_A', 'rec_B', 'idt_A']
        self.loss_names = ['D_X', 'G', 'cycle_A', 'idt_A',  'D_Y', 'F','cycle_B', 'idt_B']
        self.G = Generator(opt, opt.input_nc, opt.output_nc)
        self.F = Generator(opt, opt.output_nc, opt.input_nc)
        self.D_X = Discriminator(opt, opt.input_nc)
        self.D_Y = Discriminator(opt, opt.output_nc)
        self.adv_loss = nn.MSELoss()
        self.cycle_loss = nn.L1Loss()
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.G.parameters(), self.F.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.D_X.parameters(), self.D_Y.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizers = [self.optimizer_G, self.optimizer_D]
        self.schedulers = [get_scheduler(optimizer, opt) for optimizer in self.optimizers]

    def input(self, input):
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
    def forward(self):
        self.fake_B = self.G(self.real_A) # G(A)
        self.rec_A = self.F(self.fake_B) # F(G(A))
        self.fake_A = self.F(self.real_B) # F(B)
        self.rec_B = self.G(self.fake_A) # G(F(B))
    def updateG(self):
        r_label = torch.tensor(1.0).to(self.real_A.device)
        for param in self.D_X.parameters():
            param.requires_grad = False
        for param in self.D_Y.parameters():
            param.requires_grad = False
        self.optimizer_G.zero_grad()

        self.idt_A = self.G(self.real_B)
        self.loss_idt_A = self.cycle_loss(self.idt_A, self.real_B) * self.opt.lambda_B * self.opt.lambda_identity
        self.idt_B = self.F(self.real_A)
        self.loss_idt_B = self.cycle_loss(self.idt_B, self.real_A) * self.opt.lambda_A * self.opt.lambda_identity

        self.loss_G = self.adv_loss(self.D_X(self.fake_B), r_label)
        self.loss_F = self.adv_loss(self.D_Y(self.fake_A), r_label)
        self.loss_cycle_A = self.cycle_loss(self.rec_A, self.real_A) * self.opt.lambda_A
        self.loss_cycle_B = self.cycle_loss(self.rec_B, self.real_B) * self.opt.lambda_B

        self.loss_totG = self.loss_G + self.loss_F + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_totG.backward()
        self.optimizer_G.step()
    def updateD(self):
        f_label = torch.tensor(0.0).to(self.real_A.device)
        r_label = torch.tensor(1.0).to(self.real_A.device)
        for param in self.D_X.parameters():
            param.requires_grad = True
        for param in self.D_Y.parameters():
            param.requires_grad = True
        self.optimizer_D.zero_grad()
        self.loss_D_X = self.adv_loss(self.D_X(self.real_A), r_label) + self.adv_loss(self.D_X(self.fake_A.detach()), f_label)
        self.loss_D_X.backward()
        self.loss_D_Y = self.adv_loss(self.D_Y(self.real_B), r_label) + self.adv_loss(self.D_Y(self.fake_B.detach()), f_label)
        self.loss_D_Y.backward()
        self.optimizer_D.step()

    def optimize(self):
        self.forward()
        self.updateG()
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
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata
            # patch InstanceNorm checkpoints prior to 0.4
            # need to copy keys here because we mutate in loop
            for key in list(state_dict.keys()):  
                self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
            net.load_state_dict(state_dict)