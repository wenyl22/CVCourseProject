from options.train_options import TrainOptions
import dataset.dataset as ds
from torch.utils.data import DataLoader
from model.cyclegan import CycleGAN
import os
import matplotlib.pyplot as plt
import utils.utils as utils
def plot_num(all_losses, dir):
    plt.figure()
    for key, value in all_losses.items():
        plt.plot(value, label=key)
    plt.legend()
    plt.savefig(f'{dir}/loss.png')
    plt.close()

def main():
    opt = TrainOptions().parse()
    train_dataset = ds.Dataset(opt)
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    model = CycleGAN(opt)
    totstep = 0
    all_losses = {}
    losses = {}
    dir = opt.checkpoints_dir + '/' + opt.name
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_iter = 0
        for i, data in enumerate(train_dataloader):            
            model.input(data)
            model.optimize(totstep)
            totstep += opt.batch_size
            epoch_iter += opt.batch_size
            cur_loss = model.get_current_losses()
            for key, value in cur_loss.items():
                if key not in losses:
                    losses[key] = []
                losses[key].append(value)
            if totstep % 100 == 0:
                for key, value in losses.items():
                    if key not in all_losses:
                        all_losses[key] = []
                    all_losses[key].append(sum(value)/len(value))
                    losses[key] = []
                plot_num(all_losses, dir)
                print(f"epoch: {epoch}, iter: {epoch_iter}, loss: {cur_loss}")
                # log the losses in dir/loss.txt
                with open(f'{dir}/loss.txt', 'a') as f:
                    f.write(f"totstep:{totstep} | loss: {cur_loss}\n")
            if totstep % 1000 == 0:
                pic = model.get_current_visuals()
                for key, value in pic.items():
                    img = utils.tensor2im(value)
                    if not os.path.exists(dir + '/' + str(totstep)):
                        os.mkdir(dir + '/' + str(totstep))
                    utils.save_image(img, f'{dir}/{totstep}/{key}.png')
        model.update_learning_rate()
        if epoch % opt.save_epoch_freq == 0:
            model.save_networks(epoch)

if __name__=='__main__':
    main()