from options.train_options import TrainOptions
import dataset.dataset as ds
from torch.utils.data import DataLoader
from model.cyclegan import CycleGAN
from utils.visualizer import Visualizer
import time
def main():
    opt = TrainOptions().parse()
    train_dataset = ds.Dataset(opt)
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    dataset_size = len(train_dataset)

    model = CycleGAN(opt)
    visualizer = Visualizer(opt)
    totstep = 0
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_iter = 0
        iter_data_time = time.time()
        for i, data in enumerate(train_dataloader):
            iter_start_time = time.time()
            if totstep % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            
            model.input(data)
            model.optimize()
            #print(len(data), len(train_dataset), opt.batch_size)
            totstep += opt.batch_size
            epoch_iter += opt.batch_size
            if totstep % opt.display_freq == 0:
                save_result = totstep % opt.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)
            if totstep % opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)
            iter_data_time = time.time()
        model.update_learning_rate()
        if epoch % opt.save_epoch_freq == 0:
            model.save_networks(epoch)

if __name__=='__main__':
    main()