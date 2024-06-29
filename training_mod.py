import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import utils as utils
import process_utils as process 
from dataset import R2DDataset
from model import R2Dnet
from matplotlib import pyplot as plt

"""Bakcup of training script"""



def train(args, net, trainloader, optimizer, epoch, tb_writer):
    
        net.train()
        device = 'cuda'



        maeloss = nn.L1Loss()

        n_already_analyzed = 0
        for batch_idx, data_frame in enumerate(trainloader):
            Reverb_speech,edcs,time_axis_interpolated,n_vals = data_frame

            # To cuda if available
            n_vals = n_vals.to(device)
            edcs = edcs.to(device)
            # edcs_db_normalized = edcs_db_normalized.to(device)
            Reverb_speech = Reverb_speech.to(device)
            time_axis_interpolated = time_axis_interpolated.to(device)

            # Prediction
            t_prediction, a_prediction, n_prediction = net(Reverb_speech)

            print(".................................................................................................................................")




            n_exp_prediction = torch.clamp(n_prediction, -32, 32)
            # n_vals_prediction = torch.pow(10, n_exp_prediction)
            loss_fn = nn.L1Loss(reduction='none')
            tau_vals = torch.log(torch.tensor([1e6], device=device)) / t_prediction
            time_vals = -time_axis_interpolated * tau_vals  # batchsize x 16000 which is the length of tim axis
            time_vals = time_vals 
            exponentials = torch.exp(time_vals)
            edc = a_prediction * (exponentials - 0)
            noise = n_exp_prediction * torch.linspace(1, 0, time_axis_interpolated.shape[1], device=device)
            edcs_pred = edc + noise

            edc_loss_mae = torch.mean(loss_fn(edcs, edcs_pred))


            # edc_loss_mae=torch.tensor(edc_loss_mae).requires_grad_(True)

            # Calculate noise loss
            if args.exclude_noiseloss:
                noise_loss = 0
            else:
                # n_vals_true_db = torch.log10(n_vals+1e-15)
                # n_vals_true_db = n_vals
                n_vals_prediction_db = 10*torch.log10(n_prediction)  # network already outputs values in dB

                # print("n_vals for  noise loss",n_vals.requires_grad,n_vals_prediction_db.requires_grad)
                noise_loss = maeloss(n_vals, n_vals_prediction_db)
                # noise_loss.requires_grad_(True)
                # noise_loss=torch.tensor(noise_loss).requires_grad_(True)  # set to required grad true


            # print(edc_loss_mae.requires_grad, noise_loss.requires_grad)



            # Add up losses
            total_loss =  edc_loss_mae + noise_loss

            
            # Do optimizer step
            optimizer.zero_grad()
            total_loss.backward()


            optimizer.step()

            n_already_analyzed += edcs.shape[0]
            if batch_idx % args.log_interval == 0:

                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTotal Loss: {:.3f}, Noise Loss: {:.3f}, '
                    'EDC Loss (MAE, dB): {:.3f}'.format(epoch, n_already_analyzed,
                    # 'EDC Loss (MAE, dB): {:.3f}, MSE (dB): {:.3f}'.format(epoch, n_already_analyzed,
                                                                            len(trainloader.dataset),
                                                                            100. * n_already_analyzed / len(
                                                                                trainloader.dataset),
                                                                            total_loss, noise_loss,
                                                                            edc_loss_mae))
                                                                            # edc_loss_mae, edc_loss_mse))
                tb_writer.add_scalar('Loss/Total_train_step', total_loss, (epoch - 1) * len(trainloader) + batch_idx)
                tb_writer.add_scalar('Loss/Noise_train_step', noise_loss, (epoch - 1) * len(trainloader) + batch_idx)
                tb_writer.add_scalar('Loss/MAE_step', edc_loss_mae, (epoch - 1) * len(trainloader) + batch_idx)
                # tb_writer.add_scalar('Loss/MSE_step', edc_loss_mse, (epoch - 1) * len(trainloader) + batch_idx)
                tb_writer.flush()


def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Neural network to predict EDCs parameters from Reverberant speech")
    parser.add_argument('--units-per-layer', type=int, default=400, metavar='N',
                        help='units per layer in the neural network (default: 400)')
    parser.add_argument('--n-layers', type=int, default=3, metavar='N_layer',
                        help='number of layers in the neural network (default: 3)')
    parser.add_argument('--n-filters', type=int, default=64, metavar='N_filt',
                        help='number of filters in the conv neural network (default: 64)')
    parser.add_argument('--relu-slope', type=float, default=0.0, metavar='relu',
                        help='negative relu slope (default: 0.0)')
    parser.add_argument('--dropout', type=float, default=0.0, metavar='DO',
                        help='probability of dropout (default: 0.0)')
    parser.add_argument('--reduction-per-layer', type=float, default=1, metavar='red',
                        help='fraction for reducting the number of units in consecutive layers '
                             '(default: 1 = no reduction)')
    parser.add_argument('--skip-training', action='store_true', default=False,
                        help='skips training and loads previously trained model')
    parser.add_argument('--exclude-noiseloss', action='store_true', default=False,
                        help='has to be true if the noise loss should be excluded')
    parser.add_argument('--model-filename', default='R2DNet',
                        help='filename for saving and loading net (default: DecayFitNet')
    parser.add_argument('--batch-size', type=int, default=2048, metavar='bs',
                        help='input batch size for training (default: 2048)')
    parser.add_argument('--test-batch-size', type=int, default=2048, metavar='bs_t',
                        help='input batch size for testing (default: 2048)')
    parser.add_argument('--epochs', type=int, default=100, metavar='E',  #changed to 100
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=8e-5, metavar='LR', #changed to S2IR like
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--lr-schedule', type=int, default=40, metavar='LRsch',
                        help='learning rate is reduced with every epoch, restart after lr-schedule epochs (default: 40)')
    parser.add_argument('--weight-decay', type=float, default=3e-4, metavar='WD',
                        help='weight decay of Adam Optimizer (default: 3e-4)')
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='SEED',
                        help='random seed (default: 42)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='LOGINT',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    # set up torch and cuda
    use_cuda = args.use_cuda and torch.cuda.is_available()

    # Reproducibility, also add 'env PYTHONHASHSEED=42' to shell
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        # torch.backends.cudnn.deterministic = True  # if set as true, dilated convs are really slow
        # torch.backends.cudnn.benchmark = False

    device = torch.device("cuda")

    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}

    # TensorBoard writer
    tb_writer = SummaryWriter(log_dir='training_runs/' + args.model_filename)

    print('Reading dataset.')
    dataset_synthdecays = R2DDataset()

    # input_transform = {'edcs_db_normfactor': dataset_synthdecays.edcs_db_normfactor}


    trainloader = DataLoader(dataset_synthdecays, batch_size=4, shuffle=True)

    # Create network
    net = R2Dnet().to(device)
    net = net.float()

    # print("net",net.requires_grad_)
    # net = net.float()
    # net = net.cuda()


    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.lr_schedule)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train(args, net, trainloader, optimizer, epoch, tb_writer)
        scheduler.step()
        utils.save_model(net, '/home/prsh7458/work/R2D/model/' + args.model_filename + '.pth')


if __name__ == '__main__':
    main()
