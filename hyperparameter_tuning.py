import argparse
import logging
import random

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model_flex import R2Dnet   #change model architecture model_alter, model_flex, model_FiNS(make sure to use batchsize 128 with fins)
import optuna
import sys
import joblib
from torch.nn import DataParallel
import training
import utils as utils
from dataset import R2DDataset


"""A script that helps in hyperparameter tuning using TPE 
Implementation motivated by Georg Götz, Ricardo Falcón Pérez, Sebastian J. Schlecht, and Ville Pulkki,
"Neural network for multi-exponential sound energy decay analysis",
The Journal of the Acoustical Society of America, 152(2), pp. 942-953, 2022, https://doi.org/10.1121/10.0013416.
Github Link:https://github.com/georg-goetz/DecayFitNet
"""

global model_alter

model_alter = 'model_alter' in sys.modules and not 'model_flex' in sys.modules


def objective(trial, args):
    lr = trial.suggest_categorical("lr", [1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5,
                                          1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4])
                                        #   1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3, 9e-3])
    
    wd = trial.suggest_categorical("wd", [1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5,
                                          1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4])
                                        #   1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3, 9e-3])
    # cos_schedule_factor = trial.suggest_int('cos_sch', 1, 10)
    if not model_alter:
        n_layers = trial.suggest_int('n_layers', 7, 9)
    else: n_layers = 3
    
    # cos_schedule = cos_schedule_factor * 5
    cos_schedule = 2

    print('==== Trial {}:\t lr: {}\t wd: {}\t  '
          ' layer: {} ===='.format(trial.number, lr, wd, n_layers))

    # set up torch and cuda
    use_cuda = args.use_cuda and torch.cuda.is_available()

    # Reproducibility, also add 'env PYTHONHASHSEED=42' to shell
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True  # if set as true, dilated convs are really slow
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}

    # TensorBoard writer
    tb_writer = SummaryWriter(log_dir='/home/prsh7458/work/R2D/hyperparameter/Intense/training_runs/' + args.model_filename)

    print('Reading dataset.')
    dataset_synthdecays = R2DDataset()
    dataset_test = R2DDataset(train_flag=False)



    trainloader = DataLoader(dataset_synthdecays, batch_size=128, shuffle=True) #batch2048 128 for model_alter
    testloader = DataLoader(dataset_test, batch_size=64, shuffle=True) #batch2048 128 for model_alter

    # Create network
    # net = R2Dnet().to(device)
    net = R2Dnet(n_layers).to(device)
    net = net.float()
    net = DataParallel(net)

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, cos_schedule)

    min_epochs = int(np.max([np.ceil(args.epochs/cos_schedule)*cos_schedule, 5*cos_schedule])) ### here we have kept it 80 epochs max
    total_test_loss = None
    for epoch in range(0, min_epochs):
        training.train(args, net, trainloader, optimizer, epoch, tb_writer) #args, net, trainloader, optimizer, epoch, tb_writer
        total_test_loss = training.test(args, net, testloader, epoch, tb_writer, return_avg_loss=False) #args, net, testloader, epoch, tb_writer

        if ((epoch+1) % cos_schedule) == 0:
            iteration = int(np.floor((epoch+1)/cos_schedule))
            trial.report(total_test_loss, iteration)
            if trial.should_prune():
                raise optuna.TrialPruned()

        scheduler.step()

    if not model_alter:
        path = '/home/prsh7458/work/R2D/hyperparameter/Intense/'
    else:
        path = '/home/prsh7458/work/R2D/hyperparameter/Intense_model_alter/'

    utils.save_model(net, path + f'T{trial.number}_' + args.model_filename + '_lr' + str(lr) + '_wd' +
                     str(wd) + '_sch' + str(cos_schedule) + '_layers' + str(n_layers) + '.pth')

    return total_test_loss


if __name__ == '__main__':
    # Argument parser
    parser = argparse.ArgumentParser(description="Neural network to predict EDCs parameters from Reverberant speech")
    
    parser.add_argument('--model-filename', default='R2DNet',
                        help='filename for saving and loading net (default: R2DNet')
    
##################################################################################################################################################
    parser.add_argument('--batch-size', type=int, default=128, metavar='bs',
                        help='input batch size for training (default: 2048)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='bs_t',
                        help='input batch size for testing (default: 2048)')
    parser.add_argument('--epochs', type=int, default=80, metavar='E',  
                        help='number of epochs to test (default: 200)')
    parser.add_argument('--lr', type=float, default=8e-5, metavar='LR', 
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--lr-schedule', type=int, default=40, metavar='LRsch',
                        help='learning rate is reduced with every epoch, restart after lr-schedule epochs (default: 40)')
    parser.add_argument('--weight-decay', type=float, default=3e-4, metavar='WD',
                        help='weight decay of Adam Optimizer (default: 3e-4)')
    
##############################################################################################################################################################3

    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='SEED',
                        help='random seed (default: 42)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='LOGINT',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    sampler = optuna.samplers.TPESampler(seed=args.seed)

    optuna.logging.get_logger('optuna').addHandler(logging.StreamHandler(sys.stdout))
    study = optuna.create_study(sampler=sampler, pruner=optuna.pruners.MedianPruner(n_min_trials=10),
                                direction='minimize')

    study.optimize(lambda trial: objective(trial, args), timeout= int(86400*2))  # 86400 seconds = 1 day

    if not model_alter:
        joblib.dump(study, f'/home/prsh7458/work/R2D/hyperparameter/Intense/study.pkl') 
    else:
        joblib.dump(study, f'/home/prsh7458/work/R2D/hyperparameter/Intense_model_alter/study.pkl')     
