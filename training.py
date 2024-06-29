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
from model_flex import R2Dnet   #change model architecture model_alter, model_flex, model_FiNS(make sure to use batchsize 128 with fins),model_log_spectral_CNN(for log_spectral_features)
from matplotlib import pyplot as plt
from datetime import datetime
from torch.nn import DataParallel

"""A script to train the  model"""
def train(args, net, trainloader, optimizer, epoch, tb_writer):
    
    net.train()
    device = 'cuda'
    
    
    maeloss = nn.L1Loss()

    n_already_analyzed = 0
    for batch_idx, data_frame in enumerate(trainloader):
        Reverb_speech,edcs,time_axis_interpolated,n_vals,n_samples,speech_files = data_frame 

        # To cuda if available
        n_vals = n_vals.to(device)
        edcs = edcs.to(device)
        Reverb_speech = Reverb_speech.to(device)
        time_axis_interpolated = time_axis_interpolated.to(device)
        n_samples = n_samples.to(device)

        # Prediction
        t_prediction, a_prediction, n_prediction, s_prediction = net(Reverb_speech)

        # print(".................................................................................................................................")
  

        '''
        Calculate Losses
        '''


        # Calculate EDC Loss

        edc_loss_mae,edc_loss_edt, __, __ ,__,__ ,__ = process.edc_loss(t_prediction, a_prediction, n_prediction, edcs, device,time_axis_interpolated,s_prediction,edcs,speech_files,mse_flag =False)
        
        # noise_loss =  maeloss(n_vals, n_prediction.squeeze(0))

        total_loss = edc_loss_mae  + 1 *  edc_loss_edt # + noise_loss

        # edc_loss_mae=torch.tensor(edc_loss_mae).requires_grad_(True)

        # # Calculate noise loss
        # if args.exclude_noiseloss:
        #     noise_loss = 0
        # else:
        #     # n_vals_true_db = torch.log10(n_vals+1e-15)
        #     # n_vals_true_db = n_vals
        #     n_vals_prediction_db = n_prediction  # network already outputs values in dB

        #     # print("n_vals for  noise loss",n_vals.requires_grad,n_vals_prediction_db.requires_grad)
        #     # print("n_vals: ",n_vals.shape,"n_vals_prediction_db : ", n_vals_prediction_db.shape)
        #     noise_loss =  maeloss(n_vals, n_vals_prediction_db)

            
        #     # noise_loss.requires_grad_(True)
        #     # noise_loss=torch.tensor(noise_loss).requires_grad_(True)  # set to required grad true


        # print(edc_loss_mae.requires_grad, noise_loss.requires_grad)
        # fs = 16000 #sampling frequency

        # fs_new = (fs*(edcs.shape[1]/n_samples)) # Desired output length of edc is 16000 samples
        # fs_pred = (fs*(edcs.shape[1]/s_prediction)) 

        # timeaxis_fs_loss  =   maeloss(fs_new.unsqueeze(1),fs_pred)

        # Add up losses
        # torch.set_grad_enabled(True)
        # total_loss =  edc_loss_mae #+ noise_loss #+ timeaxis_fs_loss (Excluded noise loss)
        # total_loss = total_loss.requires_grad_(True)
        # print("Requires Grad - total_loss: {}, edc_loss_mae: {}, noise_loss: {}".format(total_loss.requires_grad, edc_loss_mae.requires_grad, noise_loss.requires_grad))
        # print("Requires Grad - total_loss: {}, edc_loss_mae: {}, noise_loss: {}".format(total_loss.grad_fn, edc_loss_mae.grad_fn, noise_loss.grad_fn))
        
        # Do optimizer step
        optimizer.zero_grad()
        # total_loss.requires_grad = True
        total_loss.backward()

        # for name, param in net.named_parameters():
        #     if param.requires_grad:s
        #         print(f"{name}: gradient = {param.grad is not None}")

        optimizer.step()

        n_already_analyzed += edcs.shape[0]
        if batch_idx % args.log_interval == 0:

            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTotal Loss: {:.3f}, ERE Loss: {:.3f}, '
            #     'EDC Loss (MAE, dB): {:.3f}, Sample_length Loss (MAE): {:.3f}'.format(epoch, n_already_analyzed,
            #                                                                             len(trainloader.dataset),
            #                                                                             100. * n_already_analyzed / len(
            #                                                                                 trainloader.dataset),
            #                                                                             total_loss, edc_loss_edt,
            #                                                                             edc_loss_mae, 0))

            #                                                             # # edc_loss_mae, edc_loss_mse))
            
            tb_writer.add_scalar('Loss/Total_train_step', total_loss, (epoch - 1) * len(trainloader) + batch_idx)
            tb_writer.add_scalar('Loss/ERE_train_step', edc_loss_edt, (epoch - 1) * len(trainloader) + batch_idx)
            tb_writer.add_scalar('Loss/MAE_step', edc_loss_mae, (epoch - 1) * len(trainloader) + batch_idx)
            # tb_writer.add_scalar('Loss/Sample Length_step', timeaxis_fs_loss, (epoch - 1) * len(trainloader) + batch_idx)
            # tb_writer.add_scalar('Loss/MSE_step', edc_loss_mse, (epoch - 1) * len(trainloader) + batch_idx)
            tb_writer.flush()



def test(args, net, testloader, epoch, tb_writer,return_avg_loss=False):
    
    net.eval()
    device = 'cuda'
    
    maeloss = nn.L1Loss()
    with torch.no_grad():
        n_already_analyzed = 0
        total_test_loss = 0
        quantile_loss = 0
        total_loss_avg = []


        for batch_idx, data_frame in enumerate(testloader):
            Reverb_speech,edcs,time_axis_interpolated,n_vals,n_samples,speechfiles = data_frame

            # To cuda if available
            n_vals = n_vals.to(device)
            edcs = edcs.to(device)
            Reverb_speech = Reverb_speech.to(device)
            time_axis_interpolated = time_axis_interpolated.to(device)

            # Prediction
            t_prediction, a_prediction, n_prediction , s_pred= net(Reverb_speech)

            # print(".................................................................................................................................")
    

            '''
            Calculate Losses
            '''


            # Calculate EDC Loss
            edc_loss_mae , edc_loss_edt, __, __ ,__,__,__  = process.edc_loss(t_prediction, a_prediction, n_prediction, edcs, device,time_axis_interpolated, s_pred,edcs,speechfiles,mse_flag=False,
            apply_mean=True)
            # edc_loss_mae=torch.tensor(edc_loss_mae).requires_grad_(True)

            n_already_analyzed += edcs.shape[0]


            n_vals_prediction_db = n_prediction  # network already outputs values in dB


            noise_loss = maeloss(n_vals, n_vals_prediction_db)

            total_loss =  edc_loss_mae + edc_loss_edt #+ noise_loss

            # print('Test Epoch: {} [{}/{} ({:.0f}%)]\tTotal Loss: {:.3f}, ERE Loss: {:.3f}, '
            #     'EDC Loss (MAE, dB): {:.3f}'.format(epoch, n_already_analyzed,
            #                                                                             len(testloader.dataset),
            #                                                                             100. * n_already_analyzed / len(
            #                                                                                 testloader.dataset),
            #                                                                             total_loss, edc_loss_edt,
            #                                                                             edc_loss_mae))
            

            tb_writer.add_scalar('Loss/Total_test_step', total_loss, (epoch - 1) * len(testloader) + batch_idx)
            tb_writer.add_scalar('Loss/ERE_test_step', edc_loss_edt, (epoch - 1) * len(testloader) + batch_idx)
            tb_writer.add_scalar('Loss/Test_EDC_step', edc_loss_mae, (epoch - 1) * len(testloader) + batch_idx)
            tb_writer.flush()  

            if return_avg_loss:
                total_loss_avg.append(total_loss.item())  


        if return_avg_loss:
            total_loss_avg_mean = (sum(total_loss_avg) / len(total_loss_avg))
            return torch.tensor(total_loss_avg_mean, device=total_loss.device)
        else:
            return total_loss
        

        # return total_loss




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
                        help='filename for saving and loading net (default: R2DNet')

    parser.add_argument('--test-batch-size', type=int, default=2048, metavar='bs_t',
                        help='input batch size for testing (default: 2048)')
    






    ################################################  TUNING   #############################################################################


    parser.add_argument('--epochs', type=int, default=80
                        , metavar='E',  
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--batch-size', type=int, default=128
                        , metavar='bs',
                        help='input batch size for training (default: 2048)')
    parser.add_argument('--lr', type=float, default= 8e-5  ##########CHANGE TO 8E-5
                        , metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--lr-schedule', type=int, default=40
                        , metavar='LRsch',
                        help='learning rate is reduced with every epoch, restart after lr-schedule epochs (default: 40)')
    parser.add_argument('--weight-decay', type=float, default=  3e-4  ##########CHANGE TO 3e-4
                        , metavar='WD',
                        help='weight decay of Adam Optimizer (default: 3e-4)')
    

    
    #################################################################################################################################





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
        torch.backends.cudnn.deterministic = True  # if set as true, dilated convs are really slow
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}

    # TensorBoard writer
    current_time = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
    log_dir = f'/home/prsh7458/work/R2D/training_runs/{args.model_filename}_{args.batch_size}_{args.lr}_{args.epochs}_{current_time}'
    tb_writer = SummaryWriter(log_dir=log_dir)

    print('Reading dataset.')
    dataset_synthdecays = R2DDataset()
    dataset_test = R2DDataset(train_flag=False)



    trainloader = DataLoader(dataset_synthdecays, batch_size=args.batch_size, shuffle=True) #batch2048 128 for model_alter
    testloader = DataLoader(dataset_test, batch_size=64, shuffle=True) #batch2048 128 for model_alter

    # Create network
    net = R2Dnet().to(device)
    net = net.float()
    net = DataParallel(net)




    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.lr_schedule) #this is for model
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.7) # this is for model_alter

# early stopping logic 
    best_test_loss = float('inf')
    epochs_no_improve = 0
    patience = 5  # Number of epochs to wait for improvement before stopping the training
    delta = 0.001 



    # Training loop


    for epoch in range(1, args.epochs+1):
        train(args, net, trainloader, optimizer, epoch, tb_writer)
        test(args, net, testloader, epoch, tb_writer)
        scheduler.step()
        utils.save_model(net, '/home/prsh7458/work/R2D/model/' + args.model_filename + '.pth')


    # for epoch in range(1, args.epochs + 1):


    #     train(args, net, trainloader, optimizer, epoch, tb_writer)
    #     current_test_loss = test(args, net, testloader, epoch, tb_writer) 
    #     scheduler.step()

    #     # Check if the test loss improved more than delta
    #     if best_test_loss - current_test_loss > delta:
    #         best_test_loss = current_test_loss
    #         epochs_no_improve = 0
    #     else:
    #         epochs_no_improve += 1

    #     # Check for early stopping
    #     if epochs_no_improve >= patience:
    #         print(f"Early stopping triggered at epoch {epoch}")
    #         break

    # utils.save_model(net, '/home/prsh7458/work/R2D/model/' + args.model_filename + '.pth')


if __name__ == '__main__':
    import os
    os.system('clear')    
    main()
