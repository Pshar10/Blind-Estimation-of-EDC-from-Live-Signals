import argparse
import random
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import utils as utils
import process_utils as process 
from dataset import R2DDataset
from model_flex import R2Dnet   #change model architecture model_alter, model_flex, model_FiNS(make sure to use batchsize 128 with fins)
from matplotlib import pyplot as plt
from datetime import datetime
from torch.nn import DataParallel
import sys

"""A script for testing the trained model"""


global model_alter
model_alter = 'model_alter' in sys.modules and not 'model_flex' in sys.modules



def get_model_path(model_dir, prefix):
    # List all files in the given directory
    files = os.listdir(model_dir)
    # Filter files that start with the given prefix
    matched_files = [file for file in files if file.startswith(prefix)]
    # Assuming you want the first match or a specific logic to select the file
    if matched_files:
        # Construct the full path for the first matched file
        full_path = os.path.join(model_dir, matched_files[0])
        return full_path
    else:
        return None
    



def test(args, net, testloader, epoch, tb_writer ,room,location,plot_analysis=False,analysis=True
         ,position_bool = False,write=False,figure = False, save_plots=False,show_plots=False):
    
    net.eval()
    device = 'cuda'
    
    
    maeloss = nn.L1Loss()
    with torch.no_grad():
        n_already_analyzed = 0
        total_test_loss = 0
        quantile_loss = 0


        for batch_idx, data_frame in enumerate(testloader):
            if (analysis and position_bool):
                Reverb_speech,edcs,time_axis_interpolated,n_vals,n_samples,speechfiles,position = data_frame 
            else:
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
            edc_loss_mae , loss_T60, T_true, T_pred,DT20m_true, DT20m_pred, DT_loss   = process.edc_loss(t_prediction, a_prediction, n_prediction, edcs, device,time_axis_interpolated, s_pred,edcs,speechfiles,room,location,mse_flag=False,
             plot_analysis=plot_analysis, apply_mean=True,figure=figure,plot_save=save_plots,plt_show=show_plots)
            # edc_loss_mae=torch.tensor(edc_loss_mae).requires_grad_(True)
            # print(edc_loss_mae,loss_T60)

            

            n_already_analyzed += edcs.shape[0]

            # Calculate noise loss

                # n_vals_true_db = torch.log10(n_vals+1e-15)
                # n_vals_true_db = n_vals
            n_vals_prediction_db = n_prediction  # network already outputs values in dB

            # print("n_vals for  noise loss",n_vals.requires_grad,n_vals_prediction_db.requires_grad)
            # print(n_vals.shape, n_vals_prediction_db.shape)
            # print( n_vals_prediction_db)
            # noise_loss = maeloss(n_vals, n_vals_prediction_db)

            total_loss =  loss_T60 #+ noise_loss


            total_loss_mae =  edc_loss_mae #+ noise_loss

            

            # print('Test Epoch: {} [{}/{} ({:.0f}%)]\tTotal Loss: {:.3f}, ERE Loss: {:.3f}, '
            #     'EDC Loss (MAE, dB): {:.3f}'.format(epoch, n_already_analyzed,
            #                                                                             len(testloader.dataset),
            #                                                                             100. * n_already_analyzed / len(
            #                                                                                 testloader.dataset),
            #                                                                             total_loss.item(), loss_T60.item()  ,
            #                                                                             edc_loss_mae))
            

# paused to write as of now
            
            if write:
                tb_writer.add_scalar('Loss/total_loss_test', loss_T60, epoch)
                tb_writer.add_scalar('Loss/total_loss_test_edc', total_loss_mae, epoch)
                tb_writer.add_scalar('Loss/total_loss_test_DT', DT_loss, epoch)
                tb_writer.add_scalar('Loss/T_true', T_true.item(), epoch)
                tb_writer.add_scalar('Loss/T_pred', T_pred.item(), epoch)
                tb_writer.add_scalar('Loss/DT_true', DT20m_true.item(), epoch)
                tb_writer.add_scalar('Loss/DT_pred', DT20m_pred.item(), epoch)

                if (analysis and position_bool):

                    x_coordinate = position[0, 0].item()  
                    y_coordinate = position[0, 1].item()  
                    z_coordinate = position[0, 2].item()

                    tb_writer.add_scalar('Position/x_coordinate', x_coordinate, epoch)
                    tb_writer.add_scalar('Position/y_coordinate', y_coordinate, epoch)
                    tb_writer.add_scalar('Position/z_coordinate', z_coordinate, epoch)

                tb_writer.flush() 
            else:
                print("not writing")
  

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
    parser.add_argument('--epochs', type=int, default=200, metavar='E',  #changed to 100
                        help='number of epochs to test (default: 200)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR', #changed to S2IR like
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
        torch.backends.cudnn.deterministic = True  # if set as true, dilated convs are really slow
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}


    print('Reading dataset.')


    rooms = ["HL00W", "HL01W", "HL02WL", "HL02WP", "HL03W", "HL04W", "HL05W", "HL06W", "HL08W"] 
    room_locations = ["BC", "FC", "FR", "SiL", "SiR"]




################################################################################################################################################

    # Load the trained model
    # model_path = '/home/prsh7458/work/R2D/model/' + args.model_filename + '.pth'
    # layer_number=9

    
    # path = '/home/prsh7458/work/R2D/hyperparameter/Intense/' #uncomment begin
    if not model_alter:
        path = '/home/prsh7458/work/R2D/hyperparameter/Intense_avg_loss/'
        # path = '/home/prsh7458/work/R2D/hyperparameter/Intense/' #hilbert one
    else:
        path  = '/home/prsh7458/work/R2D/hyperparameter/Intense_model_alter/'

    print(path)

    trial_number = 5
    model_path = get_model_path(path, f'T{trial_number}')

    # model_path = '/home/prsh7458/work/R2D/hyperparameter/T3_R2DNet_lr8e-05_wd7e-05_sch5.pth'
    # model_path = '/home/prsh7458/work/R2D/model/saved_model/' + args.model_filename + '.pth'  ###SAVED MODEL
    # Load the original state dict
    if not model_alter:
        layer_number = int(next((part.replace('layers', '').split('.')[0] for part in model_path.split('_') if 'layers' in part), '9'))
    else:
        layer_number=3 #uncomment end



 ####################################################################################################################################################   
    # Create network
    net = R2Dnet(num_conv_layers = layer_number).to(device)
    net = net.float()
    net = DataParallel(net)

    # Load the original state dict
    # original_state_dict = torch.load(model_path)

    # # Dynamically adjust state_dict based on presence of 'module.' prefix in keys
    # if not list(original_state_dict.keys())[0].startswith('module.'):
    #     # If 'module.' is not present, add it to each key
    #     adjusted_state_dict = {'module.' + k: v for k, v in original_state_dict.items()}
    # else:
    #     # If 'module.' is already present, use the state dict as is
    #     adjusted_state_dict = original_state_dict

    # # Load the adjusted or original state dictionary into your model
    # net.load_state_dict(adjusted_state_dict)

    net.load_state_dict(torch.load(model_path)) ###USE IT WHEN NOT DOING TRIALS

###################### FOR TRAINING DATASET ##########################################################
    
    # rooms = ["Training"] 
    # room_locations = ["all"]

    # train_flag = True
    # analysis = False  
    # pos_bool = False
    # write_bool = True              # if want to save the logs keep it true
    # plot_analysis = True  
    # save_plots = False #set it true if want to save plots
    # show_plots = False #set it true if want to see plots
    # validation = False

###########################################################################################################################################

    train_flag = False          # for selection of dataset 

    analysis = True                  # parameter to test per room per loc

    pos_bool = False                 # if true then dataset with position ("/home/prsh7458/Desktop/scratch4/speech_data/loc/dataset_position")
                                    # is considered, if False then dataset without position data ("/home/prsh7458/Desktop/scratch4/speech_data/loc/dataset")
    write_bool = False              # if want to save the logs keep it true

    plot_analysis = True  #set it true to analyse plots

    save_plots = True #set it true if want to save plots

    show_plots = False  #set it true if want to see plots

    validation = False ### IF THIS IS TRUE THEN ONLY VALIDATION

    noise_test  = False 

    # SNR_level = 50 #dB  #available [10,20,30,40,50]

    loc_idx_noise = 1


    # # # rooms = ["HL00W", "HL01W", "HL02WL", "HL02WP", "HL03W", "HL04W", "HL05W", "HL06W", "HL08W"] 
    # # # room_locations = ["BC", "FC", "FR", "SiL", "SiR"]

###########################################################################################################################################


    figure  = save_plots or show_plots    

    # Testing loop for each room and location

    # for loc_idx_noise in range(5):
    for SNR_level in [10] :
        for room_idx, room in enumerate(rooms):
            for loc_idx, location in enumerate(room_locations):

                if pos_bool:

                    log_dir = f'/home/prsh7458/work/R2D/test_runs_position/{args.model_filename}_{room}_{location}'
                
                elif validation : 
                    log_dir = f'/home/prsh7458/work/R2D/test_runs_validation/{args.model_filename}_R3vival_dataset'

                elif noise_test : 
                    # if location == room_locations[loc_idx_noise] : continue #only use for reverberant noise case
                    log_dir = f'/home/prsh7458/work/R2D/test_runs_dry_noise_test/{args.model_filename}__S1_{room}_{location}_SNR_{SNR_level}'
                    os.makedirs(log_dir, exist_ok= True)

                else:

                    log_dir = f'/home/prsh7458/work/R2D/test_runs/{args.model_filename}_{room}_{location}'


                if write_bool:

                    tb_writer = SummaryWriter(log_dir=log_dir)

                else:

                    tb_writer=None
                    
                # if  ((noise_test )and (location == room_locations[loc_idx_noise])) : continue #only use for reverberant noise case
                    
                print(f'Reading dataset for Room: {room}, Location: {location}.') if not validation else print("Validation data")
                dataset_synthdecays = R2DDataset(train_flag=train_flag, analysis=analysis,
                                                room_idx=room_idx, loc_idx=loc_idx, SNR_level=SNR_level,
                                                room_idx_noise= room_idx,loc_idx_noise=loc_idx_noise, # room _idx and room_idx_noise must be same
                                                position_bool=pos_bool, validation=validation,
                                                noise_test=noise_test)

                testloader = DataLoader(dataset_synthdecays, batch_size=1, shuffle=False)

                
                for epoch in range(1, 2):
                    test(args, net, testloader, epoch, tb_writer, room, location , plot_analysis=plot_analysis, 
                        analysis=analysis,position_bool = pos_bool,write=write_bool,figure = figure, save_plots=save_plots,show_plots=show_plots)
                    if validation: break
                if validation: break
            if validation: break
                    

        if write_bool:
            tb_writer.close()


if __name__ == '__main__':
    import os
    os.system('clear')    
    main()