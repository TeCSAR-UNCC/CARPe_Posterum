import torch
import time
import os
from statistics import mean
from thop import profile
from torch_geometric.data.batch import Batch as tgb

import utils.loader as dl
import utils.network as net
import utils.util as ut


#**************************************************************#
#***********************   MAIN   *****************************#
#**************************************************************#
data_files = ['eth', 'hotel', 'univ','zara1','zara2']
OBS_STEP = 8
PRED_STEP = 12
NUM_POINTS_PER_POS = 2
SAVE_MODEL=False
TRAIN=False
VAL_INTERVAL=1
output_size = NUM_POINTS_PER_POS*PRED_STEP
num_features = NUM_POINTS_PER_POS*OBS_STEP

if TRAIN==True:
    saveFolder = 'train0' # folder name where models will be saved
    lr = 0.01
    EPOCHS=80
    for test_file in data_files:
        print("Test file: " + test_file)
        if SAVE_MODEL==True:
            if not os.path.exists("models/"+str(saveFolder)+"/"):
                    os.makedirs("models/"+str(saveFolder)+"/")

        device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
        model = net.NetGINConv(num_features, output_size).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

        data_dir = 'datasets/'+test_file+'/'
        data_folders = ['test','train', 'val']


        _, train_loader = dl.data_loader(data_dir+data_folders[1])
        _, val_loader = dl.data_loader(data_dir+data_folders[2])
        # print("Data Loaded")
        
        best_info = [1000.0, 1000.0, 0]
        for epoch in range(0, EPOCHS):
            losses = ut.train(model, train_loader, optimizer, device, obs_step=OBS_STEP)
            # print(mean(losses))
            if(epoch%VAL_INTERVAL==0):
                ade, fde = ut.test(model, val_loader, device)
                if ade < best_info[0]:
                    best_info[0] = ade
                    best_info[1] = fde
                    best_info[2] = epoch
                    # print("ADE: " + str(ade) + "  FDE: " + str(fde) + "   Epoch: " + str(epoch))
                    if (SAVE_MODEL==True):
                        model_path = "models/"+str(saveFolder)+"/"+ test_file+".pt"
                        torch.save(model.state_dict(), model_path)
                    # print("ADE: " + str(ade) + "  FDE: " + str(fde) + "   Epoch: " + str(epoch))
        print(test_file + "   Best ADE: " + str(best_info[0]) + "   FDE: " + str(best_info[1]) + "   Epoch: " + str(best_info[2]) + "   lr: " + str(lr) + "\n\n")


else:
    all_ops = []
    times = []
    for test_file in data_files:
        total_traj = 0
        #device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        device = torch.device('cpu')
        model = net.NetGINConv(num_features, output_size).to(device)

        model_folder = 'models/TRAINED/'
        #model.load_state_dict(torch.load(model_folder + test_file + '.pt'))
        model.load_state_dict(torch.load(model_folder + test_file + '.pt', map_location='cpu'))
        data_dir = 'datasets/'+test_file+'/'
        data_folders = ['test','train', 'val']


        _, test_loader = dl.data_loader(data_dir+data_folders[0], batch_size=1)

        ade_batches, fde_batches = [], []
        model.eval()
        for batch in test_loader:
            batch = [tensor.to(device) for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
                non_linear_ped, loss_mask, seq_start_end, frame_id) = batch
            total_traj += pred_traj_gt.size(0)

            data_list = ut.getGraphDataList(obs_traj,obs_traj_rel, seq_start_end)
            graph_batch = tgb.from_data_list(data_list)

            start = time.time()
            pred_traj = model(obs_traj_rel, graph_batch.x.to(device), graph_batch.edge_index.to(device))
            end = time.time()
            times.append(end-start)

            pred_traj = pred_traj.reshape(pred_traj.shape[0],12,2).detach()

            pred_traj_real = ut.relative_to_abs(pred_traj, obs_traj[:,:,-1,:].squeeze(1))

            ade_batches.append(torch.sum(ut.displacement_error(pred_traj_real, pred_traj_gt, mode='raw')).detach().item())
            fde_batches.append(torch.sum(ut.final_displacement_error(pred_traj_real[:,-1,:], pred_traj_gt[:,:,-1,:].squeeze(1), mode='raw')).detach().item())

            ops, params = profile(model, inputs=(obs_traj_rel, graph_batch.x.to(device), graph_batch.edge_index.to(device)))
            all_ops.append(ops)

        ade = sum(ade_batches) / (total_traj * 12)
        fde = sum(fde_batches) / (total_traj)
        print(test_file + "  ADE: " + str(ade) + "  FDE: " + str(fde))

    print("Average Execution Time: " + str(mean(times)) + " sec")
    print("Params: " + str(params))
    print("Average OPs: " + str(mean(all_ops)))
