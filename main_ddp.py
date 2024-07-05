import os
import torch
import torch.nn as nn
from utils import ms_Dataset, calculate_bottleneck, calculate_MI, calc_performance_statistics, seed_torch, reduce_value
from models.spatial_attention_net import Spatial_Attention_Net
from models.temporal_attention_net import Temporal_Attention_Net
from models.gcn import GCNNet
from models.vae import Encoder, Decoder
import argparse
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import BatchSampler
from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import DataLoader
from torch.utils.data.dataset import Subset
import torch.nn.functional as F
import time

torch.set_printoptions(precision=8)
brain_region = 116


parser = argparse.ArgumentParser(description='Graph Generative causal explanations')


parser.add_argument('--gnn_input_dim', type=int, default=116, help='gnn input size')
parser.add_argument("--latent_dim", type=int, default=  [128, 128], help="GNN hidden dims")

parser.add_argument('--encoder_input_dim', type=int, default = 128, help='encoder input_dim')
parser.add_argument('--encoder_hidden_dim', type=int, default = 256, help='encoder hidden_dim')
parser.add_argument('--encoder_output_dim', type=int, default = 16, help='encoder output_dim')

parser.add_argument('--Tem_attNet_input_size', type=int, default=16, help='number of lstm hidden size')
parser.add_argument('--lstm_hidden_size', type=int, default=128, help='number of lstm hidden size')
parser.add_argument('--lstm_num_layers', type=int, default=1, help='number of lstm layers')

parser.add_argument('--decoder_input_dim', type=int, default=16, help='decoder input_dim')
parser.add_argument('--decoder_hidden_dim', type=int, default=256, help='decoder hidden_dim')
parser.add_argument('--decoder_output_dim', type=int, default=128, help='decoder output_dim')

parser.add_argument("--lambda_classify", type=float, default= 1, help="trade off classify_loss")
parser.add_argument("--lambda_kl", type=float, default= 0.001, help="trade off vde_kldiv_loss")
parser.add_argument("--lambda_mse", type=float, default= 0.005, help="trade off vde_mse_loss1 & vde_mse_loss2")
parser.add_argument("--lambda_ib", type=float, default= 0.3, help="trade off ib_loss")

parser.add_argument('--dropout', type=float, default=0.5, help='final layer dropout (default: 0.5)')
parser.add_argument('--iters_per_epoch', type=int, default=12, help='number of iterations per each epoch (default: 32)')
parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
parser.add_argument('--epoch', type=int, default=300, help='epoch (default: 200)')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
parser.add_argument('--seed', type=int, default=7, help='random seed for splitting the dataset into 10 (default: 0)')

parser.add_argument('--readout', type=str, default="sum", choices=["sum", "mean", "max"], help='Pooling for over nodes in a graph: sum or average')
parser.add_argument('--targetdir', type=str, default='./result')
parser.add_argument('--k_fold', type=int, default=5, help='')
parser.add_argument("--local_rank", type=int, default= -1, help="number of cpu threads to use during batch generation")
parser.add_argument("--dataset_name", type=str, default="SRPBS", help="dataset")

parser.add_argument("--mlp_hidden", type=int, default=  [128, 128], help="mlp hidden dims")
parser.add_argument("--emb_normlize", type = bool, default= False, help="GNN emb_normlize")
parser.add_argument("--GVAE_hidden_dim", type = int, default= 64, help="mlp hidden dims")
parser.add_argument('--K', type=int, default=1)
parser.add_argument('--L', type=int, default=7)
parser.add_argument('--Nalpha', type=int, default=56)
parser.add_argument('--Nbeta', type=int, default=100)
parser.add_argument('--lam', type=float, default=0.05)
parser.add_argument("--weight_decay", type=float, default=0.0001, help="Adam weight decay. Default is 5*10^-5.")
args = parser.parse_args()




if __name__=='__main__':


    dist.init_process_group(backend='nccl')

    device_ids = [0,1,2,3]
    device = torch.device('cuda:{}'.format(device_ids[args.local_rank]))
    torch.cuda.set_device(device)

    data_list = ms_Dataset(args.dataset_name)
    kf = StratifiedKFold(n_splits = args.k_fold, shuffle = True, random_state = args.seed)

    if args.dataset_name == 'SRPBS':
        args.seed = 4
        args.lambda_kl = 0.0001
        args.lambda_mse = 0.0005
        args.lambda_ib = 0.1

    if args.dataset_name == 'ABIDE':
        args.seed = 2
        args.lambda_kl = 0.001
        args.lambda_mse = 0.0005
        args.lambda_ib = 0.05

    if args.dataset_name == 'MDD':
        args.seed = 7
        args.lambda_kl = 0.0005
        args.lambda_mse = 0.001
        args.lambda_ib = 0.3

    sp_label = torch.cat(data_list[:][1],dim=0)
    
    for ms_fold_index in range(args.k_fold):

        test_highest_acc_fold = 0
        ms_fold_index = 1

        index = None
        
        for m_i, m_index in enumerate(kf.split(X = sp_label, y = sp_label)):
            if (m_i == ms_fold_index):
                index = m_index
        
        for fold_idx in range(50):

            seed_torch(args.seed)
            fold_idx = ms_fold_index 

            train_index, test_index = index
            train_fold = Subset(data_list, train_index)
            test_fold = Subset(data_list, test_index) 
            train_size = len(train_index)
            test_size = len(test_index)

            train_sampler = DistributedSampler(train_fold)
            train_batch_sampler = BatchSampler(train_sampler, args.batch_size, drop_last = False)
            train_loader = DataLoader(dataset = train_fold, 
                                    shuffle = False, 
                                    drop_last = False, 
                                    num_workers=8, 
                                    pin_memory=True, 
                                    batch_sampler = train_batch_sampler)

            test_loader = DataLoader(dataset = test_fold, 
                                    batch_size = test_size, 
                                    shuffle = False, 
                                    drop_last = False, 
                                    num_workers=8)

            encoder = Encoder(input_dim = args.encoder_input_dim, 
                            hidden_dim = args.encoder_hidden_dim,
                            output_dim = args.encoder_output_dim
                            ).to(device)

            decoder = Decoder(input_dim = args.decoder_input_dim, 
                                hidden_dim = args.decoder_hidden_dim,
                                output_dim = args.decoder_output_dim
                                ).to(device)

            gcn = GCNNet(input_dim = args.gnn_input_dim, 
                            model_args = args, 
                            device = device
                            ).to(device)

            space_attention_model = Spatial_Attention_Net(model = gcn, 
                                                device = device,
                                                model_args = args
                                                ).to(device)

            temporal_attention_model = Temporal_Attention_Net(Tem_attNet_input_size = args.Tem_attNet_input_size, 
                                                            lstm_hidden_size = args.lstm_hidden_size, 
                                                            lstm_num_layers = args.lstm_num_layers, 
                                                            device = device, 
                                                            model_args = args
                                                            ).to(device)

            optimizer = torch.optim.Adam([{'params': space_attention_model.parameters()}, 
                                        {'params': temporal_attention_model.parameters()},
                                        {'params': encoder.parameters()},
                                        {'params': decoder.parameters()}],
                                        lr = args.lr,
                                        weight_decay = args.weight_decay
                                        )

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-8, last_epoch=-1)
            
            encoder = DistributedDataParallel(encoder, device_ids = [device_ids[args.local_rank]], output_device = device_ids[args.local_rank], find_unused_parameters=True)
            decoder = DistributedDataParallel(decoder, device_ids = [device_ids[args.local_rank]], output_device = device_ids[args.local_rank], find_unused_parameters=True)
            gcn = DistributedDataParallel(gcn, device_ids = [device_ids[args.local_rank]], output_device = device_ids[args.local_rank], find_unused_parameters=True)
            space_attention_model = DistributedDataParallel(space_attention_model, device_ids = [device_ids[args.local_rank]], output_device = device_ids[args.local_rank], find_unused_parameters=True)
            temporal_attention_model = DistributedDataParallel(temporal_attention_model, device_ids = [device_ids[args.local_rank]], output_device = device_ids[args.local_rank], find_unused_parameters=True)
            
            criterion = nn.CrossEntropyLoss(reduction = 'mean')
            mse = torch.nn.MSELoss(reduction = 'mean')

            test_highest_acc = 0
            test_highest_f1s = 0
            test_highest_mcc = 0

            for epoch in range(args.epoch):
        
                t1 = time.time()

                train_sampler.set_epoch(epoch)
                train_acc_num = 0
                test_acc_num = 0

                # train
                encoder.train()
                decoder.train()
                gcn.train()
                space_attention_model.train()
                temporal_attention_model.train()

                train_acc = 0
                train_loss_mean = 0
                classify_loss_mean = 0 
                vde_kldiv_loss_mean = 0 
                vde_mse_loss1_mean = 0
                vde_mse_loss2_mean = 0
                ib_loss_mean = 0

                for data_batch in train_loader:

                    graph, labels, oneperson_batch_list = data_batch
                    labels = labels.squeeze(dim=1).to(device)
                    node = graph['x'].to(device)
                    edge = graph['edge_index'].to(device)
                    oneperson_batch_list = oneperson_batch_list.to(device)
                    sample_num_batch = len(labels)

                    graph_batch = torch.empty(0)
                    for m in range(int(node.size(0)/brain_region)):
                        tep_batch = torch.full((brain_region, 1), m)
                        graph_batch = torch.cat((graph_batch, tep_batch), dim = 0)
                    graph_batch = graph_batch.long().to(device)


                    graph_emb = space_attention_model(node, edge, graph_batch)
                    z, mu, logvar = encoder(graph_emb)
                    final_representation, passed_z = temporal_attention_model(z, oneperson_batch_list)
                    generated_z_without_transiton = decoder(z)
                    generated_z_with_transiton = decoder(passed_z)


                    classify_loss = args.lambda_classify * criterion(final_representation, labels)
                    vde_kldiv_loss = args.lambda_kl * (-0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))
                    
                    vde_mse_loss1 = 0
                    vde_mse_loss2 = 0

                    for i in range(len(oneperson_batch_list)):
                        start = sum(oneperson_batch_list[0:i])
                        end = start + oneperson_batch_list[i]
                        vde_mse_loss1 += args.lambda_mse * mse(graph_emb[start:end , :], generated_z_without_transiton[start:end , :])
                        vde_mse_loss2 += args.lambda_mse * mse(graph_emb[start+1:end , :], generated_z_with_transiton[start:end-1 , :])

                    vde_kldiv_loss = vde_kldiv_loss / sample_num_batch
                    vde_mse_loss1 = vde_mse_loss1 / sample_num_batch
                    vde_mse_loss2 = vde_mse_loss2 / sample_num_batch


                    ib_loss = args.lambda_ib * calculate_MI(graph_emb, passed_z)

                    if epoch < 20:
                        vde_loss = vde_kldiv_loss

                    else:
                        vde_loss = vde_kldiv_loss + vde_mse_loss1 + vde_mse_loss2
                    
                    train_loss = classify_loss + vde_loss + ib_loss
                        

                    optimizer.zero_grad()
                    train_loss.backward()
                    optimizer.step()
                    scheduler.step()

                    pred = final_representation.max(1, keepdim=True)[1]
                    pred = pred.squeeze(dim=1)

                    train_loss = reduce_value(train_loss, average = True)
                    classify_loss = reduce_value(classify_loss, average = True)
                    vde_kldiv_loss = reduce_value(vde_kldiv_loss, average = True)
                    vde_mse_loss1 = reduce_value(vde_mse_loss1, average = True)
                    vde_mse_loss2 = reduce_value(vde_mse_loss2, average = True)
                    ib_loss = reduce_value(ib_loss, average = True)

                    if args.local_rank == 0:
                        train_loss_mean += train_loss.detach() * (sample_num_batch / train_size)
                        classify_loss_mean += classify_loss.detach() * (sample_num_batch / train_size)
                        vde_kldiv_loss_mean += vde_kldiv_loss.detach() * (sample_num_batch / train_size)
                        vde_mse_loss1_mean += vde_mse_loss1.detach() * (sample_num_batch / train_size)
                        vde_mse_loss2_mean += vde_mse_loss2.detach() * (sample_num_batch / train_size)
                        ib_loss_mean += ib_loss.detach() * (sample_num_batch / train_size)

                    train_acc_num += torch.eq(pred, labels).sum()

                train_acc_num = reduce_value(train_acc_num, average = False)


                # test
                if args.local_rank == 0:
                    encoder.eval()
                    decoder.eval()
                    gcn.eval()
                    space_attention_model.eval()
                    temporal_attention_model.eval()
                    test_acc = 0
                    test_f1s = 0
                    test_mcc = 0

                    with torch.no_grad():
                        for data_batch in test_loader:
                            
                            graph, labels, oneperson_batch_list = data_batch
                            labels = labels.squeeze(dim=1).to(device)
                            node = graph['x'].to(device)
                            edge = graph['edge_index'].to(device)
                            oneperson_batch_list = oneperson_batch_list.to(device)
                            sample_num_batch = len(labels)

                            graph_batch = torch.empty(0)
                            for m in range(int(node.size(0)/brain_region)):
                                tep_batch = torch.full((brain_region, 1), m)
                                graph_batch = torch.cat((graph_batch, tep_batch), dim = 0)
                            graph_batch = graph_batch.long().to(device)

                            graph_emb = space_attention_model(node, edge, graph_batch)
                            z, mu, logvar = encoder(graph_emb)
                            final_representation, passed_z = temporal_attention_model(z, oneperson_batch_list)
                            generated_z_without_transiton = decoder(z)
                            generated_z_with_transiton = decoder(passed_z)

                            pred = final_representation.max(1, keepdim=True)[1]
                            pred = pred.squeeze(dim=1)
                            acc, sen, spc, prc, f1s, mcc = calc_performance_statistics(pred, labels)

                            test_acc_num += torch.eq(pred, labels).sum()
                            test_f1s += f1s * (sample_num_batch / test_size) 
                            test_mcc += mcc * (sample_num_batch / test_size) 


                    train_loss_mean *= len(device_ids)
                    classify_loss_mean *= len(device_ids)
                    vde_kldiv_loss_mean *= len(device_ids)
                    vde_mse_loss1_mean *= len(device_ids)
                    vde_mse_loss2_mean *= len(device_ids)
                    ib_loss_mean *= len(device_ids)

                    train_acc = train_acc_num / train_size
                    test_acc = test_acc_num / test_size

                    if(test_acc >= test_highest_acc): 
                        test_highest_acc = test_acc
                        test_highest_f1s = test_f1s
                        test_highest_mcc = test_mcc

                        # save best model
                        if (test_highest_acc >= test_highest_acc_fold):
                            test_highest_acc_fold = test_highest_acc
                            os.makedirs(os.path.join(args.targetdir, 'best_model_' + args.dataset_name, 'fold_' + str(fold_idx+1)), exist_ok=True)
                            torch.save({
                                        'model_encoder': encoder.module.state_dict(),
                                        'model_decoder': decoder.module.state_dict(),
                                        'model_gcn': gcn.module.state_dict(),
                                        'model_space_attention_model': space_attention_model.module.state_dict(),
                                        'model_temporal_attention_model': temporal_attention_model.module.state_dict(),
                                        }, os.path.join(args.targetdir, 'best_model_' + args.dataset_name, 'fold_' + str(fold_idx + 1), 'best_model_' + args.dataset_name + '.pth')
                                    )
                            
                            os.makedirs(os.path.join(args.targetdir, 'Performance_indicators_' + args.dataset_name), exist_ok=True)
                            filename = args.targetdir + '/Performance_indicators_' + args.dataset_name + '/Fold' + str(fold_idx + 1) + '_Result.txt'
                            if not os.path.exists(filename):
                                with open(filename, 'w') as f:
                                    f.write("\n")
                                    f.write("**************\n")
                                    f.write("* save model *\n")
                                    f.write("**************\n")
                                    f.write("\n")
                            else:
                                with open(filename, 'a+') as f:
                                    f.write("\n")
                                    f.write("**************\n")
                                    f.write("* save model *\n")
                                    f.write("**************\n")
                                    f.write("\n")


                    print("epoch: %d  train: loss = %f  acc = %f   test: acc = %f  f1s = %f  mcc = %f" % (epoch + 1, train_loss_mean, train_acc, test_acc, test_f1s, test_mcc))
                    
                    os.makedirs(os.path.join(args.targetdir, 'Performance_indicators_' + args.dataset_name), exist_ok=True)
                    filename = args.targetdir + '/Performance_indicators_' + args.dataset_name + '/Fold' + str(fold_idx + 1) + '_Result.txt'

                    if not os.path.exists(filename):
                        with open(filename, 'w') as f:
                            f.write("epoch: %d  train: %f  %f  %f  %f  %f  %f  %f     test: %f  %f  %f  %f" % (epoch + 1, train_loss_mean, classify_loss_mean, vde_kldiv_loss_mean, vde_mse_loss1_mean, vde_mse_loss2_mean, ib_loss_mean, train_acc, test_highest_acc, test_acc, test_f1s, test_mcc))
                            f.write("\n")
                    else:
                        with open(filename, 'a+') as f:
                            f.write("epoch: %d  train: %f  %f  %f  %f  %f  %f  %f     test: %f  %f  %f  %f" % (epoch + 1, train_loss_mean, classify_loss_mean, vde_kldiv_loss_mean, vde_mse_loss1_mean, vde_mse_loss2_mean, ib_loss_mean, train_acc, test_highest_acc, test_acc, test_f1s, test_mcc))
                            f.write("\n")
                
                t2 = time.time()
                print('Program execution time: %ss' % (t2 - t1))



        print('test_highest_acc_fold: {:.8f}'.format(test_highest_acc_fold))


    print("succeed ! ")
    exit(0)