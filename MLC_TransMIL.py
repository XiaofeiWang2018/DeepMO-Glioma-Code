import os
import torch
import torch.nn as nn
# import curve
import platform
import dataset
from torch.utils.data import DataLoader,Dataset
import numpy as np
import argparse, time, random
import yaml
from yaml.loader import SafeLoader
from evaluation import *
import glob
from compared_model.TransMIL import TransMIL as transMIL
import model
import net
from saver import Saver
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import roc_curve, auc
from tensorboardX import SummaryWriter
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from utils import *
from sklearn import metrics
"""
label 2016={ 0:'G2_O', 1:'G3_O', 2:'G2_A', 3:'G3_A', 4:'G2_OA', 5:'G3_OA', 6:'GBM'}
label 2021={ 0:'G2_O', 1:'G3_O', 2:'G2_A', 3:'G3_A', 4:'G4_A', 5:'GBM'}
"""
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if seed == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train(opt):

    gpuID = opt['gpus']
    ############### Model #######################
    TransMIL = transMIL(opt)
    assert opt['name'].split('_')[0]=='TransMIL'
    assert opt['name'].split('_')[1] == 'MLC'
    # print('# network parameters:', sum(param.numel() for param in TransMIL.parameters()) / 1e6, 'M')

    ############## Init #####################################


    device = torch.device('cuda:{}'.format(gpuID[0])) if gpuID else torch.device('cpu')

    # TransMIL= model.init_weights(TransMIL, init_type='xavier', init_gain=1)
    # Res_pretrain= net.Res50_pretrain()
    # Res_pretrain.to(device)
    # Res_pretrain = net.Res18_pretrain()
    # Res_pretrain.to(device)
    TransMIL.to(device)
    TransMIL_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, TransMIL.parameters()), opt['Network']['TransMIL']['lr'], weight_decay=0.00001)
    task = opt['name'].split('_')[1]
    ############## Flops #####################################
    # tensor_in = torch.randn( 1200,3,224, 224)  # only used for calculating the flops
    # if torch.cuda.is_available():
    #     tensor_in = tensor_in.cuda(gpuID[0])
    # from thop import profile
    # flops, params = profile(TransMIL, inputs=(tensor_in,))
    # print('#################################################')
    # print('Model name %s: ;flops: %.4fG;Parameters: %.4fM' % (opt['name'], flops / 1e9, params / 1e6))
    # print('#################################################')

    print('%d GPUs are working with the id of %s' % (torch.cuda.device_count(), str(gpuID)))


    ###############  Datasets #######################

    trainDataset = dataset.Our_Dataset(phase='Train',opt=opt)
    valDataset = dataset.Our_Dataset(phase='Val', opt=opt)
    testDataset = dataset.Our_Dataset(phase='Test',opt=opt)
    trainLoader = DataLoader(trainDataset, batch_size=opt['batchSize'],
                             num_workers=opt['nThreads'] if (sysstr == "Linux") else 1, shuffle=True)
    valLoader = DataLoader(valDataset, batch_size=opt['Val_batchSize'],
                             num_workers=opt['nThreads'] if (sysstr == "Linux") else 1, shuffle=False)
    testLoader = DataLoader(testDataset, batch_size=opt['Test_batchSize'],
                            num_workers=opt['nThreads'] if (sysstr == "Linux") else 1, shuffle=False)

    ############## initialize #######################

    last_ep = 0
    total_it = 0

    if opt['resume_epoch']: # if resume the pre-trained model
        if opt['resume_epoch'] == -1:  # resume the last model
            modellist = glob.glob(os.path.join(opt['modelDir'], 'model-*.pth'))
            modellist.sort(key=lambda x:int(x.split('model-')[1].split('.pth')[0]), reverse=True)
            ckptdir = modellist[0]
            epnum = int(os.path.basename(ckptdir).split('-')[-1].split('.')[0])
            checkpoint = torch.load(ckptdir)
            last_ep = checkpoint['ep'] #  epoch has been trained, e.g., 1 for tained for 1 full epoch
            total_it = checkpoint['total_it']
            TransMIL.load_state_dict(checkpoint['network'])
            TransMIL_opt.load_state_dict(checkpoint['optimizer'])
            assert last_ep == epnum #
        else:# 10
            ckptdir = os.path.join(opt['modelDir'], 'model-%04d.pth' % opt['resume_epoch'])
            epnum = opt['resume_epoch']
            checkpoint = torch.load(ckptdir)
            last_ep = checkpoint['ep']
            total_it = checkpoint['total_it']
            TransMIL.load_state_dict(checkpoint['network'])
            TransMIL_opt.load_state_dict(checkpoint['optimizer'])
            assert last_ep == epnum
        print('Finetune the model from:%s'%ckptdir)
    TransMIL_sch = model.get_scheduler(TransMIL_opt, opt['n_ep'], opt['n_ep_decay'], opt['decayType'], last_ep-1) # the index but not the number of epoch, as last_ep-1
    saver = Saver(opt)
    print('%d epochs and %d iterations has been trained' % (last_ep, total_it))
    alleps = opt['n_ep'] - last_ep
    allit = len(trainLoader)


    ############# begin training ##########################
    for epoch in range(alleps):
        curep = last_ep + epoch
        lossdict = {'train/CE': 0, 'train/Trip': 0, 'train/all': 0, 'train/IDH': 0, 'train/1p19q': 0, 'train/CDKN': 0, 'train/Diag': 0}
        count=0
        running_results = {'acc_IDH': 0,'acc_1p19q': 0,'acc_CDKN': 0,'acc_Diag': 0, 'acc_loss': 0}
        train_bar = tqdm(trainLoader)
        for packs in train_bar:
            img =packs[0][0]
            label = packs[1]
            imgPath = packs[2]
            count+=1

            if torch.cuda.is_available():
                img = img.cuda(gpuID[0])
                label = label.cuda(gpuID[0])
            TransMIL.train()
            # Res_pretrain.train()
            TransMIL.zero_grad()
            # img=Res_pretrain(img)
            results_dict = TransMIL(img)
            pred=results_dict['logits']
            # print(pred[3].shape)
            # print(label[:,3].shape)
            loss_all = TransMIL.calculateLoss(pred,label)
            loss_all.backward()
            TransMIL_opt.step()

            _, predicted_IDH = torch.max(pred[0].data, 1)
            total_IDH = label[:,0].size(0)
            correct_IDH = predicted_IDH.eq(label[:,0].data).cpu().sum()
            running_results['acc_IDH'] += 100. * correct_IDH / total_IDH

            _, predicted_1p19q = torch.max(pred[1].data, 1)
            total_1p19q = label[:, 1].size(0)
            correct_1p19q = predicted_1p19q.eq(label[:, 1].data).cpu().sum()
            running_results['acc_1p19q'] += 100. * correct_1p19q / total_1p19q

            _, predicted_CDKN = torch.max(pred[2].data, 1)
            total_CDKN = label[:, 2].size(0)
            correct_CDKN = predicted_CDKN.eq(label[:, 2].data).cpu().sum()
            running_results['acc_CDKN'] += 100. * correct_CDKN / total_CDKN

            _, predicted_Diag = torch.max(pred[3].data, 1)
            total_Diag = label[:, 3].size(0)
            correct_Diag = predicted_Diag.eq(label[:, 3].data).cpu().sum()
            running_results['acc_Diag'] += 100. * correct_Diag / total_Diag

            running_results['acc_loss'] += loss_all.item()
            total_it = total_it + 1
            lossdict['train/all'] += loss_all.item()
            lossdict['train/IDH'] += TransMIL.loss_ce_IDH.item()
            lossdict['train/1p19q'] += TransMIL.loss_ce_1p19q.item()
            lossdict['train/CDKN'] += TransMIL.loss_ce_CDKN.item()
            lossdict['train/Diag'] += TransMIL.loss_ce_Diag.item()
            train_bar.set_description(
                desc=opt['name'] + ' [%d/%d] loss: %.4f  | acc_IDH: %.4f | acc_1p19q: %.4f | acc_CDKN: %.4f | acc_Diag: %.4f' % (
                    epoch, alleps,
                    running_results['acc_loss'] / count,
                    running_results['acc_IDH'] / count,
                    running_results['acc_1p19q'] / count,
                    running_results['acc_CDKN'] / count,
                    running_results['acc_Diag'] / count
                ))
        lossdict['train/IDH'] = lossdict['train/IDH'] / allit
        lossdict['train/1p19q'] = lossdict['train/1p19q'] / allit
        lossdict['train/CDKN'] = lossdict['train/CDKN'] / allit
        lossdict['train/Diag'] = lossdict['train/Diag'] / allit
        lossdict['train/all'] = lossdict['train/all'] / allit
        TransMIL_sch.step()
        print('Training of %d interations, overall loss:%.6f, CE:%.6f' % (total_it, lossdict['train/all'], lossdict['train/CE']))
        saver.write_scalars(curep, lossdict)
        saver.write_log(curep, lossdict, 'traininglossLog')

        print('-------------------------------------Val and Test--------------------------------------')
        if (curep + 1) % opt['n_ep_save'] == 0:
            if (curep + 1) > (alleps / 2):
                save_dir = os.path.join(opt['modelDir'], 'model-%04d.pth' % (curep + 1))
                state = {
                    'network': TransMIL.state_dict(),
                    'optimizer': TransMIL_opt.state_dict(),
                    'ep': curep + 1,
                    'total_it': total_it
                }
                torch.save(state, save_dir)
            print("----------Val-------------")
            list_WSI_IDH,list_WSI_1p19q,list_WSI_CDKN,list_WSI_Diag= validation_MCL(opt,TransMIL,None, valLoader, saver, curep, opt['eva_cm'],gpuID)
            if opt['eva_cm']:
                saver.write_cm_maps(curep, list_WSI_IDH[1], opt['dataLabels'], savename='cm_WSI_IDH.png')
                saver.write_cm_maps(curep, list_WSI_1p19q[1], opt['dataLabels'], savename='cm_WSI_1p19q.png')
                saver.write_cm_maps(curep, list_WSI_CDKN[1], opt['dataLabels'], savename='cm_WSI_CDKN.png')
                saver.write_cm_maps(curep, list_WSI_Diag[1], opt['dataLabels'], savename='cm_WSI_Diag.png')
            print('validation in epoch: %d/%d, acc_IDH:%.3f,acc_1p19q:%.3f,acc_CDKN:%.3f, acc_Diag:%.3f' % (
                epoch + 1, alleps, list_WSI_IDH[0],list_WSI_1p19q[0],list_WSI_CDKN[0],list_WSI_Diag[0]))
            val_dict = {'val/acc_IDH': list_WSI_IDH[0], 'val/sen_IDH': list_WSI_IDH[3], 'val/spec_IDH': list_WSI_IDH[4],
                        'val/auc_IDH': list_WSI_IDH[5],'val/f1_IDH': list_WSI_IDH[2], 'val/prec_IDH': list_WSI_IDH[6],
                        'val/acc_1p19q': list_WSI_1p19q[0], 'val/sen_1p19q': list_WSI_1p19q[3], 'val/spec_1p19q': list_WSI_1p19q[4],
                        'val/auc_1p19q': list_WSI_1p19q[5], 'val/f1_1p19q': list_WSI_1p19q[2], 'val/prec_1p19q': list_WSI_1p19q[6],
                        'val/acc_CDKN': list_WSI_CDKN[0], 'val/sen_CDKN': list_WSI_CDKN[3], 'val/spec_CDKN': list_WSI_CDKN[4],
                        'val/auc_CDKN': list_WSI_CDKN[5], 'val/f1_CDKN': list_WSI_CDKN[2], 'val/prec_CDKN': list_WSI_CDKN[6],
                        'val/acc_Diag': list_WSI_Diag[0], 'val/sen_Diag': list_WSI_Diag[3], 'val/spec_Diag': list_WSI_Diag[4],
                        'val/auc_Diag': list_WSI_Diag[5], 'val/f1_Diag': list_WSI_Diag[2], 'val/prec_Diag': list_WSI_Diag[6],
                        }
            saver.write_scalars(curep, val_dict)
            saver.write_log(curep, val_dict, 'validationLog')

            print("----------Test-------------")
            list_WSI_IDH,list_WSI_1p19q,list_WSI_CDKN,list_WSI_Diag = validation_MCL(opt, TransMIL, None, testLoader, saver, curep, opt['eva_cm'], gpuID)
            if opt['eva_cm']:
                saver.write_cm_maps(curep, list_WSI_IDH[1], opt['dataLabels'], savename='cm_WSI_IDH.png')
                saver.write_cm_maps(curep, list_WSI_1p19q[1], opt['dataLabels'], savename='cm_WSI_1p19q.png')
                saver.write_cm_maps(curep, list_WSI_CDKN[1], opt['dataLabels'], savename='cm_WSI_CDKN.png')
                saver.write_cm_maps(curep, list_WSI_Diag[1], opt['dataLabels'], savename='cm_WSI_Diag.png')
            print('test in epoch: %d/%d, acc_IDH:%.3f,acc_1p19q:%.3f,acc_CDKN:%.3f, acc_Diag:%.3f' % (
                epoch + 1, alleps, list_WSI_IDH[0], list_WSI_1p19q[0], list_WSI_CDKN[0], list_WSI_Diag[0]))
            test_dict = {'test/acc_IDH': list_WSI_IDH[0], 'test/sen_IDH': list_WSI_IDH[3], 'test/spec_IDH': list_WSI_IDH[4],
                        'test/auc_IDH': list_WSI_IDH[5],'test/f1_IDH': list_WSI_IDH[2], 'test/prec_IDH': list_WSI_IDH[6],
                        'test/acc_1p19q': list_WSI_1p19q[0], 'test/sen_1p19q': list_WSI_1p19q[3], 'test/spec_1p19q': list_WSI_1p19q[4],
                        'test/auc_1p19q': list_WSI_1p19q[5], 'test/f1_1p19q': list_WSI_1p19q[2], 'test/prec_1p19q': list_WSI_1p19q[6],
                        'test/acc_CDKN': list_WSI_CDKN[0], 'test/sen_CDKN': list_WSI_CDKN[3], 'test/spec_CDKN': list_WSI_CDKN[4],
                        'test/auc_CDKN': list_WSI_CDKN[5], 'test/f1_CDKN': list_WSI_CDKN[2], 'test/prec_CDKN': list_WSI_CDKN[6],
                        'test/acc_Diag': list_WSI_Diag[0], 'test/sen_Diag': list_WSI_Diag[3], 'test/spec_Diag': list_WSI_Diag[4],
                        'test/auc_Diag': list_WSI_Diag[5], 'test/f1_Diag': list_WSI_Diag[2], 'test/prec_Diag': list_WSI_Diag[6], }
            saver.write_scalars(curep, test_dict)
            saver.write_log(curep, test_dict, 'testLog')












def remove_all_file(path):
    if os.path.isdir(path):
        for i in os.listdir(path):
            path_file = os.path.join(path, i)
            os.remove(path_file)
def remove_all_dir(path):
    if os.path.isdir(path):
        for i in os.listdir(path):
            path_file = os.path.join(path, i)
            for j in os.listdir(path_file):
                path_file1 = os.path.join(path_file, j)
                os.remove(path_file1)
            os.rmdir(path_file)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='config/miccai.yml')
    args = parser.parse_args()
    with open(args.opt) as f:
        opt = yaml.load(f, Loader=SafeLoader)


    if not os.path.exists(opt['logDir']):
        os.makedirs(opt['logDir'])
    if not os.path.exists(opt['modelDir']):
        os.makedirs(opt['modelDir'])
    if not os.path.exists(opt['saveDir']):
        os.makedirs(opt['saveDir'])

    sysstr = platform.system()


    setup_seed(opt['seed'])
    if opt['command']=='Train':
        cur_time = time.strftime('%m%d-%H%M', time.localtime())


        opt['name'] = opt['name']+'_{}'.format(cur_time)
        # remove_all_dir(opt['logDir'])
        # remove_all_dir(opt['modelDir'])
        # remove_all_dir(opt['saveDir'])
        # remove_all_dir(opt['cm_saveDir'])
        opt['logDir'] = os.path.join(opt['logDir'], opt['name'])
        opt['modelDir'] = os.path.join(opt['modelDir'], opt['name'])
        opt['saveDir'] = os.path.join(opt['saveDir'], opt['name'])
        opt['cm_saveDir'] = os.path.join(opt['cm_saveDir'], opt['name'])
        if not os.path.exists(opt['logDir']):
            os.makedirs(opt['logDir'])
        if not os.path.exists(opt['modelDir']):
            os.makedirs(opt['modelDir'])
        if not os.path.exists(opt['saveDir']):
            os.makedirs(opt['saveDir'])
        if not os.path.exists(opt['cm_saveDir']):
            os.makedirs(opt['cm_saveDir'])




        para_log = os.path.join(opt['modelDir'], 'params.yml')
        if os.path.exists(para_log):
            os.remove(para_log)
        with open(para_log, 'w') as f:
            data = yaml.dump(opt, f, sort_keys=False, default_flow_style=False)

        print("\n\n============> begin training <=======")
        train(opt)
        if opt['command'] == "train_test":
            print("\n\n============> begin train_test <=======")
            test(opt)


    elif opt['command'] == "test":

        opt['modelDir'] = os.path.join(opt['modelDir'], opt['name'])
        opt['cm_saveDir'] = os.path.join(opt['cm_saveDir'], opt['name'])
        if not os.path.exists(opt['cm_saveDir']):
            os.makedirs(opt['cm_saveDir'])
        test(opt)

    a=1





























