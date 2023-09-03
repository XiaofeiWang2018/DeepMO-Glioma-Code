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
import gc
from sklearn import metrics
from compared_model.cifar_CLAM import CLAM_SB as clam
from compared_model.cifar_CLAM import embeddings as embeddings
from utils import *
from torchvision import transforms, datasets
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
    CLAM = clam(opt)
    Embeddings = embeddings(opt)
    ############## Init #####################################


    device = torch.device('cuda:{}'.format(gpuID[0])) if gpuID else torch.device('cpu')

    model.init_weights(CLAM, init_type='xavier', init_gain=1)
    model.init_weights(Embeddings, init_type='xavier', init_gain=1)
    # Res_pretrain= net.Res50_pretrain()
    # Res_pretrain.to(device)

    CLAM.to(device)
    Embeddings.to(device)
    CLAM_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, CLAM.parameters()), opt['Network']['CLAM']['lr'], weight_decay=0.00001)
    Embeddings_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, Embeddings.parameters()), opt['Network']['CLAM']['lr'], weight_decay=0.00001)

    ############## Flops #####################################

    print('%d GPUs are working with the id of %s' % (torch.cuda.device_count(), str(gpuID)))


    ###############  Datasets #######################
    transform_test = transforms.Compose([
        transforms.Resize((800, 800)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    trainset = datasets.CIFAR10(root="/mnt/disk10T/fuyibing/WSI_pro/others/ViT/data",
                                train=True,
                                download=True,
                                transform=transform_test)

    trainLoader = DataLoader(trainset, batch_size=1,num_workers=opt['nThreads'] if (sysstr == "Linux") else 1, shuffle=True)
    ############## initialize #######################

    last_ep = 0
    total_it = 0


    CLAM_sch = model.get_scheduler(CLAM_opt, opt['n_ep'], opt['n_ep_decay'], opt['decayType'], last_ep-1) # the index but not the number of epoch, as last_ep-1
    Embeddings_sch = model.get_scheduler(Embeddings_opt, opt['n_ep'], opt['n_ep_decay'], opt['decayType'], last_ep-1) # the index but not the number of epoch, as last_ep-1
    saver = Saver(opt)
    print('%d epochs and %d iterations has been trained' % (last_ep, total_it))
    alleps = opt['n_ep'] - last_ep
    allit = len(trainLoader)


    ############# begin training ##########################
    for epoch in range(alleps):
        curep = last_ep + epoch
        lossdict = {'train/bag': 0,'train/ins': 0,  'train/all': 0}
        count=0
        running_results = {'acc': 0,'sen': 0,'spec': 0, 'allloss': 0, 'bagloss': 0, 'insloss': 0}
        train_bar = tqdm(trainLoader)

        for step, batch in enumerate(train_bar):
            img, label = batch
            # img = packs[0][0]
            # label = packs[1]
            count+=1

            if torch.cuda.is_available():
                img = img.cuda(gpuID[0])
                label = label.cuda(gpuID[0])

            CLAM.train()
            Embeddings.train()
            CLAM.zero_grad()
            Embeddings.zero_grad()
            img=Embeddings(img)
            results_dict = CLAM(img[0],label[0])
            pred=results_dict['logits']

            loss_all = CLAM.calculateLoss(pred,label)
            loss_all.backward()

            Embeddings_opt.step()
            CLAM_opt.step()

            _, predicted = torch.max(pred.data, 1)
            total = label.size(0)
            correct = predicted.eq(label.data).cpu().sum()

            pred0 = predicted.tolist()
            gt = label.tolist()


            running_results['acc'] += 100. * correct / total
            running_results['allloss'] += loss_all.item()
            running_results['bagloss'] += CLAM.loss_ce.item()
            running_results['insloss'] += CLAM.total_inst_loss.item()
            total_it = total_it + 1
            lossdict['train/bag'] += CLAM.loss_ce.item()
            lossdict['train/ins'] += CLAM.total_inst_loss.item()
            lossdict['train/all'] += loss_all.item()
            train_bar.set_description(
                desc=opt['name'] + ' [%d/%d] allloss: %.4f | bagloss: %.4f insloss: %.4f | Acc: %.4f' % (
                    epoch, alleps,
                    running_results['allloss'] / count,
                    running_results['bagloss'] / count,
                    running_results['insloss'] / count,
                    running_results['acc'] / count,

                ))
        lossdict['train/bag'] = lossdict['train/bag'] / allit
        lossdict['train/ins'] = lossdict['train/ins'] / allit
        lossdict['train/all'] = lossdict['train/all'] / allit
        CLAM_sch.step()
        Embeddings_sch.step()

        print(' Training Acc :%.4f' % (running_results['acc'] / count))
        saver.write_scalars(curep, lossdict)
        saver.write_log(curep, lossdict, 'traininglossLog')

        print('-------------------------------------Val and Test--------------------------------------')
        if (curep + 1) % opt['n_ep_save'] == 0:

            save_dir = os.path.join(opt['modelDir'], 'model-%04d.pth' % (curep + 1))
            state = {
                'network': CLAM.state_dict(),
                'optimizer': CLAM_opt.state_dict(),
                'ep': curep + 1,
                'total_it': total_it
            }
            torch.save(state, save_dir)







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
    parser.add_argument('--opt', type=str, default='config/cifar_pretrain.yml')
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





























