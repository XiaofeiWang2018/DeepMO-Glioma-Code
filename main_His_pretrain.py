from model import Mine
from utils import *
import dataset_mine
from net import init_weights,get_scheduler,WarmupCosineSchedule
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
    ############### Mine_model #######################
    Mine_model = Mine(opt).cuda(gpuID[0])
    device = torch.device('cuda:{}'.format(gpuID[0])) if gpuID else torch.device('cpu')
    init_weights(Mine_model, init_type='xavier', init_gain=1)
    Mine_model.to(device)
    for name, param in Mine_model.named_parameters():
        if not ("His" in name):
            param.requires_grad = False
    opt_His = torch.optim.Adam(filter(lambda p: p.requires_grad, Mine_model.parameters()), opt['Network']['lr'],weight_decay=0.00001)
    # Mine_model_sch_His = get_scheduler(opt_His, opt['n_ep'], opt['n_ep_decay'], opt['decayType'], -1) # the index but not the number of epoch, as last_ep-1

    Mine_model_sch_His = WarmupCosineSchedule(opt_His, warmup_steps=opt['decay_cos_warmup_steps'], t_total=opt['n_ep'])

    print('%d GPUs are working with the id of %s' % (torch.cuda.device_count(), str(gpuID)))


    ###############  Datasets #######################

    trainDataset = dataset_mine.Our_Dataset(phase='Train',opt=opt)
    valDataset = dataset_mine.Our_Dataset(phase='Val', opt=opt)
    testDataset = dataset_mine.Our_Dataset(phase='Test',opt=opt)
    trainLoader = DataLoader(trainDataset, batch_size=opt['batchSize'],
                             num_workers=opt['nThreads'] if (sysstr == "Linux") else 1, shuffle=True)
    valLoader = DataLoader(valDataset, batch_size=opt['Val_batchSize'],
                             num_workers=opt['nThreads'] if (sysstr == "Linux") else 1, shuffle=False)
    testLoader = DataLoader(testDataset, batch_size=opt['Test_batchSize'],
                            num_workers=opt['nThreads'] if (sysstr == "Linux") else 1, shuffle=False)

    ############## initialize #######################

    last_ep = 0
    total_it = 0

    ckptdir = os.path.join('./pretrain/', 'IDH_model-0060.pth')
    checkpoint = torch.load(ckptdir)['network']
    related_params = {k: v for k, v in checkpoint.items() if ('IDH' in k or 'position' in k or 'init' in k)}
    Mine_model.load_state_dict(related_params,strict=False)
    print('Finetune the Mine_model from:%s' % ckptdir)
    saver = Saver(opt)
    print('%d epochs and %d iterations has been trained' % (last_ep, total_it))
    alleps = opt['n_ep'] - last_ep
    allit = len(trainLoader)


    ############# begin training ##########################
    for epoch in range(alleps):
        curep = last_ep + epoch
        lossdict = {'train/cls': 0,  'train/all': 0}
        count=0
        running_results = {'acc1': 0,'acc2': 0,'allloss': 0, 'cls_loss': 0}
        train_bar = tqdm(trainLoader)
        for packs in train_bar:
            img = packs[0] ##(BS,N,1024)
            label = packs[1]
            count+=1
            if  torch.cuda.is_available():
                img = img.cuda(gpuID[0])
                label = label.cuda(gpuID[0])
            label_His=label[:,3]
            Mine_model.train()
            Mine_model.zero_grad()
            results_dict = Mine_model(img)
            pred=results_dict['logits'][3] #[BS,4]

            loss_all = Mine_model.calculateLoss_His(pred,label_His)
            loss_all.backward()
            opt_His.step()

            _, predicted = torch.max(pred[:,1:].data, 1)
            _, predicted_2 = torch.max(pred.data, 1)
            total = label_His.size(0)
            ###  first kind of acc
            label_His_numpy=label_His.detach().cpu().numpy() #[BS] AO A  O GBM//0 1 2 3
            predicted_numpy=predicted.detach().cpu().numpy() #[BS] A  O GBM //0 1 2
            correct=0
            for k in range(predicted_numpy.shape[0]):
                if label_His_numpy[k]==0 and (predicted_numpy[k]==0 or predicted_numpy[k]==1):
                    correct+=1
                if label_His_numpy[k] == 1 and predicted_numpy[k]==0:
                    correct += 1
                if label_His_numpy[k] == 2 and predicted_numpy[k]==1:
                    correct += 1
                if label_His_numpy[k] == 3 and predicted_numpy[k]==2:
                    correct += 1
            ###  second kind of acc
            correct_2 = predicted_2.eq(label_His.data).cpu().sum()

            running_results['acc1'] += 100. * correct / total
            running_results['acc2'] += 100. * correct_2 / total
            running_results['allloss'] += loss_all.item()
            running_results['cls_loss'] += Mine_model.loss_His.item()
            total_it = total_it + 1
            lossdict['train/cls'] += Mine_model.loss_His.item()
            lossdict['train/all'] += loss_all.item()
            train_bar.set_description(
                desc=opt['name'] + ' [%d/%d] allloss: %.4f | Acc1: %.4f| Acc2: %.4f' % (
                    epoch, alleps,
                    running_results['allloss'] / count,
                    running_results['acc1'] / count,
                    running_results['acc2'] / count,
                ))
        lossdict['train/cls'] = lossdict['train/cls'] / count
        lossdict['train/all'] = lossdict['train/all'] / count
        Mine_model_sch_His.step()
        print(' Training Acc1 :%.4f |  Acc2 :%.4f' % (running_results['acc1'] / count,running_results['acc2'] / count))
        saver.write_scalars(curep, lossdict)
        saver.write_log(curep, lossdict, 'traininglossLog')

        print('-------------------------------------Val and Test--------------------------------------')
        if (curep + 1) % opt['n_ep_save'] == 0:
            if (curep + 1) > (alleps / 2):
                save_dir = os.path.join(opt['modelDir'], 'Mine_model-%04d.pth' % (curep + 1))
                state = {
                    'network': Mine_model.state_dict(),
                    'optimizer': opt_His.state_dict(),
                    'ep': curep + 1,
                    'total_it': total_it
                }
                torch.save(state, save_dir)
            print("----------Val-------------")
            list_WSI = validation_His(opt, Mine_model, None, valLoader, saver, curep, opt['eva_cm'], gpuID)
            if opt['eva_cm']:
                saver.write_cm_maps(curep, list_WSI[1], opt['dataLabels'], savename='cm_WSI.png')
            print('validation in epoch: %d/%d, acc_WSI:%.3f,sen_WSI:%.3f,spec_WSI:%.3f, auc_WSI:%.3f' % (
                epoch + 1, alleps, list_WSI[7],list_WSI[3],list_WSI[4],list_WSI[5]))
            val_dict = {'val/acc3_WSI': list_WSI[7], 'val/sen_WSI': list_WSI[3], 'val/spec_WSI': list_WSI[4],
                        'val/auc_WSI': list_WSI[5],'val/f1_WSI': list_WSI[2], 'val/prec_WSI': list_WSI[6]}
            saver.write_scalars(curep, val_dict)
            saver.write_log(curep, val_dict, 'validationLog')

            print("----------Test-------------")
            list_WSI = validation_His(opt, Mine_model, None, testLoader, saver, curep, opt['eva_cm'], gpuID)

            if opt['eva_cm']:
                saver.write_cm_maps(curep, list_WSI[1], opt['dataLabels'], savename='cm_WSI.png')
            print('Test in epoch: %d/%d, acc_WSI:%.3f,sen_WSI:%.3f,spec_WSI:%.3f, auc_WSI:%.3f' % (
                epoch + 1, alleps, list_WSI[7], list_WSI[3], list_WSI[4], list_WSI[5]))
            test_dict = {'test/acc3_WSI': list_WSI[7], 'test/sen_WSI': list_WSI[3], 'test/spec_WSI': list_WSI[4],
                        'test/auc_WSI': list_WSI[5], 'test/f1_WSI': list_WSI[2], 'test/prec_WSI': list_WSI[6] }
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
    parser.add_argument('--opt', type=str, default='config/mine.yml')
    args = parser.parse_args()
    with open(args.opt) as f:
        opt = yaml.load(f, Loader=SafeLoader)


    sysstr = platform.system()


    # setup_seed(opt['seed'])
    if opt['command']=='Train':
        cur_time = time.strftime('%m%d-%H%M', time.localtime())

        # remove_all_dir(opt['logDir'])
        # remove_all_dir(opt['modelDir'])
        # remove_all_dir(opt['saveDir'])
        # remove_all_dir(opt['cm_saveDir'])
        opt['name'] = opt['name']+'_{}'.format(cur_time)
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




    a=1











