# from apex import amp
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
    Mine_model_init,Mine_model_IDH,Mine_model_1p19q,Mine_model_CDKN,Mine_model_Graph,Mine_model_His,Mine_model_Cls\
        ,opt_init,opt_IDH,opt_1p19q,opt_CDKN,opt_Graph,opt_His,opt_Cls=get_model(opt)


    # Mine_model_sch_IDH = get_scheduler(opt_IDH, opt['n_ep'], opt['n_ep_decay'], opt['decayType'], -1) # the index but not the number of epoch, as last_ep-1
    Mine_model_sch_init = WarmupCosineSchedule(opt_init, warmup_steps=opt['decay_cos_warmup_steps'], t_total=opt['n_ep'])
    Mine_model_sch_IDH = WarmupCosineSchedule(opt_IDH, warmup_steps=opt['decay_cos_warmup_steps'], t_total=opt['n_ep'])
    Mine_model_sch_1p19q = WarmupCosineSchedule(opt_1p19q, warmup_steps=opt['decay_cos_warmup_steps'], t_total=opt['n_ep'])
    Mine_model_sch_CDKN = WarmupCosineSchedule(opt_CDKN, warmup_steps=opt['decay_cos_warmup_steps'], t_total=opt['n_ep'])
    Mine_model_sch_Graph = WarmupCosineSchedule(opt_Graph, warmup_steps=opt['decay_cos_warmup_steps'],t_total=opt['n_ep'])
    Mine_model_sch_His = WarmupCosineSchedule(opt_His, warmup_steps=opt['decay_cos_warmup_steps'], t_total=opt['n_ep'])
    Mine_model_sch_Cls = WarmupCosineSchedule(opt_Cls, warmup_steps=opt['decay_cos_warmup_steps'],t_total=opt['n_ep'])

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

    # Init
    ckptdir = os.path.join('./models/Mine_dim2500_Stage1_0226-0103/', 'Mine_model-0065.pth')
    checkpoint = torch.load(ckptdir)
    related_params = {k: v for k, v in checkpoint['init'].items() if ('position' in k or 'init' in k)}
    Mine_model_init.load_state_dict(related_params,strict=False)
    print('Finetune the Mine_model from:%s'%ckptdir)
    # # IDH
    # ckptdir = os.path.join('./pretrain/', 'IDH_model-0060.pth')
    # checkpoint = torch.load(ckptdir)
    # related_params_IDH = {k: v for k, v in checkpoint['network'].items() if 'IDH' in k }
    # related_params.update(related_params_IDH)
    # # 1p19q
    # ckptdir = os.path.join('./pretrain/', 'xx.pth')
    # checkpoint = torch.load(ckptdir)
    # related_params_1p19q = {k: v for k, v in checkpoint['network'].items() if '1p19q' in k }
    # related_params.update(related_params_1p19q)
    # # CDKN
    # ckptdir = os.path.join('./pretrain/', 'xx.pth')
    # checkpoint = torch.load(ckptdir)
    # related_params_CDKN = {k: v for k, v in checkpoint['network'].items() if 'CDKN' in k}
    # related_params.update(related_params_CDKN)
    # # His
    ckptdir = os.path.join('./models/Mine_dim2500_Stage1_0226-0103/', 'Mine_model-0065.pth')
    checkpoint = torch.load(ckptdir)
    related_params = {k: v for k, v in checkpoint['His'].items() }
    Mine_model_His.load_state_dict(related_params, strict=False)
    # # Cls
    ckptdir = os.path.join('./models/Mine_dim2500_Stage1_0226-0103/', 'Mine_model-0065.pth')
    checkpoint = torch.load(ckptdir)
    related_params = {k: v for k, v in checkpoint['Cls'].items()}
    Mine_model_Cls.load_state_dict(related_params, strict=False)





    saver = Saver(opt)
    print('%d epochs and %d iterations has been trained' % (last_ep, total_it))
    alleps = opt['n_ep'] - last_ep
    allit = len(trainLoader)


    ############# begin training ##########################
    for epoch in range(alleps):
        Mine_model_sch_init.step()
        Mine_model_sch_IDH.step()
        Mine_model_sch_1p19q.step()
        Mine_model_sch_CDKN.step()
        Mine_model_sch_Graph.step()
        # Mine_model_sch_His.step()
        # Mine_model_sch_Cls.step()
        Mine_model_init.train()
        Mine_model_IDH.train()
        Mine_model_1p19q.train()
        Mine_model_CDKN.train()
        Mine_model_Graph.train()
        # Mine_model_His.train()
        # Mine_model_Cls.train()
        curep = last_ep + epoch
        lossdict = {'train/init_subnet': 0,'train/IDH_subnet': 0,  'train/1p19q_subnet': 0,'train/CDKN_subnet': 0,'train/Graph_subnet': 0,  'train/His_subnet': 0,'train/Grade_subnet': 0,'train/Cls_subnet': 0}
        count=0
        running_results = {'acc_IDH': 0,'acc_1p19q': 0,'acc_CDKN': 0,'acc_His': 0,'acc_Diag': 0,
                           'loss_IDH': 0,'loss_1p19q': 0,'loss_CDKN': 0,'loss_His': 0,'loss_Diag': 0}
        train_bar = tqdm(trainLoader)
        for packs in train_bar:
            img = packs[0] ##(BS,N,1024)
            label = packs[1]
            count+=1
            if  torch.cuda.is_available():
                img = img.cuda(gpuID[0])
                label = label.cuda(gpuID[0])
            label_IDH=label[:,0]
            label_1p19q = label[:, 1]
            label_CDKN = label[:, 2]
            label_His = label[:, 3]
            label_His_2class= label[:, 6]
            ### ### forward IDH
            init_feature=Mine_model_init(img) # (BS,2500,1024)

            hidden_states, encoded_IDH = Mine_model_IDH(init_feature)
            hidden_states, encoded_1p19q = Mine_model_1p19q(hidden_states)
            encoded_CDKN = Mine_model_CDKN(hidden_states)
            results_dict,weight_IDH_wt,weight_1p19q_codel,encoded_IDH0,encoded_1p19q0,encoded_CDKN0 = Mine_model_Graph(encoded_IDH, encoded_1p19q, encoded_CDKN)

            pred_IDH=results_dict['logits_IDH']
            pred_1p19q = results_dict['logits_1p19q']
            pred_CDKN = results_dict['logits_CDKN']

            ### ### backward IDH
            Mine_model_CDKN.zero_grad()
            Mine_model_Graph.zero_grad()
            Mine_model_1p19q.zero_grad()
            Mine_model_IDH.zero_grad()
            Mine_model_init.zero_grad()
            loss_1p19q = Mine_model_Graph.module.calculateLoss_1p19q(pred_1p19q, label_1p19q)
            loss_CDKN = Mine_model_Graph.module.calculateLoss_CDKN(pred_CDKN, label_CDKN)
            loss_IDH = Mine_model_Graph.module.calculateLoss_IDH(pred_IDH, label_IDH)
            # loss_Graph = Mine_model_Graph.module.calculateLoss_Graph(encoded_IDH0,encoded_1p19q0,encoded_CDKN0)

            loss_IDH_subnet = loss_IDH + 0.3*loss_1p19q + 0.6*loss_CDKN
            loss_IDH_subnet.backward()
            opt_init.step()
            opt_IDH.step()
            opt_1p19q.step()
            opt_CDKN.step()
            opt_Graph.step()

            _, predicted_IDH = torch.max(pred_IDH.data, 1)
            total_IDH = label_IDH.size(0)
            correct_IDH = predicted_IDH.eq(label_IDH.data).cpu().sum()
            _, predicted_1p19q = torch.max(pred_1p19q.data, 1)
            total_1p19q = label_1p19q.size(0)
            correct_1p19q = predicted_1p19q.eq(label_1p19q.data).cpu().sum()
            _, predicted_CDKN = torch.max(pred_CDKN.data, 1)
            total_CDKN = label_CDKN.size(0)
            correct_CDKN = predicted_CDKN.eq(label_CDKN.data).cpu().sum()

            running_results['acc_IDH'] += 100. * correct_IDH / total_IDH
            running_results['acc_1p19q'] += 100. * correct_1p19q / total_1p19q
            running_results['acc_CDKN'] += 100. * correct_CDKN / total_CDKN

            total_it = total_it + 1
            lossdict['train/IDH_subnet'] += loss_IDH.item()
            lossdict['train/1p19q_subnet'] += loss_1p19q.item()
            lossdict['train/CDKN_subnet'] += loss_CDKN.item()


            train_bar.set_description(
                desc=opt['name'] + ' [%d/%d] I:%.2f |1:%.2f |C:%.2f |H:%.2f' % (
                    epoch, alleps,
                    running_results['acc_IDH'] / count,
                    running_results['acc_1p19q'] / count,
                    running_results['acc_CDKN'] / count,
                    running_results['acc_His'] / count,
                ))
        lossdict['train/IDH_subnet'] = lossdict['train/IDH_subnet'] / count
        lossdict['train/1p19q_subnet'] = lossdict['train/1p19q_subnet'] / count
        lossdict['train/CDKN_subnet'] = lossdict['train/CDKN_subnet'] / count
        lossdict['train/Graph_subnet'] = lossdict['train/Graph_subnet'] / count
        lossdict['train/His_subnet'] = lossdict['train/His_subnet'] / count
        saver.write_scalars(curep, lossdict)
        saver.write_log(curep, lossdict, 'traininglossLog')


        torch.nn.utils.clip_grad_norm_(Mine_model_init.parameters(), 1)
        torch.nn.utils.clip_grad_norm_(Mine_model_IDH.parameters(), 1)
        torch.nn.utils.clip_grad_norm_(Mine_model_1p19q.parameters(), 1)
        torch.nn.utils.clip_grad_norm_(Mine_model_CDKN.parameters(), 1)
        torch.nn.utils.clip_grad_norm_(Mine_model_Graph.parameters(), 1)
        # torch.nn.utils.clip_grad_norm_(Mine_model_His.parameters(), 1)
        # torch.nn.utils.clip_grad_norm_(Mine_model_Cls.parameters(), 1)

        print('-------------------------------------Val and Test--------------------------------------')
        if (curep + 1) % opt['n_ep_save'] == 0:
            if (curep + 1) > (alleps / 2):
                save_dir = os.path.join(opt['modelDir'], 'Mine_model-%04d.pth' % (curep + 1))
                state = {
                    'init': Mine_model_init.state_dict(),
                    'IDH': Mine_model_IDH.state_dict(),
                    '1p19q': Mine_model_1p19q.state_dict(),
                    'CDKN': Mine_model_CDKN.state_dict(),
                    'Graph': Mine_model_Graph.state_dict(),
                    # 'His': Mine_model_His.state_dict(),
                    # 'Cls': Mine_model_Cls.state_dict(),
                }
                torch.save(state, save_dir)

            print("----------Val-------------")
            list_WSI_IDH,list_WSI_1p19q,list_WSI_CDKN,list_WSI_His,list_WSI_His_2class,list_WSI_Diag = validation_All(opt, Mine_model_init, Mine_model_IDH, Mine_model_1p19q,Mine_model_CDKN,Mine_model_Graph,Mine_model_His,Mine_model_Cls,valLoader, saver, curep, opt['eva_cm'], gpuID)
            print('val in epoch: %d/%d, acc_IDH:%.3f,acc_1p19q:%.3f,acc_CDKN:%.3f,acc_His:%.3f,acc_His_2class:%.3f, acc_Diag:%.3f' % (
                epoch + 1, alleps, list_WSI_IDH[0], list_WSI_1p19q[0], list_WSI_CDKN[0], list_WSI_His[0], list_WSI_His_2class[0], list_WSI_Diag[0]))
            val_dict = {'val/acc_IDH': list_WSI_IDH[0], 'val/sen_IDH': list_WSI_IDH[3], 'val/spec_IDH': list_WSI_IDH[4],
                        'val/auc_IDH': list_WSI_IDH[5],'val/f1_IDH': list_WSI_IDH[2], 'val/prec_IDH': list_WSI_IDH[6],
                        'val/acc_1p19q': list_WSI_1p19q[0], 'val/sen_1p19q': list_WSI_1p19q[3], 'val/spec_1p19q': list_WSI_1p19q[4],
                        'val/auc_1p19q': list_WSI_1p19q[5], 'val/f1_1p19q': list_WSI_1p19q[2], 'val/prec_1p19q': list_WSI_1p19q[6],
                        'val/acc_CDKN': list_WSI_CDKN[0], 'val/sen_CDKN': list_WSI_CDKN[3], 'val/spec_CDKN': list_WSI_CDKN[4],
                        'val/auc_CDKN': list_WSI_CDKN[5], 'val/f1_CDKN': list_WSI_CDKN[2], 'val/prec_CDKN': list_WSI_CDKN[6],
                        'val/acc_His': list_WSI_His[0], 'val/sen_His': list_WSI_His[3], 'val/spec_His': list_WSI_His[4],
                        'val/auc_His': list_WSI_His[5], 'val/f1_His': list_WSI_His[2], 'val/prec_His': list_WSI_His[6],
                        'val/acc_His_2class': list_WSI_His_2class[0], 'val/sen_His_2class': list_WSI_His_2class[3], 'val/spec_His_2class': list_WSI_His_2class[4],
                        'val/auc_His_2class': list_WSI_His_2class[5], 'val/f1_His_2class': list_WSI_His_2class[2], 'val/prec_His_2class': list_WSI_His_2class[6],
                        'val/acc_Diag': list_WSI_Diag[0], 'val/sen_Diag': list_WSI_Diag[3], 'val/spec_Diag': list_WSI_Diag[4],
                        'val/auc_Diag': list_WSI_Diag[5], 'val/f1_Diag': list_WSI_Diag[2], 'val/prec_Diag': list_WSI_Diag[6],
                        }
            val_dict_IDH = {'val/acc_IDH': list_WSI_IDH[0], 'val/sen_IDH': list_WSI_IDH[3], 'val/spec_IDH': list_WSI_IDH[4],
                        'val/auc_IDH': list_WSI_IDH[5], 'val/f1_IDH': list_WSI_IDH[2], 'val/prec_IDH': list_WSI_IDH[6],                        }
            val_dict_1p19q = {'val/acc_1p19q': list_WSI_1p19q[0], 'val/sen_1p19q': list_WSI_1p19q[3],'val/spec_1p19q': list_WSI_1p19q[4],
                        'val/auc_1p19q': list_WSI_1p19q[5], 'val/f1_1p19q': list_WSI_1p19q[2],'val/prec_1p19q': list_WSI_1p19q[6],}
            val_dict_CDKN = {'val/acc_CDKN': list_WSI_CDKN[0], 'val/sen_CDKN': list_WSI_CDKN[3],'val/spec_CDKN': list_WSI_CDKN[4],
                        'val/auc_CDKN': list_WSI_CDKN[5], 'val/f1_CDKN': list_WSI_CDKN[2],'val/prec_CDKN': list_WSI_CDKN[6],}
            val_dict_His = {'val/acc_His': list_WSI_His[0], 'val/sen_His': list_WSI_His[3], 'val/spec_His': list_WSI_His[4],
                        'val/auc_His': list_WSI_His[5], 'val/f1_His': list_WSI_His[2], 'val/prec_His': list_WSI_His[6],}
            val_dict_His_2class = {'val/acc_His_2class': list_WSI_His_2class[0], 'val/sen_His_2class': list_WSI_His_2class[3],'val/spec_His_2classe': list_WSI_His_2class[4],
                        'val/auc_His_2class': list_WSI_His_2class[5], 'val/f1_His_2class': list_WSI_His_2class[2],'val/prec_His_2class': list_WSI_His_2class[6],}
            val_dict_Diag = {'val/acc_Diag': list_WSI_Diag[0], 'val/sen_Diag': list_WSI_Diag[3],'val/spec_Diag': list_WSI_Diag[4],
                        'val/auc_Diag': list_WSI_Diag[5], 'val/f1_Diag': list_WSI_Diag[2],'val/prec_Diag': list_WSI_Diag[6], }
            saver.write_scalars(curep, val_dict)
            saver.write_log(curep, val_dict_IDH, 'val_IDH')
            saver.write_log(curep, val_dict_1p19q, 'val_1p19q')
            saver.write_log(curep, val_dict_CDKN, 'val_CDKN')
            saver.write_log(curep, val_dict_His, 'val_His')
            saver.write_log(curep, val_dict_His_2class, 'val_His_2class')
            saver.write_log(curep, val_dict_Diag, 'val_Diag')
            #
            #
            #
            print("----------Test-------------")
            list_WSI_IDH, list_WSI_1p19q, list_WSI_CDKN, list_WSI_His, list_WSI_His_2class, list_WSI_Diag = validation_All(
                opt, Mine_model_init, Mine_model_IDH, Mine_model_1p19q, Mine_model_CDKN, Mine_model_Graph,
                Mine_model_His, Mine_model_Cls, testLoader, saver, curep, opt['eva_cm'], gpuID)
            print('test in epoch: %d/%d, acc_IDH:%.3f,acc_1p19q:%.3f,acc_CDKN:%.3f,acc_His:%.3f,acc_His_2class:%.3f, acc_Diag:%.3f' % (
                epoch + 1, alleps, list_WSI_IDH[0], list_WSI_1p19q[0], list_WSI_CDKN[0], list_WSI_His[0], list_WSI_His_2class[0], list_WSI_Diag[0]))
            test_dict = {'test/acc_IDH': list_WSI_IDH[0], 'test/sen_IDH': list_WSI_IDH[3], 'test/spec_IDH': list_WSI_IDH[4],
                        'test/auc_IDH': list_WSI_IDH[5],'test/f1_IDH': list_WSI_IDH[2], 'test/prec_IDH': list_WSI_IDH[6],
                        'test/acc_1p19q': list_WSI_1p19q[0], 'test/sen_1p19q': list_WSI_1p19q[3], 'test/spec_1p19q': list_WSI_1p19q[4],
                        'test/auc_1p19q': list_WSI_1p19q[5], 'test/f1_1p19q': list_WSI_1p19q[2], 'test/prec_1p19q': list_WSI_1p19q[6],
                        'test/acc_CDKN': list_WSI_CDKN[0], 'test/sen_CDKN': list_WSI_CDKN[3], 'test/spec_CDKN': list_WSI_CDKN[4],
                        'test/auc_CDKN': list_WSI_CDKN[5], 'test/f1_CDKN': list_WSI_CDKN[2], 'test/prec_CDKN': list_WSI_CDKN[6],
                        'test/acc_His': list_WSI_His[0], 'test/sen_His': list_WSI_His[3], 'test/spec_His': list_WSI_His[4],
                        'test/auc_His': list_WSI_His[5], 'test/f1_His': list_WSI_His[2], 'test/prec_His': list_WSI_His[6],
                        'test/acc_His_2class': list_WSI_His_2class[0], 'test/sen_His_2class': list_WSI_His_2class[3], 'test/spec_His_2class': list_WSI_His_2class[4],
                        'test/auc_His_2class': list_WSI_His_2class[5], 'test/f1_His_2class': list_WSI_His_2class[2], 'test/prec_His_2class': list_WSI_His_2class[6],
                        'test/acc_Diag': list_WSI_Diag[0], 'test/sen_Diag': list_WSI_Diag[3], 'test/spec_Diag': list_WSI_Diag[4],
                        'test/auc_Diag': list_WSI_Diag[5], 'test/f1_Diag': list_WSI_Diag[2], 'test/prec_Diag': list_WSI_Diag[6],
                        }
            test_dict_IDH = {'test/acc_IDH': list_WSI_IDH[0], 'test/sen_IDH': list_WSI_IDH[3], 'test/spec_IDH': list_WSI_IDH[4],
                        'test/auc_IDH': list_WSI_IDH[5], 'test/f1_IDH': list_WSI_IDH[2], 'test/prec_IDH': list_WSI_IDH[6],                        }
            test_dict_1p19q = {'test/acc_1p19q': list_WSI_1p19q[0], 'test/sen_1p19q': list_WSI_1p19q[3],'test/spec_1p19q': list_WSI_1p19q[4],
                        'test/auc_1p19q': list_WSI_1p19q[5], 'test/f1_1p19q': list_WSI_1p19q[2],'test/prec_1p19q': list_WSI_1p19q[6],}
            test_dict_CDKN = {'test/acc_CDKN': list_WSI_CDKN[0], 'test/sen_CDKN': list_WSI_CDKN[3],'test/spec_CDKN': list_WSI_CDKN[4],
                        'test/auc_CDKN': list_WSI_CDKN[5], 'test/f1_CDKN': list_WSI_CDKN[2],'test/prec_CDKN': list_WSI_CDKN[6],}
            test_dict_His = {'test/acc_His': list_WSI_His[0], 'test/sen_His': list_WSI_His[3], 'test/spec_His': list_WSI_His[4],
                        'test/auc_His': list_WSI_His[5], 'test/f1_His': list_WSI_His[2], 'test/prec_His': list_WSI_His[6],}
            test_dict_His_2class = {'test/acc_His_2class': list_WSI_His_2class[0], 'test/sen_His_2class': list_WSI_His_2class[3],'test/spec_His_2class': list_WSI_His_2class[4],
                        'test/auc_His_2class': list_WSI_His_2class[5], 'test/f1_His_2class': list_WSI_His_2class[2],'test/prec_His_2class': list_WSI_His_2class[6],}
            test_dict_Diag = {'test/acc_Diag': list_WSI_Diag[0], 'test/sen_Diag': list_WSI_Diag[3],'test/spec_Diag': list_WSI_Diag[4],
                        'test/auc_Diag': list_WSI_Diag[5], 'test/f1_Diag': list_WSI_Diag[2],'test/prec_Diag': list_WSI_Diag[6], }
            saver.write_scalars(curep, test_dict)
            saver.write_log(curep, test_dict_IDH, 'test_IDH')
            saver.write_log(curep, test_dict_1p19q, 'test_1p19q')
            saver.write_log(curep, test_dict_CDKN, 'test_CDKN')
            saver.write_log(curep, test_dict_His, 'test_His')
            saver.write_log(curep, test_dict_His_2class, 'test_His_2class')
            saver.write_log(curep, test_dict_Diag, 'test_Diag')


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





























