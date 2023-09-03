import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.cuda
from torch.autograd import Variable
import basic_net as basic_net
import yaml
from yaml.loader import SafeLoader
from net import *
import copy
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from utils import FocalLoss
import scipy.sparse as sp
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx





class Mine_init(nn.Module):
    def __init__(self, opt, vis=False,if_end2end=False):
        super(Mine_init, self).__init__()
        self.opt = opt
        self.vis = vis
        self.size = [1024, 512]
        self.if_end2end=if_end2end
        if self.if_end2end:
            fc0 = [nn.Linear(224*224*3, self.size[0]), nn.ReLU()]
            self.attention_init_0 = nn.Sequential(*fc0)

        self.default_patchnum = self.opt['fixdim']
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.default_patchnum, 1024))
        fc = [nn.Linear(1024, self.size[1]), nn.ReLU()]
        fc.append(nn.Dropout(0.25))
        self.attention_init = nn.Sequential(*fc)

    def forward(self,x):
        if self.if_end2end:
            x=self.attention_init_0(x)
        embeddings = x + self.position_embeddings  # [B, n, 1024]
        hidden_states_stem = self.attention_init(embeddings)  # [B, n, 512]
        return hidden_states_stem



class Mine_IDH(nn.Module):
    def __init__(self, opt, vis=False):
        super(Mine_IDH, self).__init__()
        self.opt = opt
        self.vis = vis
        self.size = [1024, 512]
        self.n_classes_IDH = 2
        # Trans blocks
        self.layer_IDH = nn.ModuleList()
        for _ in range(self.opt['Network']['IDH_layers']):
            layer = Block(opt, self.size[1], vis)
            self.layer_IDH.append(copy.deepcopy(layer))
        self.encoder_norm_IDH = LayerNorm(self.size[1], eps=1e-6)

    def forward(self, hidden_states):
        attn_weights_IDH = []
        for layer_block in self.layer_IDH:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights_IDH.append(weights)

        encoded_IDH = self.encoder_norm_IDH(hidden_states)  # [B,2500,512]
        # encoded_IDH = torch.mean(encoded_IDH, dim=1)
        return hidden_states,encoded_IDH


class Mine_1p19q(nn.Module):
    def __init__(self, opt, vis=False):
        super(Mine_1p19q, self).__init__()
        self.opt = opt
        self.vis = vis
        self.size = [1024, 512]
        self.n_classes_1p19q = 2
        self.layer_1p19q = nn.ModuleList()
        for _ in range(self.opt['Network']['1p19q_layers']):
            layer = Block(opt, self.size[1], vis)
            self.layer_1p19q.append(copy.deepcopy(layer))
        self.encoder_norm_1p19q = LayerNorm(self.size[1], eps=1e-6)

    def forward(self,hidden_states):
        attn_weights_1p19q = []
        for layer_block in self.layer_1p19q:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights_1p19q.append(weights)


        encoded_1p19q = self.encoder_norm_1p19q(hidden_states)# [B,2500,512]
        # encoded_1p19q = torch.mean(encoded_1p19q, dim=1)

        return hidden_states,encoded_1p19q


class Mine_CDKN(nn.Module):
    def __init__(self, opt, vis=False):
        super(Mine_CDKN, self).__init__()
        self.opt = opt
        self.vis = vis
        self.size = [1024, 512]
        self.n_classes_CDKN = 2
        self.layer_CDKN = nn.ModuleList()
        for _ in range(self.opt['Network']['CDKN_layers']):
            layer = Block(opt, self.size[1], vis)
            self.layer_CDKN.append(copy.deepcopy(layer))
        self.encoder_norm_CDKN = LayerNorm(self.size[1], eps=1e-6)

    def forward(self, hidden_states):
        attn_weights_CDKN = []
        for layer_block in self.layer_CDKN:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights_CDKN.append(weights)


        encoded_CDKN = self.encoder_norm_CDKN(hidden_states)# [B,2500,512]
        # encoded_CDKN = torch.mean(encoded_CDKN, dim=1)
        return encoded_CDKN

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn import CrossEntropyLoss, Dropout
class Label_correlation_Graph(nn.Module):
    def __init__(self, opt, vis=False):
        super(Label_correlation_Graph, self).__init__()
        self.opt = opt
        self.size = [1024, 512]
        self.adj=np.array([[1,0.4038,0.3035],[1,1,0.1263],[0.2595,0.0436,1]])

        self.alpha=self.opt['Network']['graph_alpha']
        self.n_classes_IDH=2
        self.n_classes_CDKN=2
        self.n_classes_1p19q=2


        # self._fc2_IDH = nn.Linear(self.size[1], self.n_classes_IDH)
        self.criterion_ce_IDH = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1, 1])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
        self.criterion_focal_IDH = FocalLoss(alpha=1).cuda(opt['gpus'][0])

        # self._fc2_1p19q = nn.Linear(self.size[1], self.n_classes_1p19q)
        self.criterion_ce_1p19q = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1, 4.6])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
        self.criterion_focal_1p19q = FocalLoss(alpha=0.25).cuda(opt['gpus'][0])

        # self._fc2_CDKN = nn.Linear(self.size[1], self.n_classes_CDKN)
        self.criterion_ce_CDKN = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([2, 1])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
        self.criterion_focal_CDKN = FocalLoss(alpha=0.25).cuda(opt['gpus'][0])

        self.encoder_norm_IDH = LayerNorm(self.size[1], eps=1e-6)
        self.encoder_norm_1p19q = LayerNorm(self.size[1], eps=1e-6)
        self.encoder_norm_CDKN = LayerNorm(self.size[1], eps=1e-6)

        self.gc1 = GraphConvolution(self.size[1], self.size[1])
        self.gc2 = GraphConvolution(self.size[1], 2)
        self.dropout=Dropout(0.5)

        self._fc2_IDH_1 = nn.Linear(self.opt['fixdim'], self.n_classes_IDH)
        self._fc2_CDKN_1 = nn.Linear(self.opt['fixdim'], self.n_classes_CDKN)
        self._fc2_1p19q_1 = nn.Linear(self.opt['fixdim'], self.n_classes_1p19q)


    def forward(self, encoded_IDH,encoded_1p19q,encoded_CDKN):
        """
        shape:[BS,512]
        """

        encoded_IDH=torch.unsqueeze(encoded_IDH, dim=3) #[BS,2500,512,1]
        encoded_1p19q = torch.unsqueeze(encoded_1p19q, dim=3)
        encoded_CDKN = torch.unsqueeze(encoded_CDKN, dim=3)
        GCN_input=torch.cat((encoded_IDH,encoded_1p19q,encoded_CDKN),3)#[BS,2500,512,3 ]
        # GCN_output =self.gc1(GCN_input)#[BS,2500,512,3 ]
        GCN_output =  F.relu(self.gc1(GCN_input))#[BS,2500,512,3 ]
        # GCN_output = self.dropout(GCN_output)
        # GCN_output = self.gc2(GCN_output)#[BS,2,3]

        GCN_output=GCN_output*self.alpha+GCN_input*(1-self.alpha)#[BS,2500,512,3 ]

        self.encoded_IDH_0=GCN_output[...,0]#[BS,2500,512]
        self.encoded_IDH_0 = self.encoder_norm_IDH(self.encoded_IDH_0)#[BS,2500,512]
        self.encoded_IDH=torch.mean(self.encoded_IDH_0,dim=2)#[BS,2500]
        logits_IDH = self._fc2_IDH_1(self.encoded_IDH)#[BS,2]
        weight_IDH_wt=self._fc2_IDH_1.weight[0]#[2500]
        weight_IDH_mut = self._fc2_IDH_1.weight[1]  # [2500]

        self.encoded_1p19q_0 = GCN_output[..., 1]#[BS,2500,512]
        self.encoded_1p19q_0 = self.encoder_norm_1p19q(self.encoded_1p19q_0)#[BS,2500,512]
        self.encoded_1p19q = torch.mean(self.encoded_1p19q_0, dim=2)  # [BS,2500]
        logits_1p19q = self._fc2_1p19q_1(self.encoded_1p19q)#[BS,2]
        weight_1p19q_codel = self._fc2_1p19q_1.weight[1]  # [2500]

        self.encoded_CDKN_0 = GCN_output[..., 2]#[BS,2500,512]
        self.encoded_CDKN_0 = self.encoder_norm_CDKN(self.encoded_CDKN_0)#[BS,2500,512]
        self.encoded_CDKN = torch.mean(self.encoded_CDKN_0, dim=2)  # [BS,2500]
        logits_CDKN = self._fc2_CDKN_1(self.encoded_CDKN)#[BS,2]
        weight_CDKN_HOMDEL = self._fc2_CDKN_1.weight[1]  # [2500]

        results_dict = {'logits_IDH': logits_IDH,'logits_1p19q': logits_1p19q,'logits_CDKN': logits_CDKN}

        return results_dict,weight_IDH_wt,weight_IDH_mut,weight_1p19q_codel,weight_CDKN_HOMDEL,torch.mean(self.encoded_IDH_0,dim=1),torch.mean(self.encoded_1p19q_0,dim=1),torch.mean(self.encoded_CDKN_0,dim=1)



    def calculateLoss_Graph(self,encoded_IDH,encoded_1p19q,encoded_CDKN):

        dis_IDH_IDH = F.cosine_similarity(encoded_IDH, encoded_IDH, dim=1)
        dis_IDH_1p19 = F.cosine_similarity(encoded_IDH, encoded_1p19q, dim=1)
        dis_IDH_CDKN = F.cosine_similarity(encoded_IDH, encoded_CDKN, dim=1)
        dis_1p19_IDH = F.cosine_similarity(encoded_1p19q, encoded_IDH, dim=1)
        dis_1p19_1p19 = F.cosine_similarity(encoded_1p19q, encoded_1p19q, dim=1)
        dis_1p19_CDKN = F.cosine_similarity(encoded_1p19q, encoded_CDKN, dim=1)
        dis_CDKN_IDH = F.cosine_similarity(encoded_CDKN, encoded_IDH, dim=1)
        dis_CDKN_1p19 = F.cosine_similarity(encoded_CDKN, encoded_1p19q, dim=1)
        dis_CDKN_CDKN = F.cosine_similarity(encoded_CDKN, encoded_CDKN, dim=1)

        cos_dis_matrix=[dis_IDH_IDH,dis_IDH_1p19,dis_IDH_CDKN,dis_1p19_IDH,dis_1p19_1p19,dis_1p19_CDKN,dis_CDKN_IDH,
                        dis_CDKN_1p19,dis_CDKN_CDKN]

        adj_T = self.adj.T
        adj = (adj_T + self.adj) / 2
        adj=torch.from_numpy(np.array(adj)).float().cuda(self.opt['gpus'][0])
        adj=torch.unsqueeze(adj,dim=0)
        adj =adj.repeat(dis_IDH_IDH.detach().cpu().numpy().shape[0],1,1)


        dis_1p19_CDKN = dis_1p19_CDKN.detach().cpu().numpy()  # [BS]
        dis_1p19_CDKN_FLAG=np.ones(dis_IDH_IDH.detach().cpu().numpy().shape[0],dtype=float)
        for i in range(dis_IDH_IDH.detach().cpu().numpy().shape[0]):
            if dis_1p19_CDKN[i]<0.1:
                dis_1p19_CDKN_FLAG[i]=0
        dis_1p19_CDKN_FLAG=torch.from_numpy(np.array(dis_1p19_CDKN_FLAG)).float().cuda(self.opt['gpus'][0])

        dis_CDKN_1p19 = dis_CDKN_1p19.detach().cpu().numpy()  # [BS]
        dis_CDKN_1p19_FLAG = np.ones(dis_IDH_IDH.detach().cpu().numpy().shape[0], dtype=float)
        for i in range(dis_IDH_IDH.detach().cpu().numpy().shape[0]):
            if dis_CDKN_1p19[i] < 0.1:
                dis_CDKN_1p19_FLAG[i] = 0
        dis_CDKN_1p19_FLAG = torch.from_numpy(np.array(dis_CDKN_1p19_FLAG)).float().cuda(self.opt['gpus'][0])


        self.loss_Graph = (cos_dis_matrix[0] - adj[:, 0, 0]) ** 2 + (cos_dis_matrix[1] - adj[:, 0, 1]) ** 2 + (cos_dis_matrix[2] - adj[:, 0, 2]) ** 2 \
                          + (cos_dis_matrix[3] - adj[:, 1, 0]) ** 2 + (cos_dis_matrix[4] - adj[:, 1, 1]) ** 2 +dis_1p19_CDKN_FLAG*(cos_dis_matrix[5] - adj[:, 1, 2]) ** 2 \
                          + (cos_dis_matrix[6] - adj[:, 2, 0]) ** 2 + dis_CDKN_1p19_FLAG*(cos_dis_matrix[7] - adj[:, 2, 1]) ** 2 + (cos_dis_matrix[8] - adj[:, 2, 2]) ** 2
        return torch.mean(self.loss_Graph)



    def calculateLoss_IDH(self, pred, label):
        self.loss_IDH = self.criterion_ce_IDH(pred, label)
        return self.loss_IDH

    def calculateLoss_1p19q(self, pred, label):
        self.loss_1p19q = self.criterion_ce_1p19q(pred, label)
        return self.loss_1p19q

    def calculateLoss_CDKN(self, pred, label):
        self.loss_CDKN = self.criterion_ce_CDKN(pred, label)
        return self.loss_CDKN

class Mine_His(nn.Module):
    def __init__(self, opt, vis=False):
        super(Mine_His, self).__init__()
        self.opt = opt
        self.vis = vis
        self.size = [1024, 512]
        self.n_classes_His = 4
        self.layer_His = nn.ModuleList()
        for _ in range(self.opt['Network']['His_layers']):
            layer = Block(opt, self.size[1], vis)
            self.layer_His.append(copy.deepcopy(layer))
        self.encoder_norm_His = LayerNorm(self.size[1], eps=1e-6)
        self.criterion_ce_His = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([4, 3, 2.5, 1])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
        self.criterion_ce_His_2class = nn.CrossEntropyLoss().cuda(opt['gpus'][0])
        self.criterion_mse_diag = nn.MSELoss().cuda(opt['gpus'][0])



    def forward(self, hidden_states):
        attn_weights_His = []
        count = 0
        for layer_block in self.layer_His:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights_His.append(weights)
        encoded_His = self.encoder_norm_His(hidden_states)  # [B,2500,512]

        return hidden_states, encoded_His
        # encoded_His = self.encoder_norm_His(hidden_states)
        # logits_His = self._fc2_His(torch.mean(encoded_His, dim=1))
        # results_dict = {'logits': logits_His}
        # return hidden_states,results_dict

    def calculateLoss_His(self, pred, label):
        self.loss_His = self.criterion_ce_His(pred, label)
        return self.loss_His
    def calculateLoss_His_2class(self, pred, label):
        self.loss_His_2_class = self.criterion_ce_His_2class(pred, label)
        return self.loss_His_2_class
    def calculateLoss_diag(self, pred, label):
        self.loss_diag = self.criterion_mse_diag(pred, label)
        return self.loss_diag

class Mine_Grade(nn.Module):
    def __init__(self, opt, vis=False):
        super(Mine_Grade, self).__init__()
        self.opt = opt
        self.vis = vis
        self.size = [1024, 512]
        self.n_classes_Grade = 3
        self.layer_Grade = nn.ModuleList()
        for _ in range(self.opt['Network']['Grade_layers']):
            layer = Block(opt, self.size[1], vis)
            self.layer_Grade.append(copy.deepcopy(layer))
        self.encoder_norm_Grade = LayerNorm(self.size[1], eps=1e-6)
        self._fc2_Grade = nn.Linear(self.size[1], self.n_classes_Grade)
        self.criterion_ce_Grade = nn.CrossEntropyLoss(
            weight=torch.from_numpy(np.array([3.6, 4.8, 1])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
        self.criterion_focal_Grade = FocalLoss(alpha=1).cuda(opt['gpus'][0])


    def forward(self, hidden_states):
        attn_weights_Grade = []
        for layer_block in self.layer_Grade:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights_Grade.append(weights)
        encoded_Grade = self.encoder_norm_Grade(hidden_states)  # [B,2500,512]
        return encoded_Grade
        # encoded_Grade = self.encoder_norm_Grade(hidden_states)
        # logits_Grade = self._fc2_Grade(torch.mean(encoded_Grade, dim=1))
        # results_dict = {'logits': logits_Grade}
        # return results_dict

    def calculateLoss_Grade(self, pred, label):
        self.loss_Grade = self.criterion_ce_Grade(pred, label)
        return self.loss_Grade

class Cls_His_Grade(nn.Module):
    def __init__(self, opt, vis=False):
        super(Cls_His_Grade, self).__init__()
        self.opt = opt
        self.n_classes_Grade = 3
        self.n_classes_His = 4
        self._fc2_His_1 = nn.Linear(self.opt['fixdim'], self.n_classes_His)
        self._fc2_His_2class = nn.Linear(self.opt['fixdim'], 2)

    def forward(self, encoded_His):
        # encoded_Grade = torch.mean(encoded_Grade, dim=2)  # [BS,2500]
        # logits_Grade = self._fc2_Grade_1(encoded_Grade)  # [BS,2]

        encoded_His = torch.mean(encoded_His, dim=2)  # [BS,2500]
        encoded_His_de0=encoded_His[0].detach().cpu().numpy()
        encoded_His_de1 = encoded_His[1].detach().cpu().numpy()
        encoded_His_de2 = encoded_His[2].detach().cpu().numpy()
        encoded_His_de3 = encoded_His[3].detach().cpu().numpy()
        logits_His = self._fc2_His_1(encoded_His)  # [BS,2]
        weight_His_GBM = self._fc2_His_1.weight[3]  # [2500]
        weight_His_GBM_de=weight_His_GBM.detach().cpu().numpy()
        logits_His_2class=self._fc2_His_2class(encoded_His)  # [BS,2]
        weight_His_GBM_Cls2 = self._fc2_His_2class.weight[1]  # [2500]

        results_dict = {'logits_His': logits_His,'logits_His_2class': logits_His_2class}
        return  results_dict,weight_His_GBM,weight_His_GBM_Cls2
    def Loss_mutual_correlation(self,weight_IDH_wt,weight_1p19q_codel,weight_His_GBM,epoch):
        """
        shape:torch[2500]
        """

        weight_IDH_wt=weight_IDH_wt.tolist()
        weight_His_GBM=weight_His_GBM.tolist()
        x=weight_IDH_wt

        b = sorted(enumerate(x), key=lambda x: x[1], reverse=True)
        Index_IDH_wt = [x[0] for x in b]


        x = weight_His_GBM
        b = sorted(enumerate(x), key=lambda x: x[1], reverse=True)
        Index_His_GBM = [x[0] for x in b]


        #### IDH-wt  **** GBM
        loss_IDH_GBM=0
        self.opt['Network']['top_K_patch']=int(self.opt['fixdim']/2)
        top_K_patch=int(self.opt['Network']['top_K_patch']*(0.85**(int(epoch/10))))
        for i in range(top_K_patch):
            index_patch_low=Index_IDH_wt[i]
            if i<=int(self.opt['Network']['top_K_patch']/2):
                target_low_index_list=Index_His_GBM[0:self.opt['Network']['top_K_patch']]
            else:
                target_low_index_list=Index_His_GBM[i-int(self.opt['Network']['top_K_patch']/2):i+int(self.opt['Network']['top_K_patch']/2)]
            if not index_patch_low in target_low_index_list:
                loss_IDH_GBM+=1
        loss_IDH_GBM=loss_IDH_GBM/top_K_patch
        loss_GBM_IDH = 0
        top_K_patch = int(self.opt['Network']['top_K_patch'] * (0.85 ** (int(epoch / 10))))
        for i in range(top_K_patch):
            index_patch_low = Index_His_GBM[i]
            if i <= int(self.opt['Network']['top_K_patch'] / 2):
                target_low_index_list = Index_IDH_wt[0:self.opt['Network']['top_K_patch']]
            else:
                target_low_index_list = Index_IDH_wt[i - int(self.opt['Network']['top_K_patch'] / 2):i + int(self.opt['Network']['top_K_patch'] / 2)]
            if not index_patch_low in target_low_index_list:
                loss_GBM_IDH += 1
        loss_GBM_IDH = loss_GBM_IDH / top_K_patch
        loss_IDH_GBM=(loss_GBM_IDH+loss_IDH_GBM)/2
        #### 1p19q codel  **** O
        # loss_1p19q_O=0
        # top_K_patch=int(self.opt['Network']['top_K_patch']*(0.67**(int(epoch/10))))
        # for i in range(top_K_patch):
        #     index_patch_low=Index_1p19q_codel[i]
        #     if i<=int(self.opt['Network']['top_K_patch']/2):
        #         target_low_index_list=Index_His_O[0:self.opt['Network']['top_K_patch']]
        #     else:
        #         target_low_index_list=Index_His_O[i-int(self.opt['Network']['top_K_patch']/2):i+int(self.opt['Network']['top_K_patch']/2)]
        #     if not index_patch_low in target_low_index_list:
        #         loss_1p19q_O+=1
        # loss_1p19q_O=loss_1p19q_O/top_K_patch
        # loss_O_1p19q = 0
        # top_K_patch = int(self.opt['Network']['top_K_patch'] * (0.67 ** (int(epoch / 10))))
        # for i in range(top_K_patch):
        #     index_patch_low = Index_His_O[i]
        #     if i <= int(self.opt['Network']['top_K_patch'] / 2):
        #         target_low_index_list = Index_1p19q_codel[0:self.opt['Network']['top_K_patch']]
        #     else:
        #         target_low_index_list = Index_1p19q_codel[i - int(self.opt['Network']['top_K_patch'] / 2):i + int(self.opt['Network']['top_K_patch'] / 2)]
        #     if not index_patch_low in target_low_index_list:
        #         loss_O_1p19q += 1
        # loss_O_1p19q = loss_O_1p19q / top_K_patch
        # loss_1p19q_O=(loss_O_1p19q+loss_1p19q_O)/2

        loss_mutual_correlation=loss_IDH_GBM
        loss_mutual_correlation=torch.from_numpy(np.array([loss_mutual_correlation])).cuda(self.opt['gpus'][0])[0]
        return loss_mutual_correlation
import argparse, time, random
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if seed == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
if __name__ == "__main__":


    import argparse
    import h5py
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='./config/mine.yml')
    args = parser.parse_args()
    with open(args.opt) as f:
        opt = yaml.load(f, Loader=SafeLoader)
    setup_seed(opt['seed'])
    gpuID = opt['gpus']
    res_init = Mine_init(opt).cuda(opt['gpus'][0])
    res_IDH=Mine_IDH(opt).cuda(opt['gpus'][0])
    res_1p19q = Mine_1p19q(opt).cuda(opt['gpus'][0])
    res_CDKN = Mine_CDKN(opt).cuda(opt['gpus'][0])
    res_Graph = Label_correlation_Graph(opt).cuda(opt['gpus'][0])
    res_His = Mine_His(opt).cuda(opt['gpus'][0])
    # res_Grade = Mine_Grade(opt).cuda(opt['gpus'][0])
    res_Cls_His_Grade = Cls_His_Grade(opt).cuda(opt['gpus'][0])
    #
    init_weights(res_init, init_type='xavier', init_gain=1)
    init_weights(res_IDH, init_type='xavier', init_gain=1)
    init_weights(res_1p19q, init_type='xavier', init_gain=1)
    init_weights(res_CDKN, init_type='xavier', init_gain=1)
    init_weights(res_His, init_type='xavier', init_gain=1)
    # init_weights(res_Grade, init_type='xavier', init_gain=1)
    device = torch.device('cuda:{}'.format(opt['gpus'][0]))
    res_init = torch.nn.DataParallel(res_init, device_ids=gpuID)
    res_IDH = torch.nn.DataParallel(res_IDH, device_ids=gpuID)
    res_1p19q = torch.nn.DataParallel(res_1p19q, device_ids=gpuID)
    res_CDKN = torch.nn.DataParallel(res_CDKN, device_ids=gpuID)
    res_Graph = torch.nn.DataParallel(res_Graph, device_ids=gpuID)
    res_His = torch.nn.DataParallel(res_His, device_ids=gpuID)
    res_Cls_His_Grade = torch.nn.DataParallel(res_Cls_His_Grade, device_ids=gpuID)
    # res_His.to(device)
    # res_Grade.to(device)
    # res_Cls_His_Grade.to(device)
    #
    input1 = torch.ones((4, 2500,1024)).cuda(opt['gpus'][0])
    # root = opt['dataDir'] + 'Res50_feature_2500_fixdim0/'
    # # root=r'D:\PhD\Project_WSI\data\Res50_feature_2500/'
    # patch_all0 =torch.from_numpy(np.array(h5py.File(root + 'TCGA-DU-A5TY-01Z-00-DX1.h5')['Res_feature'][:])).float().cuda(opt['gpus'][0])# (1,N,1024)
    # patch_all1=torch.from_numpy(np.array(h5py.File(root + 'TCGA-HT-8104-01A-01-TS1.h5')['Res_feature'][:])).float().cuda(opt['gpus'][0])# (1,N,1024)
    # patch_all2 = torch.from_numpy(np.array(h5py.File(root + 'TCGA-CS-6188-01A-01-BS1.h5')['Res_feature'][:])).float().cuda(opt['gpus'][0])  # (1,N,1024)
    # patch_all3 = torch.from_numpy(np.array(h5py.File(root + 'TCGA-DU-7010-01Z-00-DX1.h5')['Res_feature'][:])).float().cuda(opt['gpus'][0])  # (1,N,1024)
    # input1 = torch.cat((patch_all0, patch_all1, patch_all2,patch_all3), 0)  # [4,N,1024]
    #
    hidden_states_init = res_init(input1)
    #
    hidden_states, encoded_IDH=res_IDH(hidden_states_init)
    hidden_states, encoded_1p19q = res_1p19q(hidden_states)
    encoded_CDKN = res_CDKN(hidden_states)
    # # a_max = np.max(hidden_states.detach().numpy()[0])
    # # a_min = np.min(hidden_states.detach().numpy()[0])
    #
    out,weight_IDH_wt,weight_IDH_mut,weight_1p19q_codel,weight_CDKN_HOMDEL,encoded_IDH0, encoded_1p19q0, encoded_CDKN0 = res_Graph(encoded_IDH,encoded_1p19q,encoded_CDKN)
    # loss_IDH = res_Graph.calculateLoss_IDH(out['logits_IDH'], torch.from_numpy(np.array([1,1,1,1])).cuda(opt['gpus'][0]))
    loss_Graph=res_Graph.module.calculateLoss_Graph(encoded_IDH0, encoded_1p19q0, encoded_CDKN0)
    hidden_states, encoded_His = res_His(hidden_states_init)
    # encoded_Grade=res_Grade(hidden_states)
    out,weight_His_GBM,weight_His_O=res_Cls_His_Grade(encoded_His)
    weight_IDH_wt=weight_IDH_wt[0:int(weight_IDH_wt.detach().cpu().numpy().shape[0]/len(gpuID))]
    weight_1p19q_codel = weight_1p19q_codel[0:int(weight_1p19q_codel.detach().cpu().numpy().shape[0] / len(gpuID))]
    weight_His_GBM = weight_His_GBM[0:int(weight_His_GBM.detach().cpu().numpy().shape[0] / len(gpuID))]
    loss_mutual_correlation=res_Cls_His_Grade.module.Loss_mutual_correlation(weight_IDH_wt, weight_1p19q_codel, weight_His_GBM, 0)
    a=1
































