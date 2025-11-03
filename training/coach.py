import os
import shutil
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
import matplotlib
import time
from accelerate import Accelerator

from arcface_torch.backbones import get_model
from .ranger import Ranger
from utils.loss import *
from utils.dataset_large import MyDataSet

class Coach:
    def __init__(self, opts):
        super(Coach, self).__init__()
        self.accelerator = Accelerator()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.opts = opts
        self.aamsoftmax = AAMSoftmax(30000, 64, 0.5)
        self.k = self.opts.k 
        
        if opts.training_mode == 'dual':
            self.net = get_model(opts.net_type, fp16=False)
        elif opts.training_mode == 'single':
            self.net = get_model(opts.net_type, fp16=False)
        else:
            raise Exception('Unsupported training mode!')
        self.net_params = list(self.net.parameters())
        if opts.loss_type == 'AAMSoftmax':
            self.net_params += list(self.aamsoftmax.parameters()) 
        if opts.optimizer == 'Adam':
            self.net_optimizer = torch.optim.Adam(self.net_params, lr=opts.lr, betas=(opts.beta_1, opts.beta_2), weight_decay=opts.weight_decay)
        elif opts.optimizer == 'Ranger':
            self.net_optimizer = Ranger(self.net_params, lr=opts.lr, betas=(opts.beta_1, opts.beta_2), weight_decay=opts.weight_decay)
        else:
            raise Exception('Unsupported optimizer!')
        if opts.scheduler == 'StepLR':
            self.net_scheduler = torch.optim.lr_scheduler.StepLR(self.net_optimizer, step_size=opts.step_size, gamma=opts.gamma)
        elif opts.scheduler == 'Cosine':
            self.net_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.net_optimizer, opts.total_epochs)
        else:
            raise Exception('Unsupported scheduler!')

        # dataset & dataloader preparation
        train_dir_file = os.path.join(opts.dir_path_root, opts.quality, opts.train_file_name)
        train_dataset = MyDataSet(output_size=opts.img_size, dir_file_path=train_dir_file, count=opts.frame_count, status=opts.train_status, mode=opts.training_mode)
        self.train_loader = data.DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

        cross_test_dir_file = os.path.join(opts.dir_path_root, opts.quality, opts.test_file_name)
        cross_test_dataset = MyDataSet(output_size=opts.img_size, dir_file_path=cross_test_dir_file, count=opts.frame_count, status=opts.test_status, mode=opts.testing_mode)
        self.cross_test_loader = data.DataLoader(cross_test_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)  
        
        if self.opts.cross:
            origin_cross_dir_file = os.path.join(opts.dir_path_root, opts.quality, opts.test_origin_file_name)
            origin_cross_dataset = MyDataSet(output_size=opts.img_size, dir_file_path=origin_cross_dir_file, count=opts.frame_count, status=opts.test_status, mode=opts.testing_mode) 
            self.origin_cross_loader = data.DataLoader(origin_cross_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)       

        # prepare accelerator
        self.net, self.aamsoftmax, self.net_optimizer, \
            self.train_loader, self.cross_test_loader = \
            self.accelerator.prepare(self.net, self.aamsoftmax, self.net_optimizer, \
                                     self.train_loader, self.cross_test_loader)

        self.topk_accs = []
        self.cross_topk_accs = []

    def train(self, n_epoch):
        self.net.train()
        self.accelerator.print('Training epoch %d......'%n_epoch)
        self.accelerator.print('-----------------Training Process-----------------')
        if self.accelerator.is_main_process:
            tic = time.time()   
        topkacc = torch.tensor(0.)     
        for i, batch in enumerate(self.train_loader):
            img_list, src_list, tar_list, labels = batch
            avg_pred = 0.
            avg_gt = 0.
            for index in range(self.opts.frame_count):
                base_pred = self.net(img_list[index], tar_list[index])
                avg_pred += base_pred / self.opts.frame_count

            self.net_optimizer.zero_grad()
            if self.opts.loss_type == 'aamsoftmax':
                self.loss, topkacc = self.aamsoftmax(avg_pred, labels, 'normal', self.k)
            else:
                raise Exception('Unsupported loss type!')
            self.accelerator.backward(self.loss)
            self.net_optimizer.step()
            if (i + 1) % self.opts.print_rate == 0:
                self.accelerator.print(f"Loss value of iteration {i + 1}: {self.loss.mean().detach().cpu().numpy():.4f}")
                self.accelerator.print(f"Top-{self.k} accuracy of iteration {i + 1}: {topkacc.mean().detach().cpu().numpy():.4f}")
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            toc = time.time() 
            self.accelerator.print(f"Training time of Epoch {n_epoch}: {int((toc - tic)/60)}min")

    def cross_test(self, epoch):
        self.net.eval()
        cross_test_loss = 0.
        cross_topk_acc = 0.
        self.accelerator.print('-----------------Cross Testing Process-----------------')
        if self.accelerator.is_main_process:
            tic = time.time() 
        hard_labels = torch.zeros(len(self.origin_cross_loader), 512).cuda()
        hard_indices = torch.zeros(len(self.origin_cross_loader)).long().cuda()
        with torch.no_grad():
            for i, ori_batch in enumerate(self.origin_cross_loader):
                ori_img_list, ori_src_list, ori_tar_list, ori_label = ori_batch
                ori_avg_pred, ori_avg_gt = self.get_embedding(ori_img_list, ori_src_list, ori_tar_list)
                hard_labels[i] = ori_avg_pred
                hard_indices[i] = ori_label.cuda()
        self.accelerator.print('Label calculated!')

        for i, batch in enumerate(self.cross_test_loader):
            img_list, src_list, tar_list, label = batch
            avg_pred, avg_gt = self.get_embedding(img_list, src_list, tar_list)
            if self.opts.loss_type == 'aamsoftmax':
                prec = 0.
                for j in range(len(avg_pred)):
                    cur_sim = 1 - cosine_loss(avg_pred[j], hard_labels)
                    value, indices = torch.topk(cur_sim, self.k)
                    if label[j] in hard_indices[indices]:
                        prec += 1
                prec = prec * 100 / 8
                cross_topk_acc += prec
            else:
                raise Exception('Unsupported loss type!')
        self.accelerator.wait_for_everyone()
        cross_test_loss = cross_test_loss / len(self.cross_test_loader)
        cross_topk_acc = cross_topk_acc / len(self.cross_test_loader)
        self.accelerator.print(f"Cross testing loss value of epoch {epoch}: {cross_test_loss:.4f}")
        self.accelerator.print(f"Cross testing top-{self.k} accuracy of epoch {epoch}: {cross_topk_acc:.4f}")
        if self.accelerator.is_main_process:
            self.cross_topk_accs.append(cross_topk_acc)
            self.make_loss_figure('cross_topk_acc', self.opts.log_path)
            toc = time.time() 
            self.accelerator.print(f"Cross testing time of Epoch {epoch}: {int((toc - tic)/60)}min")

    def get_embedding(self, img_list, src_list, tar_list):
        avg_pred = 0.
        avg_gt = 0.
        # blank image is need when calculate ground truth of a raw image
        emb = torch.zeros_like(img_list[0]).to(self.accelerator.device)
        for index in range(self.opts.frame_count):
            base_pred = self.net(img_list[index].to(self.accelerator.device), tar_list[index].to(self.accelerator.device))
            avg_pred += (base_pred / self.opts.frame_count)
            avg_gt += self.net(src_list[index].to(self.accelerator.device), emb) / self.opts.frame_count
        return avg_pred, avg_gt
    
    def save_model(self, log_path):
        self.accelerator.wait_for_everyone()
        unwrapped_model_net = self.accelerator.unwrap_model(self.net)
        self.accelerator.save(unwrapped_model_net.state_dict(), '{:s}/net.pth.tar'.format(log_path))
        unwrapped_model_aamsoftmax = self.accelerator.unwrap_model(self.aamsoftmax)
        self.accelerator.save(unwrapped_model_aamsoftmax.state_dict(), '{:s}/aamsoftmax.pth.tar'.format(log_path))

    def save_checkpoint(self, n_epoch, log_path):
        if self.accelerator.is_main_process:
            if os.path.exists('{:s}/checkpoint_{:d}'.format(log_path, n_epoch)) and os.path.isdir('{:s}/checkpoint_{:d}'.format(log_path, n_epoch)):
                shutil.rmtree('{:s}/checkpoint_{:d}'.format(log_path, n_epoch))
            self.accelerator.save_state('{:s}/checkpoint_{:d}'.format(log_path, n_epoch))
    
    def load_model(self, log_path):
        unwrapped_model_net = self.accelerator.unwrap_model(self.net)
        unwrapped_model_net.load_state_dict(torch.load('{:s}/net.pth.tar'.format(log_path)))
        self.accelerator.print('Net loaded:{:s}/net.pth.tar'.format(log_path))
        unwrapped_model_aamsoftmax = self.accelerator.unwrap_model(self.aamsoftmax)
        unwrapped_model_aamsoftmax.load_state_dict(torch.load('{:s}/aamsoftmax.pth.tar'.format(log_path)))
        self.accelerator.print('AAMSoftmax loaded:{:s}/aamsoftmax.pth.tar'.format(log_path))
    
    def load_checkpoint(self, checkpoint_path):
        if self.accelerator.is_main_process:
            self.accelerator.load_state(checkpoint_path)
            self.accelerator.print('loading checkpoint:' + checkpoint_path + '...')
        epoch_count = int(checkpoint_path.split('/')[-1].split('_')[-1])
        return epoch_count + 1

    def make_loss_figure(self, process, log_path):
        matplotlib.use('Agg')
        if process == 'topk_acc':
            plt.plot(range(len(self.topk_accs)), self.topk_accs, marker='o', linestyle='-')
            plt.xlabel('Epochs')
            plt.ylabel('Top-k acc')
            plt.savefig(log_path + 'topk_acc.png')
        elif process == 'cross_topk_acc':
            plt.plot(range(len(self.cross_topk_accs)), self.cross_topk_accs, marker='o', linestyle='-')
            plt.xlabel('Epochs')
            plt.ylabel('Cross top-k acc')
            plt.savefig(log_path + 'cross_topk_acc.png')                         
        else:
            raise Exception('Unsupportted process type!')
        plt.close()
