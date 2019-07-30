from __future__ import print_function

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from utils.lr_schedule import inv_lr_scheduler
from utils.loss import entropy, adentropy

class Entropy():
    def __init__(self, G, F1, args, source_loader, target_loader, target_loader_unl, class_list):
        self.G = G
        self.F1 = F1
        self.args = args
        self.source_loader = source_loader
        self.target_loader = target_loader
        self.target_loader_unl = target_loader_unl
        self.class_list = class_list
        self.num_class = len(class_list)
        self.params = []
        for key, value in dict(self.G.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    self.params += [{'params': [value], 'lr': self.args.multi, 'weight_decay': 0.0005}]
                else:
                    self.params += [{'params': [value], 'lr': self.args.multi, 'weight_decay': 0.0005}]
        
        
        self.im_data_s = torch.FloatTensor(1)
        self.im_data_t = torch.FloatTensor(1)
        self.im_data_tu = torch.FloatTensor(1)
        self.gt_labels_s = torch.LongTensor(1)
        self.gt_labels_t = torch.LongTensor(1)
        self.sample_labels_t = torch.LongTensor(1)
        self.sample_labels_s = torch.LongTensor(1)

        self.im_data_s = self.im_data_s.cuda()
        self.im_data_t = self.im_data_t.cuda()
        self.im_data_tu = self.im_data_tu.cuda()
        self.gt_labels_s = self.gt_labels_s.cuda()
        self.gt_labels_t = self.gt_labels_t.cuda()
        self.sample_labels_t = self.sample_labels_t.cuda()
        self.sample_labels_s = self.sample_labels_s.cuda()

        self.im_data_s = Variable(self.im_data_s)
        self.im_data_t = Variable(self.im_data_t)
        self.im_data_tu = Variable(self.im_data_tu)
        self.gt_labels_s = Variable(self.gt_labels_s)
        self.gt_labels_t = Variable(self.gt_labels_t)
        self.sample_labels_t = Variable(self.sample_labels_t)
        self.sample_labels_s = Variable(self.sample_labels_s)

        if os.path.exists(self.args.checkpath) == False:
            os.mkdir(self.args.checkpath)

    def train(self):
        self.G.train()
        self.F1.train()
        optimizer_g = optim.SGD(self.params, momentum=0.9, weight_decay=0.0005,
                            nesterov=True)
        optimizer_f = optim.SGD(list(self.F1.parameters()), lr=1.0, momentum=0.9, weight_decay=0.0005,
                            nesterov=True)

        def zero_grad_all():
            optimizer_g.zero_grad()
            optimizer_f.zero_grad()

        param_lr_g = []
        for param_group in optimizer_g.param_groups:
            param_lr_g.append(param_group["lr"])
        param_lr_f = []
        for param_group in optimizer_f.param_groups:
            param_lr_f.append(param_group["lr"])

        criterion = nn.CrossEntropyLoss().cuda()
        all_step = self.args.steps
        data_iter_s = iter(self.source_loader)
        data_iter_t = iter(self.target_loader)
        data_iter_t_unl = iter(self.target_loader_unl)
        len_train_source = len(self.source_loader)
        len_train_target = len(self.target_loader)
        len_train_target_semi = len(self.target_loader_unl)
        for step in range(all_step):
            optimizer_g = inv_lr_scheduler(param_lr_g, optimizer_g, step, init_lr=self.args.lr)
            optimizer_f = inv_lr_scheduler(param_lr_f, optimizer_f, step, init_lr=self.args.lr)

            lr = optimizer_f.param_groups[0]['lr']
            if step % len_train_target == 0:
                data_iter_t = iter(self.target_loader)
            if step % len_train_target_semi == 0:
                data_iter_t_unl = iter(self.target_loader_unl)
            if step % len_train_source == 0:
                data_iter_s = iter(self.source_loader)
            data_t = next(data_iter_t)
            data_t_unl = next(data_iter_t_unl)
            data_s = next(data_iter_s)
            self.im_data_s.resize_(data_s[0].size()).copy_(data_s[0])
            self.gt_labels_s.resize_(data_s[1].size()).copy_(data_s[1])
            self.im_data_t.resize_(data_t[0].size()).copy_(data_t[0])
            self.gt_labels_t.resize_(data_t[1].size()).copy_(data_t[1])
            self.im_data_tu.resize_(data_t_unl[0].size()).copy_(data_t_unl[0])
            zero_grad_all()
            data = torch.cat((self.im_data_s, self.im_data_t), 0)
            target = torch.cat((self.gt_labels_s, self.gt_labels_t), 0)
            output = self.G(data)
            out1 = self.F1(output)
            loss = criterion(out1, target)
            loss.backward(retain_graph=True)
            optimizer_g.step()
            optimizer_f.step()
            zero_grad_all()
            
            output = self.G(self.im_data_tu)
            loss_t = entropy(self.F1, output, self.args.lamda)
            loss_t.backward()
            optimizer_f.step()
            optimizer_g.step()

            log_train = 'S {} T {} Train Ep: {} lr{} \t Loss Classification: {:.6f} Method {}\n'.format(
                self.args.source, self.args.target,
                step, lr, loss.data, self.args.method)
            self.G.zero_grad()
            self.F1.zero_grad()

            if step % self.args.log_interval == 0:
                print(log_train)
            if step % self.args.save_interval == 0 and step > 0:
                self.test(self.target_loader_unl)
                self.G.train()
                self.F1.train()
                if self.args.save_check:
                    print('saving model')
                    torch.save(self.G.state_dict(), os.path.join(self.args.checkpath,
                                                          "G_iter_model_{}_{}_to_{}_step_{}.pth.tar".format(
                                                              self.args.method, self.args.source, self.args.target, step)))
                    torch.save(self.F1.state_dict(),
                           os.path.join(self.args.checkpath, "F1_iter_model_{}_{}_to_{}_step_{}.pth.tar".format(
                               self.args.method, self.args.source, self.args.target, step)))
    
    def test(self, loader):
        self.G.eval()
        self.F1.eval()
        test_loss = 0
        correct = 0
        size = 0
        num_class = len(self.class_list)
        output_all = np.zeros((0, num_class))
        criterion = nn.CrossEntropyLoss().cuda()
        confusion_matrix = torch.zeros(num_class, num_class)
        with torch.no_grad():
            for batch_idx, data_t in enumerate(loader):
                self.im_data_t.resize_(data_t[0].size()).copy_(data_t[0])
                self.gt_labels_t.resize_(data_t[1].size()).copy_(data_t[1])
                feat = self.G(self.im_data_t)
                output1 = self.F1(feat)
                output_all = np.r_[output_all, output1.data.cpu().numpy()]
                size += self.im_data_t.size(0)
                pred1 = output1.data.max(1)[1]  # get the index of the max log-probability
                for t, p in zip(self.gt_labels_t.view(-1), pred1.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
                correct += pred1.eq(self.gt_labels_t.data).cpu().sum()
                test_loss += criterion(output1, self.gt_labels_t) / len(loader)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} F1 ({:.0f}%)\n'.format(
            test_loss, correct, size,
            100. * correct / size))
        return test_loss.data, 100. * float(correct) / size
