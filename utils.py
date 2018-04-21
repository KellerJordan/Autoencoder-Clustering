import os

import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
import torchvision


def plot_tensor(img, fs=(8, 8), title=''):
    # preprocess input
    if type(img) == Variable:
        img = img.data
    img = img.cpu()
    if torch.tensor._TensorBase in type(img).__bases__:
        npimg = img.numpy()
    else:
        npimg = img
    if len(npimg.shape) == 4:
        npimg = npimg[0]
    npimg = npimg.transpose(1, 2, 0)
    plt.figure(figsize=fs)
    if npimg.shape[2] > 1:
        plt.imshow(npimg)
    else:
        npimg = npimg.squeeze()
        plt.imshow(npimg, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def plot_batch(samples):
    sample_grid = torchvision.utils.make_grid(samples)
    plot_tensor(sample_grid)

def one_hotify(x, n_classes=10):
    x_long = torch.LongTensor(x)
    x_onehot = torch.sparse.torch.eye(n_classes).index_select(0, x_long)
    return x_onehot
    
def get_argmax(scores):
    val, idx = torch.max(scores, dim=1)
    return idx.data.view(-1).cpu()

def get_accuracy(pred, target):
    correct = torch.sum(pred == target)
    return correct / pred.size(0)

def count_params(model):
    return sum(p.numel() for p in model.parameters())

class Trainer():
    def __init__(self, model, optimizer, criterion,
                 trn_loader, tst_loader,
                 one_hot=False, use_reconstructions=False, use_cuda=False,
                 print_every=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.trn_loader = trn_loader
        self.tst_loader = tst_loader
        
        self.one_hot = one_hot
        self.use_reconstructions = use_reconstructions
        self.use_cuda = use_cuda
        
        self.metrics = {
            'loss': {
                'trn':[],
                'tst':[]
            },
            'accuracy': {
                'trn':[],
                'tst':[]
            },
        }
        self.print_every = print_every
    
    def run(self, epochs):
        print('[*] Training for %d epochs' % epochs)
        for epoch in range(1, epochs+1):
            trn_loss, trn_acc = self.train()
            tst_loss, tst_acc = self.test()
            print('[*] Epoch %d, TrnLoss: %.3f, TrnAcc: %.3f, TstLoss: %.3f, TstAcc: %.3f'
                % (epoch, trn_loss, trn_acc, tst_loss, tst_acc))
            self.metrics['accuracy']['trn'].append(trn_acc)
            self.metrics['accuracy']['tst'].append(tst_acc)
            self.metrics['loss']['trn'].append(trn_loss)
            self.metrics['loss']['tst'].append(tst_loss)
    
    def train(self):
        self.model.train()
        return self.run_epoch(self.trn_loader)
    
    def test(self):
        self.model.eval()
        return self.run_epoch(self.tst_loader, train=False)
    
    def make_var(self, tensor):
        if self.use_cuda:
            tensor = tensor.cuda()
        return Variable(tensor)
    
    def run_epoch(self, loader, train=True):
        n_batches = len(loader)
        cum_loss = 0
        cum_acc = 0
        for i, (X, y) in enumerate(loader):
            X_var = self.make_var(X)
            y_var = self.make_var(one_hotify(y) if self.one_hot else y)

            if self.use_reconstructions:
                scores, reconstructions = self.model(X_var, y_var)
                loss_var = self.criterion(scores, y_var, reconstructions, X_var)
            else:
                scores = self.model(X_var)
                loss_var = self.criterion(scores, y_var)

            if train:
                self.optimizer.zero_grad()
                loss_var.backward()
                self.optimizer.step()

            pred = get_argmax(scores)
            acc = get_accuracy(pred, y)
            loss = loss_var.data[0]
            cum_acc += acc
            cum_loss += loss
            
            if self.print_every is not None and i % self.print_every == 0:
                print('[*] Batch %d, Loss: %.3f, Acc: %.3f' % (i, loss, acc))
        
        return cum_loss / n_batches, cum_acc / n_batches

    def save_checkpoint(self, filename='checkpoint.pth.tar'):
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()}
        torch.save(state, filename)

    def load_checkpoint(self, filename='checkpoint.pth.tar'):
        if os.path.isfile(filename):
            state = torch.load(filename)
            self.model.load_state_dict(state['model'])
            self.optimizer.load_state_dict(state['optimizer'])
        else:
            print('%s not found.' % filename)
