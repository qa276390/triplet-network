from __future__ import print_function
import argparse
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from triplet_mnist_loader import MNIST_t
from triplet_image_loader import TripletImageLoader
from triplet_image_loader import TripletEmbedLoader
from tripletnet import Tripletnet
from visdom import Visdom
import numpy as np
import time

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 100)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--margin', type=float, default=0.2, metavar='M',
                    help='margin for triplet loss (default: 0.2)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='TripletNet', type=str,
                    help='name of experiment')
parser.add_argument('--base-path', default='./data/polyvore_outfits/nondisjoint/', type=str,
                    help='base path for the data')
parser.add_argument('--emb-size', type=int, default=64, metavar='M',
                    help='embedding size')
parser.add_argument('--rand-cat', action='store_true', default=False,
                    help='randomly in concat order')
parser.add_argument('--cnn', action='store_true', default=False,
                    help='using cnn model')
best_acc = 0

class SimpleCustomBatch:
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.img1 = torch.stack(transposed_data[0], 0)
        self.img2 = torch.stack(transposed_data[1], 0)
        self.img3 = torch.stack(transposed_data[2], 0)

    def pin_memory(self):
        self.img1 = self.img1.pin_memory()
        self.img2 = self.img2.pin_memory()
        self.img3 = self.img3.pin_memory()
        return self

def collate_wrapper(batch):
    return SimpleCustomBatch(batch)

class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=64)
        self.conv2 = nn.Conv1d(64, 72, kernel_size=2)
        self.conv3 = nn.Conv1d(72, 72, kernel_size=2)
        self.conv4 = nn.Conv1d(72, 80, kernel_size=2)
        self.conv1_drop = nn.Dropout()
        self.fc1 = nn.Linear(320, 120)
        self.fc2 = nn.Linear(120, 50)
        self.fc3 = nn.Linear(50, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        #x = F.relu(F.max_pool1d(self.conv2(x), 3))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        #x = F.relu(F.max_pool1d(self.conv1_drop(self.conv4(x)), 3))
        x = F.relu(self.conv1_drop(self.conv4(x)))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def main():
    print('pid:', os.getpid())
    global args, best_acc
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if(not args.cuda):
        print('no cuda!')
    else:
        print('we\'ve got cuda!')
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    global plotter 
    plotter = VisdomLinePlotter(env_name=args.name)

    ######################
    base_path = args.base_path
    embed_size = args.emb_size
    ######################
    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
    print('loading training data...')
    m = 'train'
    #m = 'test'
    train_loader = torch.utils.data.DataLoader(
        TripletEmbedLoader(args,base_path, m+'_embed_index.csv', 'small_train'+'.json', 
                            'train', m+'_embeddings.pt'),
        batch_size=args.batch_size, shuffle=True, collate_fn=collate_wrapper, **kwargs)
    print('loading testing data...')
    test_loader = torch.utils.data.DataLoader(
        TripletEmbedLoader(args, base_path, 'test_embed_index.csv', 
        'test.json', 'train', 'test_embeddings.pt')
        ,
        batch_size=args.batch_size, shuffle=True, collate_fn=collate_wrapper, **kwargs)
    class Net(nn.Module):
        def __init__(self, embed_size):
            super(Net, self).__init__()
            self.nfc1 = nn.Linear(embed_size * 2, 480)
            self.nfc2 = nn.Linear(480, 320)
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, 1)

        def forward(self, x):
            x = F.relu(self.nfc1(x))
            x = F.relu(self.nfc2(x))
            x = F.dropout(x, training=self.training)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            return self.fc2(x)

    if(args.cnn):
        model = CNNNet()
    else:
        model = Net(embed_size)
    #model = CNNNet()
    if(args.cuda):
        model.cuda()
    tnet = Tripletnet(model, args)
    print('net built.')
    if args.cuda:
        tnet.cuda()
    print('tnet.cuda()')
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            tnet.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    criterion = torch.nn.MarginRankingLoss(margin = args.margin)
    optimizer = optim.SGD(tnet.parameters(), lr=args.lr, momentum=args.momentum)
    #optimizer = optim.Adam(tnet.parameters(), lr=args.lr)

    n_parameters = sum([p.data.nelement() for p in tnet.parameters()])
    print('  + Number of params: {}'.format(n_parameters))

    print('start training!') 
    for epoch in range(1, args.epochs + 1):
        # train for one epoch
        #start_time = time.time()
        train(train_loader, tnet, criterion, optimizer, epoch)
        #print("------- train: %s seconds ---" % (time.time()-start_time))
        # evaluate on validation set
        acc = test(test_loader, tnet, criterion, epoch)

        # remember best acc and save checkpoint
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': tnet.state_dict(),
            'best_prec1': best_acc,
        }, is_best)

def train(train_loader, tnet, criterion, optimizer, epoch):
    losses = AverageMeter()
    accs = AverageMeter()
    emb_norms = AverageMeter()

    # switch to train mode
    tnet.train()
    for batch_idx, sample in enumerate(train_loader):
    #for batch_idx, (data1, data2, data3) in enumerate(train_loader):
        #print('img1 is_pinned:', sample.img1.is_pinned())
        data1 = sample.img1
        data2 = sample.img2
        data3 = sample.img3
        if args.cuda:
            data1, data2, data3 = data1.cuda(non_blocking=True), data2.cuda(non_blocking=True), data3.cuda(non_blocking=True)
        data1, data2, data3 = Variable(data1), Variable(data2), Variable(data3)

        #print('a/data1:', data1.is_pinned())
        #print('a/data2:', data2.shape)
        #print('a/data3:', data3.shape)
        #start_time = time.time()
        # compute output
        dista, distb, embedded_x, embedded_y, embedded_z = tnet(data1, data2, data3)
        #print("----- compute: %s seconds ---" % (time.time()-start_time))
        # 1 means, dista should be larger than distb
        target = torch.FloatTensor(dista.size()).fill_(1)
        if args.cuda:
            target = target.cuda()
        target = Variable(target)
        
        loss_triplet = criterion(dista, distb, target)
        loss_embedd = embedded_x.norm(2) + embedded_y.norm(2) + embedded_z.norm(2)
        loss = loss_triplet + 0.001 * loss_embedd

        # measure accuracy and record loss
        acc = accuracy(dista, distb)
        losses.update(loss_triplet.data, data1.size(0))
        accs.update(acc, data1.size(0))
        emb_norms.update(loss_embedd.data/3, data1.size(0))

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{}]\t'
                  'Loss: {:.4f} ({:.4f}) \t'
                  'Acc: {:.2f}% ({:.2f}%) \t'
                  'Emb_Norm: {:.2f} ({:.2f})'.format(
                epoch, batch_idx * len(data1), len(train_loader.dataset),
                losses.val, losses.avg, 
                100. * accs.val[0], 100. * accs.avg[0], emb_norms.val, emb_norms.avg))
    # log avg values to somewhere
    #plotter.plot('acc', 'train', epoch, accs.avg[0])
    #plotter.plot('loss', 'train', epoch, losses.avg)
    #plotter.plot('emb_norms', 'train', epoch, emb_norms.avg)

def test(test_loader, tnet, criterion, epoch):
    losses = AverageMeter()
    accs = AverageMeter()

    # switch to evaluation mode
    tnet.eval()
    for batch_idx, sample in enumerate(test_loader):
        data1 = sample.img1
        data2 = sample.img2
        data3 = sample.img3
        if args.cuda:
            data1, data2, data3 = data1.cuda(), data2.cuda(), data3.cuda()
        data1, data2, data3 = Variable(data1), Variable(data2), Variable(data3)

        # compute output
        dista, distb, _, _, _ = tnet(data1, data2, data3)
        target = torch.FloatTensor(dista.size()).fill_(1)
        if args.cuda:
            target = target.cuda()
        target = Variable(target)
        test_loss =  criterion(dista, distb, target).data

        # measure accuracy and record loss
        acc = accuracy(dista, distb)
        accs.update(acc, data1.size(0))
        losses.update(test_loss, data1.size(0))      

    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
        losses.avg, 100. * accs.avg[0]))
    #plotter.plot('acc', 'test', epoch, accs.avg[0])
    #plotter.plot('loss', 'test', epoch, losses.avg)
    return accs.avg[0]

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/"%(args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/'%(args.name) + 'model_best.pth.tar')

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array(torch.stack([x,x])), Y=np.array(torch.stack([y,y])), env=self.env, opts=dict(
                legend=[split_name],
                title=var_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update='append')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(dista, distb):
    margin = 0
    is_cuda = dista.is_cuda
    pred = (dista - distb - margin).cpu().data
    acc = (1.0* int((pred > 0).sum()))/ (1.0* dista.size()[0])
    acc = torch.from_numpy(np.array([acc], np.float32))
    if(is_cuda):
        acc = acc.cuda()
    return Variable(acc)
    #return (pred > 0).sum()*1.0/dista.size()[0]

if __name__ == '__main__':
    main()    
