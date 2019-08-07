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
from tqdm import tqdm
import random

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
parser.add_argument('--pred', action='store_true', default=False,
                    help='prediction')
parser.add_argument('--binary-classify', action='store_true', default=False,
                    help='only positive sample')
parser.add_argument('--multiply', action='store_true', default=False,
                    help='element-wise multiply')
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

class bin_SimpleCustomBatch:
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.catemb = torch.stack(transposed_data[0], 0)
        self.label = torch.stack(transposed_data[1], 0)

    def pin_memory(self):
        self.catemb = self.catemb.pin_memory()
        self.label = self.label.pin_memory()
        return self

def collate_wrapper(batch):
    return SimpleCustomBatch(batch)

def bin_collate_wrapper(batch):
    return bin_SimpleCustomBatch(batch)

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
    if args.pred:
        np.random.seed(args.seed)  # Numpy module.
        random.seed(args.seed)  # Python random module.
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
    ######################
    base_path = args.base_path
    embed_size = args.emb_size
    ######################
    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
   
    m = 'train'
    #m = 'test'
    #trainortest = 'test'
    trainortest = 'small_train'
    if args.binary_classify:
        coll = bin_collate_wrapper
    else:
        coll = collate_wrapper
    if not args.pred:
        print('loading training data...')
        train_loader = torch.utils.data.DataLoader(
            TripletEmbedLoader(args,base_path, m+'_embed_index.csv', trainortest +'.json', 
                                'train', m+'_embeddings.pt'),
            batch_size=args.batch_size, shuffle=True, collate_fn=coll, **kwargs)
    print('loading testing data...')
    if args.pred:
        shuff = False
    else:
        shuff = True
    test_loader = torch.utils.data.DataLoader(
        TripletEmbedLoader(args, base_path, 'test_embed_index.csv', 
        'test.json', 'train', 'test_embeddings.pt')
        ,
        batch_size=args.batch_size, shuffle=shuff, collate_fn=coll, **kwargs)
    class Net(nn.Module):
        def __init__(self, embed_size):
            super(Net, self).__init__()
            self.nfc1 = nn.Linear(embed_size * 2, 480)
            self.nfc2 = nn.Linear(480, 320)
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, 1)
            if args.binary_classify:
                self.out = nn.Sigmoid()
        def forward(self, x):
            x = F.relu(self.nfc1(x))
            x = F.dropout(x, p=0.5, training=self.training)
            x = F.relu(self.nfc2(x))
            x = F.dropout(x, p=0.75, training=self.training)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, p=0.75, training=self.training)
            if args.binary_classify:
                x = self.fc2(x)
                return self.out(x)
            return self.fc2(x)

    if(args.cnn):
        model = CNNNet()
    else:
        model = Net(embed_size)
    #model = CNNNet()
    if(args.cuda):
        model.cuda()
    if(args.binary_classify):
        tnet = model
    else:
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


    if not args.pred:
        cudnn.benchmark = True
    if args.binary_classify:
        criterion = torch.nn.BCELoss()
    else:
        criterion = torch.nn.MarginRankingLoss(margin = args.margin)
    #optimizer = optim.SGD(tnet.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adam(tnet.parameters(), lr=args.lr)

    n_parameters = sum([p.data.nelement() for p in tnet.parameters()])
    print('  + Number of params: {}'.format(n_parameters))

    if args.pred:
        print('testing...')
        acc = test(test_loader, tnet, criterion, 0)
        #exit(1)
        print('predicting...')
        predict(test_loader, tnet)
        exit(1)
    
    print('start training!') 
    for epoch in range(1, args.epochs + 1):
        # train for one epoch
        #start_time = time.time()
        if args.binary_classify:
            bin_train(train_loader, tnet, criterion, optimizer, epoch)
            acc = bin_test(test_loader, tnet, criterion, epoch)
        else:
            train(train_loader, tnet, criterion, optimizer, epoch)
            acc = test(test_loader, tnet, criterion, epoch)
        #print("------- train: %s seconds ---" % (time.time()-start_time))
        # evaluate on validation set

        # remember best acc and save checkpoint
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': tnet.state_dict(),
            'best_prec1': best_acc,
        }, is_best)

def bin_train(train_loader, tnet, criterion, optimizer, epoch):
    losses = AverageMeter()
    accs = AverageMeter()
    emb_norms = AverageMeter()

    # switch to train mode
    tnet.train()
    for batch_idx, sample in enumerate(train_loader):
    #for batch_idx, (data1, data2, data3) in enumerate(train_loader):
        #print('img1 is_pinned:', sample.img1.is_pinned())
        data1 = sample.catemb
        data2 = sample.label
        if args.cuda:
            data1, data2 = data1.cuda(non_blocking=True), data2.cuda(non_blocking=True)
        data1, data2 = Variable(data1), Variable(data2)

        #start_time = time.time()
        # compute output
        y_pred = tnet(data1)
        #print("----- compute: %s seconds ---" % (time.time()-start_time))
        # 1 means, dista should be larger than distb
        #target = torch.FloatTensor(dista.size()).fill_(1)
        #if args.cuda:
        #    target = target.cuda()
        #target = Variable(target)
        
        loss = criterion(y_pred, data2)
        #loss_embedd = embedded_x.norm(2) + embedded_y.norm(2) + embedded_z.norm(2)
        #loss = loss_triplet + 0.001 * loss_embedd

        # measure accuracy and record loss
        acc = bin_acc(y_pred, data2)
        losses.update(loss.data, data1.size(0))
        accs.update(acc, data1.size(0))
        #emb_norms.update(loss_embedd.data/3, data1.size(0))

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
                #100. * accs.val[0], 100. * accs.avg[0], emb_norms.val, emb_norms.avg))
                100. * accs.val, 100. * accs.avg, emb_norms.val, emb_norms.avg))
    # log avg values to somewhere
    plotter.plot('acc', 'train', epoch, accs.avg.cpu())
    plotter.plot('loss', 'train', epoch, losses.avg.cpu())
    #plotter.plot('emb_norms', 'train', epoch, emb_norms.avg.cpu())


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
    plotter.plot('acc', 'train', epoch, accs.avg.cpu())
    plotter.plot('loss', 'train', epoch, losses.avg.cpu())
    plotter.plot('emb_norms', 'train', epoch, emb_norms.avg.cpu())

def bin_test(test_loader, tnet, criterion, epoch):
    losses = AverageMeter()
    accs = AverageMeter()

    # switch to evaluation mode
    tnet.eval()
    #print(len(test_loader.dataset))
    for batch_idx, sample in enumerate(test_loader):
        #print('batch_idx:', batch_idx)
        data1 = sample.catemb
        data2 = sample.label
        if args.cuda:
            data1, data2 = data1.cuda(non_blocking=True), data2.cuda(non_blocking=True)
        data1, data2 = Variable(data1), Variable(data2)

        #start_time = time.time()
        # compute output
        y_pred = tnet(data1)
        
        #target = torch.FloatTensor(dista.size()).fill_(1)
        #if args.cuda:
        #    target = target.cuda()
        #target = Variable(target)
        test_loss = criterion(y_pred, data2).data
        #loss_embedd = embedded_x.norm(2) + embedded_y.norm(2) + embedded_z.norm(2)
        #loss = loss_triplet + 0.001 * loss_embedd

        # measure accuracy and record loss
        acc = bin_acc(y_pred, data2)
        losses.update(test_loss.data, data1.size(0))
        accs.update(acc, data1.size(0))

    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
        #losses.avg, 100. * accs.avg[0]))
        losses.avg, 100. * accs.avg))
    #plotter.plot('acc', 'test', epoch, accs.avg[0].cpu())
    plotter.plot('acc', 'test', epoch, accs.avg.cpu())
    plotter.plot('loss', 'test', epoch, losses.avg.cpu())
    #return accs.avg[0]
    return accs.avg
def test(test_loader, tnet, criterion, epoch):
    losses = AverageMeter()
    accs = AverageMeter()

    # switch to evaluation mode
    tnet.eval()
    #print(len(test_loader.dataset))
    for batch_idx, sample in enumerate(test_loader):
        #print('batch_idx:', batch_idx)
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
    plotter.plot('acc', 'test', epoch, accs.avg[0].cpu())
    plotter.plot('loss', 'test', epoch, losses.avg.cpu())
    return accs.avg[0]

def predict(test_loader, tnet):
    # define file name
    query_embedding_path = os.path.join(args.base_path, 'query_embeddings.pt')
    query_detail_path = os.path.join(args.base_path, 'query.txt')
    html_result_path = './results/result1.html'
    query_image_path = './data/polyvore_outfits/query_images'
    test_image_path = './data/polyvore_outfits/images'

    # switch to evaluation mode
    tnet.eval()

    indexlist = test_loader.dataset.indexlist
    # generate type masks
    type2idxs = {key: torch.LongTensor(len(indexlist)).zero_() for key in indexlist['type'].unique()}
    for index, row in tqdm(indexlist.iterrows(), total=len(indexlist)):
        type2idxs[row['type']][index] = 1

    # load test embedding
    test_embed = test_loader.dataset.emb_tensor[:,-1,:]

    # load query embedding
    query_embed = torch.load(query_embedding_path)
    assert query_embed.size()[1] == test_embed.size()[1]

    # read query details
    with open(query_detail_path, 'r') as f:
        lines = f.readlines()
    query_details = []
    for i, line in enumerate(lines):
        img, cat = line.strip().split()
        query_details.append((img, cat))

    html_writer = open(html_result_path, 'w')
    # for each query, search for
    for i, query in enumerate(tqdm(query_embed)):
        query_type = query_details[i][1]
        query = query.view(1,-1)
        for test_type, test_mask in type2idxs.items():
            mask = test_mask.nonzero().view(-1)
            dist_rank = [0] * test_embed[mask].size()[0]


            tests = test_embed[mask]
            querys = torch.cat([query]*len(tests))
            #print('tests shape:', tests.shape)
            #print('querys shape:', querys.shape)

            if args.cuda:
                querys, tests = querys.cuda(non_blocking=True), tests.cuda(non_blocking=True)
            query_variable, test_variable = Variable(querys), Variable(tests)
            
            start_time = time.time()
            dists = tnet(query_variable, test_variable)
            #print("--- tnet pred: %s seconds ---" % (time.time()-start_time))
            #print('dists shape:', dists.shape) 
            print('dists:')
            print(dists)
            #start_time = time.time()
            
            for j, test in enumerate(test_embed[mask]):
                """
                test = test.view(1,-1)
                 
                if args.cuda:
                    query, test = query.cuda(non_blocking=True), test.cuda(non_blocking=True)
                query_variable, test_variable = Variable(query), Variable(test)
                start_time = time.time()
                dist = tnet(query_variable, test_variable)
                print("--- tnet prediction: %s seconds ---" % (time.time()-start_time))
                #print(test_mask)
                #print(mask[j])
                #print(indexlist.iloc[int(mask[j])]['image'])
                """    
                dist_rank[j] = (str(indexlist.iloc[int(mask[j])]['image']), dists[j])
            
            #indexlist['image'] = indexlist['image'].astype(str)
            #dist_rank = zip(indexlist.iloc[mask]['images'], dists) 
            
            #start_time = time.time()

            #random sample 300 to sort
            choices = np.random.choice(len(dist_rank), size = 300)
            dist_rank = [dist_rank[j] for j in choices]
            dist_rank.sort(key=lambda k: k[1][0])

            #print("----sort time: %s seconds ---" % (time.time()-start_time))
            showresults(v_html=html_writer, query_folder=query_image_path, answer_folder=test_image_path,
                        query_img=query_details[i][0], img_cat=(query_type, test_type), dist=dist_rank[:10])

def showresults(v_html, query_folder, answer_folder, query_img, img_cat, gt_img=None, dist=None):
    """
    Display results
        
    Inputs:
    img1: (str) query image id
    dist: (tuple) set_id, result image id, and distance
    """

    img_width = str(300)
    v_html.write("<div id=\"image-table\"><table><tr>")
    v_html.write("<td style=\"padding:5px\">")
    v_html.write("<img src=\""+os.path.join('../', query_folder, query_img+'.jpg') + "\" width=\"{}\">".format(img_width))
    v_html.write("<p style=\"text-align:center;font-size:30px;\">{}</p></td>".format(img_cat[0]))

    if gt_img:
        v_html.write("<img src=\""+os.path.join('../', query_folder, gt_img+'.jpg') + "\" width=\"{}\">".format(img_width))

    for img2, _ in dist:
        v_html.write("<td style=\"padding:5px\">")
        v_html.write("<img src=\""+os.path.join('../', answer_folder, img2+'.jpg') + "\" width=\"{}\">".format(img_width))
        v_html.write("<p style=\"text-align:center;font-size:30px;\">{}</p></td>".format(img_cat[1]))
    v_html.write("</tr></table></div>")
    v_html.write("<br>")


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
            self.plots[var_name] = self.viz.line(X=np.column_stack((x,x)), Y=np.column_stack((y,y)), env=self.env, opts=dict            (
                #legend=[split_name],
                title=var_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update='append')
            #self.viz.line(X=np.column_stack(x), Y=np.column_stack(y), env=self.env, win=self.plots[var_name], name=split_name, update='append')

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
def bin_acc(output, target):
    is_cuda = target.is_cuda
    pred = output >= 0.5
    truth = target >= 0.5
    tp = pred.mul(truth).sum(0).float()
    tn = (1 - pred).mul(1 - truth).sum(0).float()
    fp = pred.mul(1 - truth).sum(0).float()
    fn = (1 - pred).mul(truth).sum(0).float()
    acc = (tp + tn).sum() / (tp + tn + fp + fn).sum()
    if(is_cuda):
        acc = acc.cuda()
    return Variable(acc)
if __name__ == '__main__':
    main()    
