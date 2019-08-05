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
from tqdm import tqdm

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
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
parser.add_argument('--pred', action='store_true', default=False,
                    help='prediction')
best_acc = 0


def main():
    global args, best_acc
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    global plotter 
    plotter = VisdomLinePlotter(env_name=args.name)

    ######################
    base_path = args.base_path
    embed_size = args.emb_size
    ######################
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    if not args.pred:
        train_loader = torch.utils.data.DataLoader(
            TripletEmbedLoader(base_path, 'embed_index.csv', 'test.json', 
                                'train', 'test_embeddings.pt'),
            batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        TripletEmbedLoader(base_path, 'embed_index.csv', 
        'test.json', 'train', 'test_embeddings.pt')
        ,
        batch_size=args.batch_size, shuffle=True, **kwargs)

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
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            return self.fc2(x)

    model = Net(embed_size)
    tnet = Tripletnet(model)
    if args.cuda:
        tnet.cuda()

    if args.pred:
        predict(test_loader, tnet)
        exit(1)

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
    #optimizer = optim.SGD(tnet.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adam(tnet.parameters(), lr=args.lr)

    n_parameters = sum([p.data.nelement() for p in tnet.parameters()])
    print('  + Number of params: {}'.format(n_parameters))

    for epoch in range(1, args.epochs + 1):
        # train for one epoch
        train(train_loader, tnet, criterion, optimizer, epoch)
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
    for batch_idx, (data1, data2, data3) in enumerate(train_loader):
        if args.cuda:
            data1, data2, data3 = data1.cuda(), data2.cuda(), data3.cuda()
        data1, data2, data3 = Variable(data1), Variable(data2), Variable(data3)

        # compute output
        dista, distb, embedded_x, embedded_y, embedded_z = tnet(data1, data2, data3)
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
        losses.update(loss_triplet.data[0], data1.size(0))
        accs.update(acc, data1.size(0))
        emb_norms.update(loss_embedd.data[0]/3, data1.size(0))

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
                100. * accs.val, 100. * accs.avg, emb_norms.val, emb_norms.avg))
    # log avg values to somewhere
    plotter.plot('acc', 'train', epoch, accs.avg)
    plotter.plot('loss', 'train', epoch, losses.avg)
    plotter.plot('emb_norms', 'train', epoch, emb_norms.avg)

def test(test_loader, tnet, criterion, epoch):
    losses = AverageMeter()
    accs = AverageMeter()

    # switch to evaluation mode
    tnet.eval()
    for batch_idx, (data1, data2, data3) in enumerate(test_loader):
        if args.cuda:
            data1, data2, data3 = data1.cuda(), data2.cuda(), data3.cuda()
        data1, data2, data3 = Variable(data1), Variable(data2), Variable(data3)

        # compute output
        dista, distb, _, _, _ = tnet(data1, data2, data3)
        target = torch.FloatTensor(dista.size()).fill_(1)
        if args.cuda:
            target = target.cuda()
        target = Variable(target)
        test_loss =  criterion(dista, distb, target).data[0]

        # measure accuracy and record loss
        acc = accuracy(dista, distb)
        accs.update(acc, data1.size(0))
        losses.update(test_loss, data1.size(0))      

    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
        losses.avg, 100. * accs.avg))
    plotter.plot('acc', 'test', epoch, accs.avg)
    plotter.plot('loss', 'test', epoch, losses.avg)
    return accs.avg

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
            for j, test in enumerate(test_embed[mask]):
                test = test.view(1,-1)
                query_variable, test_variable = Variable(query), Variable(test)
                dist = tnet(query_variable, test_variable)
                dist_rank[j] = (str(indexlist.iloc[mask[j]]['image']), dist)

            dist_rank.sort(key=lambda k: k[1][0][0])
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
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
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
    pred = (dista - distb - margin).cpu().data
    return (pred > 0).sum()*1.0/dista.size()[0]

if __name__ == '__main__':
    main()    
