import os
import sys

import argparse
import logging

from torch import optim
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from thop import profile, clever_format

from networks import *
from losses import LabelSmoothingCrossEntropy, UnilateralMse, WarmUpLoss, LabelSmoothingBCEWithLogitsLoss
from common import *
from datasets import get_dvsg_data, get_dvsc10_data, get_nmnist_data

import warnings
warnings.filterwarnings("ignore")

scaler = GradScaler()

torch.backends.cudnn.benchmark = True
torch.set_num_threads(64)

parser = argparse.ArgumentParser()
parser.add_argument('model', type=str)
parser.add_argument('--epochs', type=int, default=300)

parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--device', type=int, default=-1)

parser.add_argument('--step', type=int, default=16)
parser.add_argument('--encode', type=str, default='direct')
parser.add_argument('--node', type=str, default='HTGLIFNode')
parser.add_argument('--thresh', type=float, default=.5)
parser.add_argument('--decay', type=float, default=1.)

parser.add_argument('--suffix', type=str, default='')
parser.add_argument('--infer_every', type=int, default=1)
parser.add_argument('--warmup', type=int, default=5)
parser.add_argument('--optim', type=str, default='AdamW')
parser.add_argument('--loss', type=str, default='mse')

parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=1e-4)
# parser.add_argument('--grad-clip', type=float, default=5.)
parser.add_argument('--lr_min', type=float, default=1e-5)
parser.add_argument('--resume', type=str, default='')
parser.add_argument('--base_dir', type=str, default='/path/to/checkpoint')
parser.add_argument('--warm-up', type=int, default=0)
parser.add_argument('--save', type=bool, default=False)
args = parser.parse_args()

print(args)
torch.autograd.set_detect_anomaly(True)
# DEBUG
# torch.autograd.set_detect_anomaly(True)

CKPT_DIR = args.base_dir
# CKPT_DIR = './ckpt'
fname = '%s_%s_%s_%s_%d_%s_%s_%s_%.2f.pt' % (args.model, args.loss, args.suffix, args.dataset, args.step, args.encode,
                                             'finetune' if args.resume else '', args.node, args.thresh)
flog = fname + '.txt'
# log = open(os.path.join(CKPT_DIR, flog), 'w')

print(fname)

device = torch.device("cuda:%d" % args.device if args.device >= 0 else "cpu")
best_acc = 0.

num_classes = 10


if args.dataset == 'dvsg':
    train_loader, test_loader, _, _ = get_dvsg_data(args.batch_size, args.step)
    num_classes = 11
elif args.dataset == 'dvsc10':
    train_loader, test_loader, _, _ = get_dvsc10_data(args.batch_size, args.step)
elif args.dataset == 'nmnist':
    train_loader, test_loader, _, _ = get_nmnist_data(args.batch_size, args.step)
else:
    raise NotImplementedError

model = eval(args.model)(step=args.step,
                         dataset=args.dataset,
                         batch_size=args.batch_size,
                         num_classes=num_classes,
                         device=device,
                         encode_type=args.encode,
                         node=args.node,
                         threshold=args.thresh,
                         decay=args.decay).to(device)

# optimizer = optim.SGD(
#     [{'params': [param for name, param in model.named_parameters() if 'floyed' not in name]}],
#     lr=args.lr,
#     weight_decay=args.weight_decay
# )

if args.optim == 'AdamW':
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
elif args.optim == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
else:
    raise NotImplementedError
#
# scheduler = optim.lr_scheduler.CosineAnnealingLR(
#     optimizer, T_max=float(args.epochs), eta_min=args.lr_min, last_epoch=-1
# )

# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [30, 60, 90], gamma=0.1)

if args.loss == 'ce':
    criterion = nn.CrossEntropyLoss()
elif args.loss == 'bce':
    criterion = nn.BCEWithLogitsLoss()
elif args.loss == 'mse':
    criterion = UnilateralMse(1.)
elif args.loss == 'sce':
    criterion = LabelSmoothingCrossEntropy()
elif args.loss == 'sbce':
    criterion = LabelSmoothingBCEWithLogitsLoss()
elif args.loss == 'umse':
    criterion = UnilateralMse(.5)
elif args.loss == 'mixed':
    criterion = WarmUpLoss()
else:
    raise NotImplementedError

epoch_start = 0


def train():
    if args.warm_up != 0:
        model.set_warm_up(True)
        # criterion = nn.CrossEntropyLoss()
    for epoch in range(epoch_start, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        # warm-up
        if args.warm_up != 0 and epoch == args.warm_up:
            model.set_warm_up(False)
            # criterion = UnilateralMse(1.)

        model.train()
        print("[EPOCH]: %d/%d" % (epoch, args.epochs))
        loss_tot = AverageMeter()
        loss_sum = AverageMeter()
        acc_tot = AverageMeter()
        acc_sum = AverageMeter()

        # if epoch < 50:
        #     model.set_ltd(False)
        # else:
        #     model.set_ltd(True)

        for idx, data in enumerate(train_loader):
            # model train
            images = data[0].float().to(device)
            labels = data[1].to(device)
            # print(images.shape, labels.shape)

            optimizer.zero_grad()

            # with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels) if args.loss != 'mixed' else criterion(outputs, labels, epoch - epoch_start)
            # scaler.scale(loss).backward()
            loss.backward()
            # nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)

            optimizer.step()

            acc, = accuracy(outputs, labels, topk=(1,))
            loss_tot.update(loss.item(), outputs.shape[0])
            loss_sum.update(loss.item(), outputs.shape[0])
            acc_tot.update(acc, outputs.shape[0])
            acc_sum.update(acc, outputs.shape[0])

            # logging
            if idx % 1 == 0:
                # log.write('[train]:%d, loss:%.3f, accuracy:%.3f, m_lr:%.5f\n'
                #                % (idx, loss_tot.avg, acc_tot.avg,
                #                   optimizer.param_groups[0]['lr']))
                print('[train]:%d, loss:%.3f, accuracy:%.3f, m_lr:%.5f, '
                      % (idx, loss_tot.avg, acc_tot.avg,
                         optimizer.param_groups[0]['lr']), end='\r')
            if idx % 10 == 0:
                loss_tot.reset()
                acc_tot.reset()

        print('[train total]:loss:%.3f, accuracy:%.3f, m_lr:%.5f'
              % (loss_sum.avg, acc_sum.avg,
                 optimizer.param_groups[0]['lr']), end='\r')

        print('', end='\n')
        if epoch % args.infer_every == 0:
            infer(epoch)

        # scheduler.step()


def infer(epoch):
    global best_acc
    model.eval()
    loss_tot = AverageMeter()
    acc_tot = AverageMeter()
    for idx, data in enumerate(test_loader):
        # model train
        with torch.no_grad():
            images = data[0].float().to(device)
            labels = data[1].to(device)
            # print(images.shape, labels.shape)
            # with autocast():
            outputs = model(images).detach()
            loss = criterion(outputs, labels) if args.loss != 'mixed' else criterion(outputs, labels, epoch - epoch_start)

            acc, = accuracy(outputs, labels, topk=(1,))
            loss_tot.update(loss.item(), outputs.shape[0])
            acc_tot.update(acc, outputs.shape[0])
            # print(model.get_fire_rate())
            # fire_rate = model.get_fire_rate()
            # logging
    # if idx % 1 == 0:
    s = '[I]:%d,ls:%.2f,acc:%.2f,' % (idx, loss_tot.avg, acc_tot.avg)
    s = s + 'fr:' + ''.join(['{:.2f},'.format(i) for i in model.get_fire_rate()])
        # + 'th:' + ''.join(['{:.2f},'.format(i) for i in model.get_threshold()])
    # + 'dy:' + ''.join(['{:.2f}, '.format(i) for i in model.get_decay()])
    print(s, end='\r')

    print('', end='\n')
    if best_acc < acc_tot.avg:
        best_acc = acc_tot.avg
        ckpt = {
            'network': model.state_dict(),
            'epoch': epoch + 1,
            'optimizer': optimizer.state_dict(),
            # 'scheduler': scheduler.state_dict(),
            'best_acc': best_acc,
        }
        if args.save:
            save(ckpt)
            print('\033[0;32;40m[SAVE]\033[0m %.5f' % best_acc)
        else:
            print('\033[0;32;40m[BEST]\033[0m %.5f' % best_acc)


def save(ckpt):
    if not os.path.exists(CKPT_DIR):
        os.mkdir(CKPT_DIR)
    torch.save(ckpt, os.path.join(CKPT_DIR, fname))


def load(fname):
    global model, scheduler, epoch_start, best_acc
    ckpt = torch.load(fname, map_location=device)
    print('[best accuracy]: %f' % ckpt['best_acc'])

    model.load_state_dict(ckpt['network'], strict=False)
    # optimizer.load_state_dict(ckpt['optimizer'])
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=float(args.epochs), eta_min=args.lr_min, last_epoch=ckpt['epoch']
    # )
    print(ckpt['epoch'])
    epoch_start = ckpt['epoch']
    best_acc = 0.  # ckpt['best_acc']


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr
    if hasattr(args, 'warmup') and epoch < args.warmup:
        lr = lr / (args.warmup - epoch)
    # elif not args.disable_cos:
    else:
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup) / (args.epochs - args.warmup)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # lr = args.lr * (0.1 ** (epoch // 80))
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr


if __name__ == '__main__':
    # model.set_warm_up(True)
    # if args.dataset == 'imnet':
    #     inputs = torch.rand(1, 3, 224, 224, device=device)
    # elif args.dataset == 'cifar10':
    #     inputs = torch.rand(1, 3, 32, 32, device=device)
    # elif args.dataset == 'fashion' or 'mnist':
    #     inputs = torch.rand(1, 1, 28, 28, device=device)
    # else:
    #     raise NotImplementedError
    # model.eval()
    # flops, params = profile(model, inputs=(inputs,))
    # flops, params = clever_format([flops, params], '%.3f')
    # model.set_warm_up(False)

    # model.train()
    # print('[FLOPS] {}, PARAMS {}'.format(flops, params))

    if args.resume != '':
        load(args.resume)
        infer(0)
    train()
    print('[BEST accuracy] {}'.format(best_acc))
