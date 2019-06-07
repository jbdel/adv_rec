import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import utils

from dataset import Dictionary, VQAFeatureDataset
import base_model
from train import train, evaluate
import utils
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='baseline0_newatt')
    parser.add_argument('--output', type=str, default='saved_models/exp0')
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--eval', type=bool, default=False)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--reconstruction', type=bool, default=False)
    parser.add_argument('--gamma_r', type=float, default=0.5)
    parser.add_argument('--gamma_a', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--dropout_hid', type=float, default=0.0)
    parser.add_argument('--early_stop', type=int, default=5)
    parser.add_argument('--adv_mode', type=str, default="wgan", choices=['wgan','waae'])
    parser.add_argument('--penality', type=int, default=10)
    parser.add_argument('--critic', type=int, default=5)
    parser.add_argument('--load_cpu', type=bool, help='put float32 in memory directly', default=False)
    parser.add_argument('--adv', type=int, default=1)
    #resnet
    parser.add_argument('--size', type=int, default=224)


    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    utils.create_dir(args.output)
    logger = utils.Logger(os.path.join(args.output, 'log.txt'))

    dictionary = Dictionary.load_from_file('data/dictionary.pkl')

    if not args.eval:
        train_dset = VQAFeatureDataset('train36', dictionary, dataroot=args.data_root, size=args.size,
                                       load_cpu=args.load_cpu,
                                       )

    eval_dset = VQAFeatureDataset('val36', dictionary, dataroot=args.data_root, size=args.size,
                                  load_cpu=args.load_cpu,
                                  )

    batch_size = args.batch_size
    constructor = 'build_%s' % args.model

    model = getattr(base_model, constructor)(eval_dset, args.num_hid,
                                             args.reconstruction,
                                             size=args.size,
                                             dropout_hid=args.dropout_hid,
                                             gamma_r=args.gamma_r,
                                             adv_mode=args.adv_mode,
                                             logger=logger).cuda()

    model.w_emb.init_embedding('data/glove6b_init_300d.npy')
    model = (model).cuda()

    if not args.eval:
        train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=1)

    eval_loader =  DataLoader(eval_dset, batch_size, shuffle=True, num_workers=1)
    if args.eval:
        evaluate(model, eval_loader, args.output, args.ckpt)
    else:
        train(model, train_loader, eval_loader,
              args.epochs,
              args.batch_size,
              args.output,
              args.reconstruction,
              args.lr,
              args.gamma_r,
              args.gamma_a,
              args.size,
              args.early_stop,
              args.dropout_hid,
              args.adv_mode,
              args.penality,
              args.critic,
              args.adv,
              args.ckpt,
              logger)


