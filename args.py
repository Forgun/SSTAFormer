# -*- coding:utf-8 -*-
import argparse
import torch

def SSTAFormer_args_parser():

    parser = argparse.ArgumentParser()
    # parser.add_argument('--epochs', type=int, default=23, help='training epochs')#原来批次设置为20,GCN,TCN
    # parser.add_argument('--epochs', type=int, default=20, help='training epochs')#原来批次设置为20,GCN,TCN
    # parser.add_argument('--epochs', type=int, default=10, help='training epochs')#原来批次设置为20,CNN
    parser.add_argument('--epochs', type=int, default=40, help='training epochs')#硫回收
    # parser.add_argument('--epochs', type=int, default=35, help='training epochs')#第二篇的
    # parser.add_argument('--epochs', type=int, default=40, help='training epochs')#
    # parser.add_argument('--epochs', type=int, default=100, help='training epochs')
    parser.add_argument('--input_size', type=int, default=8, help='input dimension')
    # parser.add_argument('--input_size', type=int, default=6, help='input dimension')
    # parser.add_argument('--input_size', type=int, default=5, help='input dimension')
    # parser.add_argument('--input_size', type=int, default=39, help='input dimension')
    # parser.add_argument('--seq_len', type=int, default=128, help='seq len')#原来的,TCN,GCN,CNN模型
    # parser.add_argument('--seq_len', type=int, default=16, help='seq len')
    parser.add_argument('--seq_len', type=int, default=1, help='seq len')#对比实验才注释的硫回收
    parser.add_argument('--output_size', type=int, default=1, help='output dimension')
    parser.add_argument('--num_channels', type=list, default=[32, 32], help='num_channels')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    # parser.add_argument('--batch_size', type=int, default=100, help='batch size')原来
    parser.add_argument('--batch_size', type=int, default=100, help='batch size')
    # parser.add_argument('--batch_size', type=int, default=32, help='batch size')#GCN-based model
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--step_size', type=int, default=150, help='step size')#原来的
    # parser.add_argument('--step_size', type=int, default=10, help='step size')
    # parser.add_argument('--gamma', type=float, default=0.5, help='gamma')#原来的
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma')

    args = parser.parse_args()

    return args
