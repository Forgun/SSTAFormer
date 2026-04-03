# -*- coding:utf-8 -*-
import os
import sys
from get_data import nn_seq_GraphSAGE
from args import SSTAFormer_args_parser
from model_train import train
from model_test import test
import time


start_time = time.time()

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

args = SSTAFormer_args_parser()
path = os.path.abspath(os.path.dirname(os.getcwd()))


def main():
    graph, Dtr, Val, Dte, scaler = nn_seq_GraphSAGE(args.input_size, args.seq_len,
                                                    args.batch_size, args.output_size)
    print(len(Dtr), len(Val), len(Dte))
    print(graph)
    print(scaler)

    train(args, Dtr, Val, model_type='SSTAFormer', path=path)
    test(args, Dte, scaler, path=path)

    end_time = time.time()
    runtime = end_time - start_time
    print("运行时间: {:.2f} 秒".format(runtime))


if __name__ == '__main__':
    main()
