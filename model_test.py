# -*- coding:utf-8 -*-
import os
import sys
from itertools import chain
import numpy as np
import torch
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tqdm import tqdm
from get_data import setup_seed
from model_train import device
from models import SSTAFormer
import matplotlib
import matplotlib.pyplot as plt
print(matplotlib.get_backend())
import statsmodels.api as sm
import pandas as pd

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams.update({"font.size": 28})
plt.rcParams['axes.unicode_minus'] = False

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def test(args, Dte, scaler, path):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print('loading models...')

    model = SSTAFormer(args).to(device)
    total_params = count_parameters(model)
    print(f"Total number of parameters: {total_params}")
    model.load_state_dict(torch.load(current_dir + '/models/SSTAFormer.pkl')['model'], strict=False)
    model.eval()
    print('predicting...')
    ys = [[] for i in range(args.input_size)]
    preds = [[] for i in range(args.input_size)]
    for graph in tqdm(Dte):
        graph = graph.to(device)
        _pred, targets = model(graph)
        targets = np.array(targets.data.tolist())
        for i in range(args.input_size):
            target = targets[:, i, :]
            target = list(chain.from_iterable(target))
            ys[i].extend(target)
        for i in range(_pred.shape[0]):
            pred = _pred[i]
            pred = list(chain.from_iterable(pred.data.tolist()))
            preds[i].extend(pred)


    ys, preds = np.array(ys).T, np.array(preds).T
    ys = scaler.inverse_transform(ys).T
    preds = scaler.inverse_transform(preds).T

    #
    errors = (ys.T)[1:,-1] - (preds.T)[1:,-1]
    df_errors = pd.DataFrame(errors)
    df_real_value = pd.DataFrame((ys.T)[1:-1])
    df_pred = pd.DataFrame((preds.T)[1:,-1])


    excel_path_pred = 'pred.xlsx'
    excel_path = 'real_value.xlsx'
    excel_path_errors = 'errors.xlsx'
    df_pred.to_excel(excel_path_pred, index=False, engine='openpyxl')
    df_errors.to_excel(excel_path_errors, index=False, engine='openpyxl')
    df_real_value.to_excel(excel_path, index=False, engine='openpyxl')




    error = []
    for i in range(len(ys)):
        error.append(preds[i] - ys[i])

    print(preds.shape)
    mses, rmses, maes, sees = [], [], [], []
    for ind, (y, pred) in enumerate(zip(ys, preds), 0):
        if ind == 7:
            print('--------------------------------')
            print('r2:', get_r2(y, pred))
            print('rmse:', get_rmse(y, pred))
            print('mae:', get_mae(y, pred))
            print('see:', calculate_standard_error(y, pred))

            plt.figure(figsize=(12, 15))
            plt.plot(y, label='y_true', color='red')
            plt.plot(pred, label='y_pred', color='blue')
            # plt.plot(pred,  color='blue')
            plt.title(f'SSTAFormer')
            plt.xlabel('Samples')
            plt.ylabel('Output Value')
            plt.legend(handles=[])

            ax = plt.gca()

            ax.spines['top'].set_linewidth(3)
            ax.spines['right'].set_linewidth(3)
            ax.spines['bottom'].set_linewidth(3)
            ax.spines['left'].set_linewidth(3)

            plt.show()

            mses.append(get_mse(y, pred))
            rmses.append(get_rmse(y, pred))
            maes.append(get_mae(y, pred))
            sees.append(calculate_standard_error(y, pred))
            print('--------------------------------')


def get_r2(y, pred):
    return r2_score(y, pred)


def get_mae(y, pred):
    return mean_absolute_error(y, pred)


def get_mse(y, pred):
    return mean_squared_error(y, pred)


def get_rmse(y, pred):
    return np.sqrt(mean_squared_error(y, pred))


def calculate_standard_error(x, y):
    x = sm.add_constant(x)
    model = sm.OLS(y, x)
    results = model.fit()
    standard_error = np.sqrt(results.mse_resid)
    return standard_error
