# -*- coding: utf-8 -*-
import torch


def pearson(x, y):
    x_mean = torch.mean(x, 1).unsqueeze(1)
    y_mean = torch.mean(y, 1).unsqueeze(1)
    x_norm = x - x_mean
    y_norm = y - y_mean
    coef_pearson = torch.sum(x_norm * y_norm, 1) / (torch.sqrt(torch.sum(x_norm ** 2, 1)) * torch.sqrt(torch.sum(y_norm ** 2, 1)))
    coef_pearson = (coef_pearson + 1) / 2
    return coef_pearson.unsqueeze(1)
