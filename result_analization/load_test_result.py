import glob
import numpy as np
import shutil
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import json


def get_vals(path, exec_type):
    items = glob.glob(os.path.join(path, exec_type + '*'))
    items = list(map(lambda x: os.path.join(x, 'test_metrics.json'), items))
    return items


def to_map(d):
    dd = [json.loads("\n".join(open(i, "r").readlines())) for i in d]
    return [{int(k): v for k, v in d.items()} for d in dd]


def create_dataset(datasets):
    return [(n, to_map(d)) for n, d in datasets]


"""
def draw_metric(metric_name, datas, plt):
    x = [i for i in range(100)]
    items = {}
    for name, ys in datas:
        y_values = []
        for idx in range(100):
            averages = []
            for f_idx in range(10):
                # print(name, idx, f_idx)
                # print(ys)
                averages.append(ys[f_idx][idx][metric_name])
            y_values.append(np.mean(np.array(averages)))
        items[name] = (x, y_values)

    plt.figure(figsize=(30, 10))
    plt.title(metric_name)
    for n, (x, y) in items.items():
        plt.plot(x, y, label=n)
    plt.legend()
    #plt.show()
    #plt.clf()
"""


def draw_metrics(name_data, figsize=(30, 10)):
    plots = len(name_data)
    f, axes = plt.subplots(plots, 1, figsize=figsize)
    for idx in range(plots):
        name = name_data[idx][0]
        data = name_data[idx][1]
        axe = axes[idx]
        __draw_metric_subplots(name, data, axe)
    plt.legend()
    return plt


def __draw_metric_subplots(metric_name, datas, subplot):
    x = [i for i in range(100)]
    items = {}
    for name, ys in datas:
        y_values = []
        for idx in range(100):
            averages = []
            for f_idx in range(len(ys)):
                if idx in ys[f_idx]:
                    averages.append(ys[f_idx][idx][metric_name])
                else:
                    pass
                    # averages.append(-1.0)
            y_values.append(np.mean(np.array(averages)))
        items[name] = (x, y_values)

    subplot.set_title(metric_name)
    for n, (x, y) in items.items():
        subplot.plot(x, y, label=n)
    # subplot.legend()


def draw_metrics_by_deceases(name_data, figsize=(25, 10)):
    plots = len(name_data)
    f, axes = plt.subplots(plots, 1, figsize=figsize)
    for idx in range(plots):
        name = name_data[idx][1]
        deceases = name_data[idx][0]
        data = name_data[idx][2]
        axe = axes[idx]
        __draw_metric_by_deceases_subplot(deceases, name, data, axe)
    plt.legend(loc='lower left')
    return plt


def __draw_metric_by_deceases_subplot(deceases, metric_name, datas, axes):
    x = [i for i in range(100)]
    items = {}
    for name, ys in datas:
        y_values = []
        for idx in range(100):
            averages = []
            for f_idx in range(len(ys)):
                if idx in ys[f_idx]:
                    averages.append(ys[f_idx][idx][deceases][metric_name])
                else:
                    pass
                    #averages.append(-1.0)
            y_values.append(np.mean(np.array(averages)))
        items[name] = (x, y_values)

    axes.set_title(deceases + " " + metric_name)
    for n, (x, y) in items.items():
        axes.plot(x, y, label=n)
    # axes.legend(loc='lower left')


def get_last_value(dataset, *args):
    res = []
    for (name, data) in dataset:
        max_epoch = max([max(i for i in launch.keys()) for launch in data])
        values = []
        for launch in data:
            if max_epoch in launch:
                node = launch[max_epoch]
                for p in args:
                    node = node[p]
                values.append(node)
        res.append((name, np.mean(np.array(values))))
    return res
