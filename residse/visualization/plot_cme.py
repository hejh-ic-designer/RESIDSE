import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from utils import clear_none_in_lst
import numpy as np
from residse.classes.cost_model.cost_model import CostModelEvaluation


def plot_cme_edp(cmes: list[CostModelEvaluation], save_path = 'default_edp.png'):
    plt.figure(dpi=1000)

    x = [cme.a_buf_size/1024 for cme in clear_none_in_lst(cmes)]
    y = [cme.edp for cme in clear_none_in_lst(cmes)]

    # 创建散点图
    plt.scatter(x, y, color='blue', marker='o')  # 'o'代表圆圈，color设置点的颜色


    plt.title('EDP -- buffer size')
    plt.xlabel('activation buffer size (KB)')
    plt.ylabel('EDP')
    plt.savefig(save_path, dpi=1000)


def plot_cme_ema(cmes: list[CostModelEvaluation], save_path = 'default_ema.png'):
    plt.figure(dpi=1000)

    x = [cme.a_buf_size/1024 for cme in clear_none_in_lst(cmes)]
    y = [cme.ema for cme in clear_none_in_lst(cmes)]

    # 创建散点图
    plt.scatter(x, y, color='blue', marker='o')  # 'o'代表圆圈，color设置点的颜色

    plt.title('EMA -- buffer size')
    plt.xlabel('activation buffer size (KB)')
    plt.ylabel('EMA')
    plt.savefig(save_path, dpi=1000)

def plot_cme_edp_times_buf(cmes: list[CostModelEvaluation], save_path = 'default_ema.png'):
    plt.figure(dpi=1000)
    # todo
    x = [cme.a_buf_size/1024 for cme in clear_none_in_lst(cmes)]
    y = [cme.ema for cme in clear_none_in_lst(cmes)]

    # 创建散点图
    plt.scatter(x, y, color='blue', marker='o')  # 'o'代表圆圈，color设置点的颜色

    plt.title('EMA -- buffer size')
    plt.xlabel('activation buffer size (KB)')
    plt.ylabel('EMA')
    plt.savefig(save_path, dpi=1000)


def plot_two_lines_ema(pkl_paths: list[str], save_path: str):
    """
    The EMA of:
    feature merging VS. non-feature merging
    """
    plt.figure(dpi=1000)
    with open(pkl_paths[0], "rb") as handle:
        list_of_cme_0 = pickle.load(handle)
    with open(pkl_paths[1], "rb") as handle:
        list_of_cme_1 = pickle.load(handle)
    
    # set label
    label0 = 'merge' if 'True' in pkl_paths[0] else 'not-merge'
    label1 = 'merge' if 'True' in pkl_paths[1] else 'not-merge'
    assert label1 != label0, 'pkl path error !!!'


    # line 0
    x0 = [cme.a_buf_size/1024 for cme in clear_none_in_lst(list_of_cme_0)]
    y0 = [cme.ema for cme in clear_none_in_lst(list_of_cme_0)]
    plt.scatter(x0, y0, color='blue', marker='o', label=label0)   # 'o'代表圆圈，color设置点的颜色
    
    # line 1
    x1 = [cme.a_buf_size/1024 for cme in clear_none_in_lst(list_of_cme_1)]
    y1 = [cme.ema for cme in clear_none_in_lst(list_of_cme_1)]
    plt.scatter(x1, y1, color='red', marker='o', label=label1)   # 'o'代表圆圈，color设置点的颜色


    plt.title('EMA -- buffer size')
    plt.xlabel('activation buffer size (KB)')
    plt.ylabel('EMA')
    plt.legend(loc='upper right')
    plt.savefig(save_path, dpi=1000)


def plot_two_lines_edp(pkl_paths: list[str], save_path: str):
    """
    The EDP of:
    feature merging VS. non-feature merging
    """
    plt.figure(dpi=1000)
    with open(pkl_paths[0], "rb") as handle:
        list_of_cme_0 = pickle.load(handle)
    with open(pkl_paths[1], "rb") as handle:
        list_of_cme_1 = pickle.load(handle)
    
    # set label
    label0 = 'merge' if 'True' in pkl_paths[0] else 'not-merge'
    label1 = 'merge' if 'True' in pkl_paths[1] else 'not-merge'
    assert label1 != label0, 'pkl path error !!!'

    # line 0
    x0 = [cme.a_buf_size/1024 for cme in clear_none_in_lst(list_of_cme_0)]
    y0 = [cme.edp for cme in clear_none_in_lst(list_of_cme_0)]
    plt.scatter(x0, y0, c='royalblue', marker='o', s=3, label=label0)   # 'o'代表圆圈，color设置点的颜色
    
    # line 1
    x1 = [cme.a_buf_size/1024 for cme in clear_none_in_lst(list_of_cme_1)]
    y1 = [cme.edp for cme in clear_none_in_lst(list_of_cme_1)]
    plt.scatter(x1, y1, c='darkorange', marker='o', s=3, label=label1)   # 'o'代表圆圈，color设置点的颜色


    plt.title('EDP -- buffer size')
    plt.xlabel('activation buffer size (KB)')
    plt.ylabel('EDP')
    plt.legend(loc='upper right')
    plt.savefig(save_path, dpi=1000)


def plot_two_lines_tsize_edp(pkl_paths: list[str], save_path: str):
    """
    The EDP of:
    fixed tile size VS. free tile size
    """
    plt.figure(dpi=1000)
    with open(pkl_paths[0], "rb") as handle:
        list_of_cme_0 = pickle.load(handle)
    with open(pkl_paths[1], "rb") as handle:
        list_of_cme_1 = pickle.load(handle)
    
    # set label
    label0 = 'fixed-size' if 'fix_tsize' in pkl_paths[0] else 'free-size'
    label1 = 'fixed-size' if 'fix_tsize' in pkl_paths[1] else 'free-size'
    assert label1 != label0, 'pkl path error !!!'

    # line 0
    x0 = [cme.a_buf_size/1024 for cme in clear_none_in_lst(list_of_cme_0)]
    y0 = [cme.edp for cme in clear_none_in_lst(list_of_cme_0)]
    plt.scatter(x0, y0, color='blue', marker='o', label=label0)   # 'o'代表圆圈，color设置点的颜色
    
    # line 1
    x1 = [cme.a_buf_size/1024 for cme in clear_none_in_lst(list_of_cme_1)]
    y1 = [cme.edp for cme in clear_none_in_lst(list_of_cme_1)]
    plt.scatter(x1, y1, color='red', marker='o', label=label1)   # 'o'代表圆圈，color设置点的颜色


    plt.title('EDP -- buffer size')
    plt.xlabel('activation buffer size (KB)')
    plt.ylabel('EDP')
    plt.legend(loc='upper right')
    plt.savefig(save_path, dpi=1000)


def plot_two_lines_tsize_ema(pkl_paths: list[str], save_path: str):
    """
    The EMA of:
    fixed tile size VS. free tile size
    """
    plt.figure(dpi=1000)
    with open(pkl_paths[0], "rb") as handle:
        list_of_cme_0 = pickle.load(handle)
    with open(pkl_paths[1], "rb") as handle:
        list_of_cme_1 = pickle.load(handle)
    
    # set label
    label0 = 'fixed-size' if 'fix_tsize' in pkl_paths[0] else 'free-size'
    label1 = 'fixed-size' if 'fix_tsize' in pkl_paths[1] else 'free-size'
    assert label1 != label0, 'pkl path error !!!'

    # line 0
    x0 = [cme.a_buf_size/1024 for cme in clear_none_in_lst(list_of_cme_0)]
    y0 = [cme.ema for cme in clear_none_in_lst(list_of_cme_0)]
    plt.scatter(x0, y0, color='blue', marker='o', label=label0)   # 'o'代表圆圈，color设置点的颜色
    
    # line 1
    x1 = [cme.a_buf_size/1024 for cme in clear_none_in_lst(list_of_cme_1)]
    y1 = [cme.ema for cme in clear_none_in_lst(list_of_cme_1)]
    plt.scatter(x1, y1, color='red', marker='o', label=label1)   # 'o'代表圆圈，color设置点的颜色


    plt.title('EMA -- buffer size')
    plt.xlabel('activation buffer size (KB)')
    plt.ylabel('EMA')
    plt.legend(loc='upper right')
    plt.savefig(save_path, dpi=1000)

