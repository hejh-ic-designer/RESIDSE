import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from utils import clear_none_in_lst
import numpy as np
from residse.classes.cost_model.cost_model import CostModelEvaluation
import seaborn as sns

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


def plot_cme_tileiter(cmes: list[CostModelEvaluation], save_path = 'default_tileiter.png'):
    """
    Plot a heatmap of CostModelEvaluation objects based on tile sizes and their corresponding EDP values.
    
    :param cmes: List of CostModelEvaluation objects.
    :param save_path: Path where the plot will be saved.
    """
    # Remove None entries from the list
    cmes = clear_none_in_lst(cmes)
    
    # Extract unique tile heights and widths
    tile_heights = sorted(set([cme.tile_size[0] for cme in cmes]))
    tile_widths = sorted(set([cme.tile_size[1] for cme in cmes]))
    
    # Create an empty matrix for EDP values
    edp_matrix = np.zeros((len(tile_heights), len(tile_widths)))
    
    # Fill the matrix with EDP values
    for cme in cmes:
        h_idx = tile_heights.index(cme.tile_size[0])
        w_idx = tile_widths.index(cme.tile_size[1])
        edp_matrix[h_idx, w_idx] = cme.edp
    
    # Plotting
    plt.figure(dpi=1000)
    sns.heatmap(edp_matrix, annot=True, fmt=".1f", xticklabels=tile_widths, yticklabels=tile_heights, cmap="viridis")
    plt.xlabel('Tile Width')
    plt.ylabel('Tile Height')
    plt.title('EDP Heatmap for Different Tile Sizes')
    plt.colorbar()
    
    # Save the figure
    plt.savefig(save_path)
    

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
    label0 = 'With Feature Merging' if 'True' in pkl_paths[0] else 'Without Feature Merging'
    label1 = 'With Feature Merging' if 'True' in pkl_paths[1] else 'Without Feature Merging'
    assert label1 != label0, 'pkl path error !!!'

    # line 0
    x0 = [cme.a_buf_size/1024 for cme in clear_none_in_lst(list_of_cme_0)]
    y0 = [cme.edp for cme in clear_none_in_lst(list_of_cme_0)]
    plt.scatter(x0, y0, c='darkorange', marker='o', s=5, label=label0)   # 'o'代表圆圈，color设置点的颜色
    
    # line 1
    x1 = [cme.a_buf_size/1024 for cme in clear_none_in_lst(list_of_cme_1)]
    y1 = [cme.edp for cme in clear_none_in_lst(list_of_cme_1)]
    plt.scatter(x1, y1, c='royalblue', marker='o', s=5, label=label1)   # 'o'代表圆圈，color设置点的颜色


    plt.title('ResNet18',fontsize=18)
    plt.xlabel('On-Chip Memory Footprint for Feature (KB)',fontsize=15)
    plt.ylabel('EDP (pJ x Cycles)',fontsize=15)
    plt.legend(prop={'size':13},loc='upper right',markerscale=3)
    plt.savefig(save_path, dpi=1000)


def plot_compare_lines_edp(pkl_paths: list[str], save_path: str):
    """
    The EDP of:
    feature merging VS. non-feature merging
    """
    plt.figure(dpi=1000)
    with open(pkl_paths[0], "rb") as handle:
        list_of_cme_0 = pickle.load(handle)
    with open(pkl_paths[1], "rb") as handle:
        list_of_cme_1 = pickle.load(handle)
    with open(pkl_paths[2], "rb") as handle:
        list_of_cme_2 = pickle.load(handle)
    
    # set label
    label0 = 'Merging / RDA' if '--merge_True--rda_True' in pkl_paths[0] else 'Wrong'
    label1 = 'No Merging / RDA' if '--merge_False--rda_True' in pkl_paths[1] else 'Wrong'
    label2 = 'No Merging / No RDA' if '--merge_False--rda_False' in pkl_paths[2] else 'Wrong'
    assert label1 != label0, 'pkl path error !!!'
    assert label1 != label2, 'pkl path error !!!'
    assert label2 != label0, 'pkl path error !!!'

    # line 0
    x0 = [cme.a_buf_size/1024 for cme in clear_none_in_lst(list_of_cme_0)]
    y0 = [cme.edp for cme in clear_none_in_lst(list_of_cme_0)]
    plt.scatter(x0, y0, c='darkorange', marker='o', s=5, label=label0)   # 'o'代表圆圈，color设置点的颜色
    
    # line 1
    x1 = [cme.a_buf_size/1024 for cme in clear_none_in_lst(list_of_cme_1)]
    y1 = [cme.edp for cme in clear_none_in_lst(list_of_cme_1)]
    plt.scatter(x1, y1, c='royalblue', marker='o', s=5, label=label1)

    # line 2
    x2 = [cme.a_buf_size/1024 for cme in clear_none_in_lst(list_of_cme_2)]
    y2 = [cme.edp for cme in clear_none_in_lst(list_of_cme_2)]
    plt.scatter(x2, y2, c='green', marker='o', s=5, label=label2)
    for item1, item2 in zip(x0, y0):
        print('Merging / RDA', item1, item2)
    for item1, item2 in zip(x1, y1):
        print('No Merging / RDA', item1, item2)
    for item1, item2 in zip(x2, y2):
        print('No Merging / No RDA', item1, item2)
    if 'resnet18' in save_path:
        plt.title('ResNet18',fontsize=18)
    elif 'srgan' in save_path:
        plt.title('SRGAN',fontsize=18)
    plt.xlabel('On-Chip Memory Footprint for Feature (KB)',fontsize=15)
    plt.ylabel('EDP (pJ x Cycles)',fontsize=15)
    # plt.legend(prop={'size':13},loc='upper right',markerscale=3)
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

