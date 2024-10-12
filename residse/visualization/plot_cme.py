import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import numpy as np
from residse.classes.cost_model.cost_model import CostModelEvaluation


def plot_cme_edp(cmes: list[CostModelEvaluation], save_path = 'edp.png'):
    plt.figure(dpi=1000)

    x = [cme.a_buf_size/1024 for cme in cmes if cme is not None]
    y = [cme.edp for cme in cmes if cme is not None]

    # 创建散点图
    plt.scatter(x, y, color='blue', marker='o')  # 'o'代表圆圈，color设置点的颜色


    plt.title('EDP -- buffer size')
    plt.xlabel('activation buffer size (KB)')
    plt.ylabel('EDP')
    plt.savefig(save_path, dpi=1000)


def plot_cme_ema(cmes: list[CostModelEvaluation], save_path = 'ema.png'):
    plt.figure(dpi=1000)

    x = [cme.a_buf_size/1024 for cme in cmes if cme is not None]
    y = [cme.ema for cme in cmes if cme is not None]

    # 创建散点图
    plt.scatter(x, y, color='blue', marker='o')  # 'o'代表圆圈，color设置点的颜色


    plt.title('EMA -- buffer size')
    plt.xlabel('activation buffer size (KB)')
    plt.ylabel('EMA')
    plt.savefig(save_path, dpi=1000)

