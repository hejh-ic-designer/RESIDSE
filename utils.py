import pickle
import os
from pprint import pprint
from copy import deepcopy
from typing import TYPE_CHECKING
import itertools



if TYPE_CHECKING:
    from residse.classes.cost_model.cost_model import CostModelEvaluation


def pickle_deepcopy(to_copy):
    copy = None
    copied = False
    try:
        copy = pickle.loads(pickle.dumps(to_copy, -1))
        return copy
    except:
        pass
        # fallback to other options

    if not copied:
        return deepcopy(to_copy)


def export(export_path, file_to_export):
    """export dict format file at export_path
    """
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    with open(export_path, 'a') as ff:
        pprint(object=file_to_export, stream=ff, indent=4, width=150, sort_dicts=False)
    print(f'export at path {export_path} DONE!')
    
    
def generate_tile_sequence(out_len: int, stride: list, power: int) -> list:
    sequence = [out_len]
    reverse_and_pop = stride[::-1][:-1]
    square = [i**power for i in reverse_and_pop]
    for ratio in square:
        next_item = sequence[-1] * ratio
        sequence.append(next_item)
    return sequence[::-1]


def find_first_true_index(bool_list):
    # 使用 enumerate 和 next 函数找到第一个 True 的索引
    return next((i for i, value in enumerate(bool_list) if value), None)


def cumulative_sum(lst):
    # 使用 itertools.accumulate 计算累积和
    return list(itertools.accumulate(lst))


def find_lzc(lst):
    try:
        return next(i for i, value in enumerate(lst) if value == 1)
    except StopIteration:
        return None

def sum_cme(lst_of_cme: list['CostModelEvaluation']):
    summation = None
    for cme in lst_of_cme:
        if cme is None:
            return None
        if summation is None:
            summation = cme
        else:
            summation += cme
    return summation


def clear_none_in_lst(lst: list):
    return [x for x in lst if x is not None]


if __name__ == '__main__':
    a = 10
    p = [1, 2, 2]

    # 生成数列
    sequence = generate_tile_sequence(a, p)
    print(f"Generated sequence: {sequence}")
    
    
    # 示例列表
    bool_list = [False, False, False, False, False]

    # 找到第一个 True 的索引
    index = find_first_true_index(bool_list)
    print(f"The index of the first True is: {index}")

    
    