import logging
import sys
from typing import Generator, Callable, List, Tuple, Any
from residse.classes.hardware.dla import Dla
from residse.classes.hardware.memory import Memory
from residse.classes.workload.stack import Stack
from utils import generate_tile_sequence, find_first_true_index, cumulative_sum, find_lzc, pickle_deepcopy
from math import prod, ceil, floor
logger = logging.getLogger(__name__)

class CostModelEvaluation:
    """
    calculate EMA, Priority of data storage in a_buffer:
    
    if is_feature_merging:
    |input/output tile (1 layer) > |residual tile (residual stack) > |x merging part (a tile)         > |xolp (except merging part)   >  |next input tile (a tile) > |y merging part (n+1 tiles)    >   |yolp (except merging part)
    |tile size with max channel    |tile size - shift size           |merge w * (tile h - shift h)      |other layer's (k-1)*h*c         |input tile size            |merge h * (tile w + shift w)      |other layer's (k-1)*(w+k-1)*c   

    if not is_feature_merging (but reuse_distance_aware):
    input/output tile (1 layer) > residual tile (residual stack) > |xolp                  > next input tile (a tile) > |x residual shift (a tile)   >   |yolp              >           |y residual shift (n+1 tiles)
                                                                   |all layer's (k-1)*h*c                              |shift w * (tile h - shift h)    |all layer's (k-1)*(w+k-1)*c   |shift h * (tile w + shift w)
    
    if not is_feature_merging (and no reuse_distance_aware): #residual都扔到片外最后再考虑
    input/output tile (1 layer) > xolp > next input tile (a tile) > yolp > residual tile (residual stack) > x residual shift (a tile) > y residual shift (n+1 tiles)

    in coding:
    minimal_abuf_for_stack_under_tsize - max i/o tile amount
    next_tile_data_amount
    residual_tile_data_amount  
    x_merging_or_resi_data_amount
    x_olp_data_amount
    y_merging_or_resi_data_amount
    y_olp_data_amount

                   |                                              | F        | HL       | HM       | HR       | WU       | WM       | WD       | LU       | U        | RU       | L        | M        | R           | LD       | D        | RD       |
io tile            | 不考虑逐层变化的话,只需做边界块尺寸管理            |          |          |          | 宽度      |          |          | 高度     |          |          | 宽度      |          |          | 宽度         | 高度     | 高度      | 宽度 高度  |
next tile          | 无预测功能只用当前tile数据量统计的话 同上          |          |          |          | 宽度      |          |          | 高度     |          |          | 宽度      |          |          | 宽度         | 高度     | 高度      | 宽度 高度  |
residual tile      | 边界块没有某一维度偏移,没有tile尺寸变化            |无偏移     |无y偏移    |无y偏移    |无y偏移    |无x偏移    |无x偏移    |无x偏移    |          |          |无x偏移    |          |          |无x偏移       |无y偏移    |无y偏移    |无偏移     |
xolp               | 不做精细考虑(比如RU/R/RD) 只考虑部分情况无xolp     |   X      |          |          |          |   X      |   X      |   X      |          |          |          |          |          |             |          |          |          |
x merge/residual   | 同上                                          |   X      |          |          |          |   X      |   X      |   X      |          |          |          |          |          |             |          |          |          |
yolp               | 需要考虑一组tile的yolp总量和buffer容量,求一个比例  |   X      |   X      |   X      |   X      |以一整行的yolp数据为一组 和总容量对比 超出部分均摊给每一个tile的数据 即用一个比例乘上每个tile的数据量                                                |
y merge/residual   | 同上                                          |   X      |   X      |   X      |   X      |以一整行的yolp数据为一组 和总容量对比 超出部分均摊给每一个tile的数据 即用一个比例乘上每个tile的数据量                                                |
    """
    def __init__(self, *, dla: Dla, a_buf_size: int, stack: Stack, tile_size: Tuple[int], tile_type: str, is_feature_merging: bool, is_rda: bool):
        self.a_buf_size = a_buf_size * 1024     # Byte
        self.dla = dla
        self.stack = stack
        self.tile_type = tile_type
        self.tile_size = tile_size
        # 这里只是给出一个固定的ifm划分的tile size，具体边缘tile尺寸以及每层的tile size需要另行计算。re-calculating each layer's tile size according tile_type
        #TODO: 所有stack使用同一tile size，进行tile size iteration
        if tile_size[0] > stack.ofm_h:
            self.tile_h = stack.ofm_h
        else:
            self.tile_h = tile_size[0]
        if tile_size[1] > stack.ofm_w:
            self.tile_w = stack.ofm_w
        else:
            self.tile_w = tile_size[1]
        self.is_feature_merging = is_feature_merging
        self.is_rda = is_rda
        self.first_true_id = find_first_true_index(self.stack.in_resb)
        # self.tile_index = []
        # self.y_olp_data_amount_each_index = []
        # self.y_merging_or_resi_data_amount_each_index = []
        # self.ema_each_index = []
        self.calc_data_amount()
        self.calc_edp()


    def calc_data_amount(self):
        """
        计算对于某一固定tile, 其运算过程中每种数据类型所需SRAM空间
        """
        self.calc_minimal_abuf_for_stack_under_tsize() # calculate the minimal capacity for abuf under the input tile_size, only consider the tile_size change due to stride
        self.calc_next_tile_data_amount()
        # if self.tile_type in ['U', 'D']:
        #     self.generate_tile_index_list()
        self.calc_olp_data_amount() #目前没有考虑边缘tile overlap所需数据量减少的变化，例如最下方和最上方的tile其实并没有（n+1）个tile的yolp占用着SRAM，全部一视同仁其实会导致SRAM的浪费

        if self.first_true_id is None:  
            # this isn't a residual block
            self.residual_tile_data_amount = 0
            self.x_merging_or_resi_data_amount = 0
            self.y_merging_or_resi_data_amount = 0
            self.current_tile_y_merging_or_resi_data_amount = 0  # 新增初始化
        else:  
            self.calc_merging_length_or_resi_shift()
            self.calc_residual_tile_data_amount()
            self.calc_merging_or_residual_data_amount()


    def calc_edp(self):
        self.calc_ema()
        if self.stack.has_outer_add() and (self.ema is not None):
            self.ema += self.tile_w * self.tile_h * self.stack.och_per_layer[-1]  # outer_add is big residual feature map
        self.calc_en()
        self.calc_la()
        self.edp = self.en * self.la
        self.times_tile_number()


    def calc_minimal_abuf_for_stack_under_tsize(self):
        self.backpropagation_tile_data_amount()
        self.number_of_tile()
        self.minimal_abuf_for_stack_under_tsize = max(self.in_out_tile_data_amount_lst)    # element or byte
        # logger.info(f'lower limit of abuf for stack_{self.stack.id} of tile_size {self.tile_size} is {self.minimal_abuf_for_stack_under_tsize}')

    def backpropagation_tile_data_amount(self):
        
        if (self.stack.ofm_w % self.tile_w == 0):
            boundary_tile_w = self.tile_w
        else:
            boundary_tile_w = (self.stack.ofm_w % self.tile_w)
        
        if (self.stack.ofm_h % self.tile_h == 0):
            boundary_tile_h = self.tile_h
        else:
            boundary_tile_h = (self.stack.ofm_h % self.tile_h)
        
        if self.tile_type in ['HR', 'RU', 'R', 'RD']:
            true_tile_w = boundary_tile_w
        else:
            true_tile_w = self.tile_w

        if self.tile_type in ['WD', 'LD', 'D', 'RD']:
            true_tile_h = boundary_tile_h
        else:
            true_tile_h = self.tile_h

        # self.tile_area = prod(self.tile_size) #计算的是当前tile自己的area，还不支持预测功能
        self.tile_area = true_tile_w * true_tile_h #计算的是当前tile自己的area，还不支持预测功能

        tile_h_per_layer = generate_tile_sequence(out_len=true_tile_h, stride=self.stack.stride_per_layer, power=1) #目前的generate_tile_sequence只是根据stride生成tile尺寸，并没有根据tile type逐层的尺寸变化
        tile_w_per_layer = generate_tile_sequence(out_len=true_tile_w, stride=self.stack.stride_per_layer, power=1)
        tile_h_all_layer = [tile_h_per_layer[0] * prod(self.stack.stride_per_layer)] + tile_h_per_layer        # from ifm to ofm of a stack, number = stack_len + 1
        tile_w_all_layer = [tile_w_per_layer[0] * prod(self.stack.stride_per_layer)] + tile_w_per_layer        # from ifm to ofm of a stack, number = stack_len + 1

        # calc tile h and w
        self.out_tile_h_lst = tile_h_all_layer[1:]
        self.out_tile_w_lst = tile_w_all_layer[1:]
        # self.in_tile_h_lst = tile_h_all_layer[:-1]
        self.in_tile_h_lst = [
            tile if tile <= ifm else ifm 
            for tile, ifm in zip(tile_h_all_layer[:-1], self.stack.ifm_h_per_layer)
        ]
        # self.in_tile_w_lst = tile_w_all_layer[:-1]
        self.in_tile_w_lst = [
            tile if tile <= ifm else ifm 
            for tile, ifm in zip(tile_w_all_layer[:-1], self.stack.ifm_w_per_layer)
        ]
        # calc tile area and data amount
        self.out_tile_area_lst = [h * w for h, w in zip(self.out_tile_h_lst, self.out_tile_w_lst)]
        self.in_tile_area_lst  = [h * w for h, w in zip(self.in_tile_h_lst, self.in_tile_w_lst)]
        self.out_tile_data_amount_lst = [area * ch for area, ch in zip(self.out_tile_area_lst, self.stack.och_per_layer)]
        self.in_tile_data_amount_lst  = [area * ch for area, ch in zip(self.in_tile_area_lst, self.stack.ich_per_layer)]
        self.in_out_tile_data_amount_lst = [a + b for a, b in zip(self.out_tile_data_amount_lst, self.in_tile_data_amount_lst)]


    def number_of_tile(self): #逐层变化的话，如果有违法尺寸，在tile size gen中去规避？
        """
        calc numbers of current tile according tile_type
        """
        if self.tile_type in ['LU', 'U', 'RU', 'L', 'M', 'R', 'LD', 'D', 'RD']:
            number_of_tile_in_row = ceil(self.stack.ofm_w / self.tile_w)
            number_of_tile_in_col = ceil(self.stack.ofm_h / self.tile_h)
            if self.tile_type in ['LU', 'RU', 'LD', 'RD']:
                tile_number_of_current_type = 1
            elif self.tile_type in ['U', 'D']:
                if (number_of_tile_in_row - 2) < 0:
                    tile_number_of_current_type = 0
                else:
                    tile_number_of_current_type = number_of_tile_in_row - 2
            elif self.tile_type in ['L', 'R']:
                if (number_of_tile_in_col - 2) < 0:
                    tile_number_of_current_type = 0
                else:
                    tile_number_of_current_type = number_of_tile_in_col - 2
            else:
                if ((number_of_tile_in_row - 2) < 0) or ((number_of_tile_in_col - 2) < 0):
                    tile_number_of_current_type = 0
                else:
                    tile_number_of_current_type = (number_of_tile_in_row - 2) * (number_of_tile_in_col - 2)
        elif self.tile_type in ['HL', 'HM', 'HR']:
            number_of_tile_in_row = ceil(self.stack.ofm_w / self.tile_w)
            number_of_tile_in_col = 1
            if self.tile_type in ['HL', 'HR']:
                tile_number_of_current_type = 1
            else:
                if (number_of_tile_in_row - 2) < 0:
                    tile_number_of_current_type = 0
                else:
                    tile_number_of_current_type = number_of_tile_in_row - 2                
        elif self.tile_type in ['WU', 'WM', 'WD']:
            number_of_tile_in_row = 1
            number_of_tile_in_col = ceil(self.stack.ofm_h / self.tile_h)
            if self.tile_type in ['WU', 'WD']:
                tile_number_of_current_type = 1
            else:
                if (number_of_tile_in_col - 2) < 0:
                    tile_number_of_current_type = 0
                else:
                    tile_number_of_current_type = number_of_tile_in_col - 2
        elif self.tile_type in ['F']:
            number_of_tile_in_row = 1
            number_of_tile_in_col = 1
            tile_number_of_current_type = 1
        else:
            raise ValueError(f'tile type ERROR: {self.tile_type}')

        self.number_of_tile_in_row = number_of_tile_in_row
        self.number_of_tile_in_col = number_of_tile_in_col
        self.tile_number_of_current_type = tile_number_of_current_type

    def calc_merging_length_or_resi_shift(self):
        residual_shift = self.residual_shift(in_resb=self.stack.in_resb, kernel_size=self.stack.kernel_size)
        olp_to_merging = self.stack.kernel_size[self.first_true_id] - 1
        if self.is_feature_merging:
            merging_length_or_resi_shift = max(residual_shift, olp_to_merging)
        else:
            merging_length_or_resi_shift = residual_shift
        self.merging_length_or_resi_shift = merging_length_or_resi_shift
        self.resi_shift_distance = residual_shift 

    def calc_residual_tile_data_amount(self): #residual tile应该剔除偏移部分
        if self.tile_type in ['F', 'RD']:
            self.residual_tile_data_amount = self.in_tile_data_amount_lst[self.first_true_id]
        elif self.tile_type in ['HL', 'HM', 'HR', 'LD', 'D']:
            self.residual_tile_data_amount = self.in_tile_data_amount_lst[self.first_true_id] / self.in_tile_area_lst[self.first_true_id] * (self.in_tile_h_lst[self.first_true_id]) * (self.in_tile_w_lst[self.first_true_id] - self.resi_shift_distance) 
        elif self.tile_type in ['WU', 'WM', 'WD', 'RU', 'R']:
            self.residual_tile_data_amount = self.in_tile_data_amount_lst[self.first_true_id] / self.in_tile_area_lst[self.first_true_id] * (self.in_tile_h_lst[self.first_true_id] - self.resi_shift_distance) * (self.in_tile_w_lst[self.first_true_id]) 
        else:
            self.residual_tile_data_amount = self.in_tile_data_amount_lst[self.first_true_id] / self.in_tile_area_lst[self.first_true_id] * (self.in_tile_h_lst[self.first_true_id] - self.resi_shift_distance) * (self.in_tile_w_lst[self.first_true_id] - self.resi_shift_distance) 
    
    def calc_merging_or_residual_data_amount(self):
        cu_tile_h = self.in_tile_h_lst[self.first_true_id]
        cu_tile_w = self.in_tile_w_lst[self.first_true_id]
        cu_ifm_w = self.stack.ifm_w_per_layer[self.first_true_id]

        if self.tile_type in ['F', 'WU', 'WM', 'WD']:
            self.x_merging_or_resi_data_amount = 0
        else:
            self.x_merging_or_resi_data_amount = self.merging_length_or_resi_shift * cu_tile_h * self.stack.ich_per_layer[self.first_true_id]

        if self.tile_type in ['F', 'HL', 'HM', 'HR']:
            self.y_merging_or_resi_data_amount = 0
            self.current_tile_y_merging_or_resi_data_amount = 0
        else:
            self.y_merging_or_resi_data_amount = self.merging_length_or_resi_shift * cu_ifm_w * self.stack.ich_per_layer[self.first_true_id]
            self.current_tile_y_merging_or_resi_data_amount = self.merging_length_or_resi_shift * cu_tile_w * self.stack.ich_per_layer[self.first_true_id]

        # elif self.tile_type in ['WU', 'WM', 'WD']:
        #     self.y_merging_or_resi_data_amount = self.merging_length_or_resi_shift * cu_tile_w * self.stack.ich_per_layer[self.first_true_id]
        # elif self.tile_type in ['L', 'M', 'R']:
        #     self.y_merging_or_resi_data_amount = self.merging_length_or_resi_shift * (cu_tile_w + cu_ifm_w) * self.stack.ich_per_layer[self.first_true_id]
        # elif self.tile_type in ['LU', 'RD']:
        #     self.y_merging_or_resi_data_amount = self.merging_length_or_resi_shift * cu_tile_w * self.stack.ich_per_layer[self.first_true_id]
        # elif self.tile_type in ['RU', 'LD']:
        #     self.y_merging_or_resi_data_amount = self.merging_length_or_resi_shift * cu_ifm_w * self.stack.ich_per_layer[self.first_true_id]
        # elif self.tile_type in ['U', 'D']: #从左到右，计算第一行中间或最后一行中间每个tile所需的yolp数量，并把每个tile yolp数量记录在self.y_olp_data_amount_each_index内
        #     for i in self.tile_index:
        #         if self.tile_type in ['U']:
        #             self.y_merging_or_resi_data_amount = self.merging_length_or_resi_shift * cu_tile_w * self.stack.ich_per_layer[self.first_true_id] * (i + 1)
        #         elif self.tile_type in ['D']:
        #             self.y_merging_or_resi_data_amount = self.merging_length_or_resi_shift * (cu_ifm_w - i * cu_tile_w) * self.stack.ich_per_layer[self.first_true_id]
        #         self.y_merging_or_resi_data_amount_each_index.append(self.y_merging_or_resi_data_amount)
        # else:
        #     sys.exit("error: Wrong Tile Type YOU MOTHERFUCKER.")
        
    def residual_shift(self, in_resb: List[bool], kernel_size: List[int]):
        res_shift_per_layer = [(k-1)/2 for k, in_res_block in zip(kernel_size, in_resb) if in_res_block]
        return sum(res_shift_per_layer)
    
    def calc_next_tile_data_amount(self):
        self.next_tile_data_amount = self.tile_area * self.stack.ich_per_layer[0] # 是用当前tile尺寸假装作为下一个tile数据量的，没有考虑tile type然后去分析下一个tile size的情况

    def calc_olp_data_amount(self):
        olp_length = [k - 1 for k in self.stack.kernel_size] #self.in_tile_h_lst和self.in_tile_w_lst生成函数里，其边缘tile type本应当随着layer变化

        if self.tile_type in ['F', 'WU', 'WM', 'WD']:
            self.x_olp_data_amount = 0
        else:
            x_olp_data_amount = [h * x_olp * ich for h, x_olp, ich in zip(self.in_tile_h_lst, olp_length, self.stack.ich_per_layer)]
            if self.is_feature_merging and (self.first_true_id is not None):
                del x_olp_data_amount[self.first_true_id] #merge的话residual输入层olp单独算在merging_or_residual_data_amount内
            self.x_olp_data_amount = sum(x_olp_data_amount)
        
        if self.tile_type in ['F', 'HL', 'HM', 'HR']:
            self.y_olp_data_amount = 0
            self.current_tile_y_olp_data_amount = 0
        else:
            y_olp_data_amount = [ifm_w * olp * ich for ifm_w, olp, ich in zip(self.stack.ifm_w_per_layer, olp_length, self.stack.ich_per_layer)]
            current_tile_y_olp_data_amount = [tile_w * olp * ich for tile_w, olp, ich in zip(self.in_tile_w_lst, olp_length, self.stack.ich_per_layer)]
            if self.is_feature_merging and (self.first_true_id is not None):
                del y_olp_data_amount[self.first_true_id]
                del current_tile_y_olp_data_amount[self.first_true_id]
            self.y_olp_data_amount = sum(y_olp_data_amount)
            self.current_tile_y_olp_data_amount = sum(current_tile_y_olp_data_amount)

        # elif self.tile_type in ['WU', 'WM', 'WD']:
        #     y_olp_data_amount = [w * olp * ich for w, olp, ich in zip(self.in_tile_w_lst, olp_length, self.stack.ich_per_layer)]
        #     if self.is_feature_merging and (self.first_true_id is not None):
        #         del y_olp_data_amount[self.first_true_id]
        #     self.y_olp_data_amount = sum(y_olp_data_amount)
        # elif self.tile_type in ['L', 'M', 'R']:
        #     y_olp_data_amount = [(ifm_w + tile_w) * olp * ich for ifm_w, tile_w, olp, ich in zip(self.stack.ifm_w_per_layer, self.in_tile_w_lst, olp_length, self.stack.ich_per_layer)]
        #     if self.is_feature_merging and (self.first_true_id is not None):
        #         del y_olp_data_amount[self.first_true_id]
        #     self.y_olp_data_amount = sum(y_olp_data_amount)
        # elif self.tile_type in ['LU', 'RD']:
        #     y_olp_data_amount = [tile_w * olp * ich for tile_w, olp, ich in zip(self.in_tile_w_lst, olp_length, self.stack.ich_per_layer)]
        #     if self.is_feature_merging and (self.first_true_id is not None):
        #         del y_olp_data_amount[self.first_true_id]
        #     self.y_olp_data_amount = sum(y_olp_data_amount) 
        # elif self.tile_type in ['RU', 'LD']:
        #     y_olp_data_amount = [ifm_w * olp * ich for ifm_w, olp, ich in zip(self.stack.ifm_w_per_layer, olp_length, self.stack.ich_per_layer)]
        #     if self.is_feature_merging and (self.first_true_id is not None):
        #         del y_olp_data_amount[self.first_true_id]
        #     self.y_olp_data_amount = sum(y_olp_data_amount)
        # elif self.tile_type in ['U', 'D']: #从左到右，计算第一行中间或最后一行中间每个tile所需的yolp数量，并把每个tile yolp数量记录在self.y_olp_data_amount_each_index内
        #     for i in self.tile_index:
        #         if self.tile_type in ['U']:
        #             y_olp_data_amount = [tile_w * olp * ich * (i + 1) for tile_w, olp, ich in zip(self.in_tile_w_lst, olp_length, self.stack.ich_per_layer)]
        #         elif self.tile_type in ['D']:
        #             y_olp_data_amount = [(ifm_w - i * tile_w) * olp * ich for ifm_w, tile_w, olp, ich in zip(self.stack.ifm_w_per_layer, self.in_tile_w_lst, olp_length, self.stack.ich_per_layer)]
        #         if self.is_feature_merging and (self.first_true_id is not None):
        #             del y_olp_data_amount[self.first_true_id]
        #         y_olp_data_amount_each_index = sum(y_olp_data_amount)
        #         self.y_olp_data_amount_each_index.append(y_olp_data_amount_each_index)
        # else:
        #     sys.exit("error: Wrong Tile Type YOU MOTHERFUCKER.")

    # def generate_tile_index_list(self):
    #     if self.tile_number_of_current_type == 0:
    #         self.tile_index = []
    #     else:
    #         if self.tile_type in ['U', 'D']:
    #             self.tile_index = [i for i in range(1, self.tile_number_of_current_type + 1)]
    #         else:
    #             sys.exit("error: Wrong Tile Type YOU MOTHERFUCKER.")

    def calc_ema(self):
        x_olp_ratio = self.x_olp_ema_ratio(ttype=self.tile_type) 
        y_olp_ratio = self.y_olp_ema_ratio(ttype=self.tile_type) 
        residual_ratio = 1 if self.first_true_id == 0 else 2
        tile_io_ema = self.get_tile_io_ema()
        # if self.tile_type in ['U', 'D']:
        #     for yolp_amount, y_mergresi_amount in zip(self.y_olp_data_amount_each_index, self.y_merging_or_resi_data_amount_each_index):
        if self.is_feature_merging and self.is_rda:
            self.every_data_amount = [
                self.minimal_abuf_for_stack_under_tsize ,
                self.residual_tile_data_amount          ,
                self.x_merging_or_resi_data_amount      ,
                self.x_olp_data_amount                  ,
                self.next_tile_data_amount              ,
                self.y_merging_or_resi_data_amount      ,
                self.y_olp_data_amount
                ]
            self.data_increase_line = cumulative_sum(self.every_data_amount)
            self.ratio = [0 if dt <= self.a_buf_size else 1 for dt in self.data_increase_line]
            lzc = find_lzc(self.ratio)
            if lzc is None:
                ''' the a_buf is big enough to store all data on chip '''
                ema = tile_io_ema
            elif lzc == 0:
                ''' the a_buf size is too small to process layer fusion '''
                ema = None
            else :
                part_of_data_block = self.data_increase_line[lzc] - self.a_buf_size
                if lzc == 1:
                    ''' cut at residual_tile_data_amount '''
                    ema = part_of_data_block * residual_ratio \
                        + (self.x_merging_or_resi_data_amount + self.x_olp_data_amount) * x_olp_ratio \
                        + (self.current_tile_y_merging_or_resi_data_amount + self.current_tile_y_olp_data_amount) * y_olp_ratio \
                        + tile_io_ema
                elif lzc == 2:
                    ''' cut at x_merging_or_resi_data_amount '''
                    ema = (part_of_data_block + self.x_olp_data_amount) * x_olp_ratio \
                        + (self.current_tile_y_merging_or_resi_data_amount + self.current_tile_y_olp_data_amount) * y_olp_ratio \
                        + tile_io_ema
                elif lzc == 3:
                    ''' cut at x_olp_data_amount '''
                    ema = part_of_data_block * x_olp_ratio + (self.current_tile_y_merging_or_resi_data_amount + self.current_tile_y_olp_data_amount) * y_olp_ratio \
                        + tile_io_ema
                elif lzc == 4:
                    ''' cut at next_tile_data_amount '''
                    ema = (self.current_tile_y_merging_or_resi_data_amount + self.current_tile_y_olp_data_amount) * y_olp_ratio \
                        + tile_io_ema
                elif lzc == 5:
                    ''' cut at y_merging_or_resi_data_amount '''
                    oversize_y_part_ratio = self.get_oversize_y_part_ratio(part_of_data_block, 1)
                    ema = (self.current_tile_y_merging_or_resi_data_amount * oversize_y_part_ratio + self.current_tile_y_olp_data_amount) * y_olp_ratio + tile_io_ema
                else :
                    ''' cut at y_olp_data_amount '''
                    oversize_y_part_ratio = self.get_oversize_y_part_ratio(part_of_data_block, 2)
                    ema = self.current_tile_y_olp_data_amount * oversize_y_part_ratio * y_olp_ratio + tile_io_ema
            self.ema = ema

        elif not self.is_feature_merging and self.is_rda:
            # self.every_data_amount = [
            #     self.minimal_abuf_for_stack_under_tsize ,
            #     self.residual_tile_data_amount          ,
            #     self.x_olp_data_amount                  ,
            #     self.next_tile_data_amount              ,
            #     self.x_merging_or_resi_data_amount      ,
            #     self.y_olp_data_amount                  ,
            #     self.y_merging_or_resi_data_amount
            #     ]
            # self.data_increase_line = cumulative_sum(self.every_data_amount)
            # self.ratio = [0 if dt <= self.a_buf_size else 1 for dt in self.data_increase_line]
            # lzc = find_lzc(self.ratio)
            # if lzc is None:
            #     ''' the a_buf is big enough to store all data on chip '''
            #     ema = tile_io_ema
            # elif lzc == 0:
            #     ''' the a_buf size is too small to process layer fusion '''
            #     ema = None
            # else :
            #     part_of_data_block = self.data_increase_line[lzc] - self.a_buf_size
            #     if lzc == 1:
            #         ''' cut at residual_tile_data_amount '''
            #         ema = part_of_data_block * residual_ratio \
            #             + (self.x_merging_or_resi_data_amount + self.x_olp_data_amount) * x_olp_ratio \
            #             + (self.current_tile_y_merging_or_resi_data_amount + self.current_tile_y_olp_data_amount) * y_olp_ratio \
            #             + tile_io_ema
            #     elif lzc == 2:
            #         ''' cut at x_olp_data_amount '''
            #         ema = (part_of_data_block + self.x_merging_or_resi_data_amount) * x_olp_ratio \
            #             + (self.current_tile_y_merging_or_resi_data_amount + self.current_tile_y_olp_data_amount) * y_olp_ratio \
            #             + tile_io_ema
            #     elif lzc == 3:
            #         ''' cut at next_tile_data_amount '''
            #         ema = self.x_merging_or_resi_data_amount * x_olp_ratio \
            #             + (self.current_tile_y_merging_or_resi_data_amount + self.current_tile_y_olp_data_amount) * y_olp_ratio \
            #             + tile_io_ema
            #     elif lzc == 4:
            #         ''' cut at x_merging_or_resi_data_amount '''
            #         ema = part_of_data_block * x_olp_ratio \
            #             + (self.current_tile_y_merging_or_resi_data_amount + self.current_tile_y_olp_data_amount) * y_olp_ratio \
            #             + tile_io_ema
            #     elif lzc == 5:
            #         ''' cut at y_olp_data_amount '''
            #         oversize_y_part_ratio = self.get_oversize_y_part_ratio(part_of_data_block, 2)
            #         ema = (self.current_tile_y_olp_data_amount * oversize_y_part_ratio + self.current_tile_y_merging_or_resi_data_amount) * y_olp_ratio + tile_io_ema
            #     else :
            #         ''' cut at y_merging_or_resi_data_amount '''
            #         oversize_y_part_ratio = self.get_oversize_y_part_ratio(part_of_data_block, 1)
            #         ema = self.current_tile_y_merging_or_resi_data_amount * oversize_y_part_ratio * y_olp_ratio + tile_io_ema
            # self.ema = ema
            self.every_data_amount = [
                self.minimal_abuf_for_stack_under_tsize ,
                self.residual_tile_data_amount          ,
                self.x_merging_or_resi_data_amount      ,
                self.x_olp_data_amount                  ,
                self.next_tile_data_amount              ,
                self.y_merging_or_resi_data_amount      ,
                self.y_olp_data_amount
                ]
            self.data_increase_line = cumulative_sum(self.every_data_amount)
            self.ratio = [0 if dt <= self.a_buf_size else 1 for dt in self.data_increase_line]
            lzc = find_lzc(self.ratio)
            if lzc is None:
                ''' the a_buf is big enough to store all data on chip '''
                ema = tile_io_ema
            elif lzc == 0:
                ''' the a_buf size is too small to process layer fusion '''
                ema = None
            else :
                part_of_data_block = self.data_increase_line[lzc] - self.a_buf_size
                if lzc == 1:
                    ''' cut at residual_tile_data_amount '''
                    ema = part_of_data_block * residual_ratio \
                        + (self.x_merging_or_resi_data_amount + self.x_olp_data_amount) * x_olp_ratio \
                        + (self.current_tile_y_merging_or_resi_data_amount + self.current_tile_y_olp_data_amount) * y_olp_ratio \
                        + tile_io_ema
                elif lzc == 2:
                    ''' cut at x_merging_or_resi_data_amount '''
                    ema = (part_of_data_block + self.x_olp_data_amount) * x_olp_ratio \
                        + (self.current_tile_y_merging_or_resi_data_amount + self.current_tile_y_olp_data_amount) * y_olp_ratio \
                        + tile_io_ema
                elif lzc == 3:
                    ''' cut at x_olp_data_amount '''
                    ema = part_of_data_block * x_olp_ratio + (self.current_tile_y_merging_or_resi_data_amount + self.current_tile_y_olp_data_amount) * y_olp_ratio \
                        + tile_io_ema
                elif lzc == 4:
                    ''' cut at next_tile_data_amount '''
                    ema = (self.current_tile_y_merging_or_resi_data_amount + self.current_tile_y_olp_data_amount) * y_olp_ratio \
                        + tile_io_ema
                elif lzc == 5:
                    ''' cut at y_merging_or_resi_data_amount '''
                    oversize_y_part_ratio = self.get_oversize_y_part_ratio(part_of_data_block, 1)
                    ema = (self.current_tile_y_merging_or_resi_data_amount * oversize_y_part_ratio + self.current_tile_y_olp_data_amount) * y_olp_ratio + tile_io_ema
                else :
                    ''' cut at y_olp_data_amount '''
                    oversize_y_part_ratio = self.get_oversize_y_part_ratio(part_of_data_block, 2)
                    ema = self.current_tile_y_olp_data_amount * oversize_y_part_ratio * y_olp_ratio + tile_io_ema
            self.ema = ema

        elif not self.is_feature_merging and not self.is_rda:
            self.every_data_amount = [
                self.minimal_abuf_for_stack_under_tsize ,
                self.x_olp_data_amount                  ,
                self.next_tile_data_amount              ,
                self.y_olp_data_amount                  ,
                self.residual_tile_data_amount          ,
                self.x_merging_or_resi_data_amount      ,
                self.y_merging_or_resi_data_amount
                ]
            self.data_increase_line = cumulative_sum(self.every_data_amount)
            self.ratio = [0 if dt <= self.a_buf_size else 1 for dt in self.data_increase_line]
            lzc = find_lzc(self.ratio)
            if lzc is None:
                ''' the a_buf is big enough to store all data on chip '''
                ema = tile_io_ema
            elif lzc == 0:
                ''' the a_buf size is too small to process layer fusion '''
                ema = None
            else :
                part_of_data_block = self.data_increase_line[lzc] - self.a_buf_size
                if lzc == 1:
                    ''' cut at x_olp_data_amount '''
                    ema = self.residual_tile_data_amount * residual_ratio \
                        + (self.x_merging_or_resi_data_amount + part_of_data_block) * x_olp_ratio \
                        + (self.current_tile_y_merging_or_resi_data_amount + self.current_tile_y_olp_data_amount) * y_olp_ratio \
                        + tile_io_ema
                elif lzc == 2:
                    ''' cut at next_tile_data_amount '''
                    ema = self.residual_tile_data_amount * residual_ratio \
                        + self.x_merging_or_resi_data_amount * x_olp_ratio \
                        + (self.current_tile_y_merging_or_resi_data_amount + self.current_tile_y_olp_data_amount) * y_olp_ratio \
                        + tile_io_ema
                elif lzc == 3:
                    ''' cut at y_olp_data_amount '''
                    oversize_y_part_ratio = self.get_oversize_y_part_ratio(part_of_data_block, 2)
                    ema = self.residual_tile_data_amount * residual_ratio \
                        + self.x_merging_or_resi_data_amount * x_olp_ratio \
                        + (self.current_tile_y_merging_or_resi_data_amount + self.current_tile_y_olp_data_amount * oversize_y_part_ratio) * y_olp_ratio \
                        + tile_io_ema
                elif lzc == 4:
                    ''' cut at residual_tile_data_amount '''
                    ema = part_of_data_block * residual_ratio \
                        + self.x_merging_or_resi_data_amount * x_olp_ratio \
                        + self.current_tile_y_merging_or_resi_data_amount * y_olp_ratio \
                        + tile_io_ema
                elif lzc == 5:
                    ''' cut at x_merging_or_resi_data_amount '''
                    ema = part_of_data_block * x_olp_ratio \
                        + self.current_tile_y_merging_or_resi_data_amount * y_olp_ratio \
                        + tile_io_ema
                else :
                    ''' cut at y_merging_or_resi_data_amount '''
                    oversize_y_part_ratio = self.get_oversize_y_part_ratio(part_of_data_block, 1)
                    ema = self.current_tile_y_merging_or_resi_data_amount * oversize_y_part_ratio * y_olp_ratio \
                        + tile_io_ema
            self.ema = ema
       
        else:
            sys.exit("error: Merging But No RDA.")
        # self.data_increase_line = cumulative_sum(self.every_data_amount)
        # self.ratio = [0 if dt <= self.a_buf_size else 1 for dt in self.data_increase_line]
        # lzc = find_lzc(self.ratio)
        
        # if lzc is None:
        #     ''' the a_buf is big enough to store all data on chip '''
        #     ema = self.get_tile_io_ema()
        # elif lzc == 0:
        #     ''' the a_buf size is too small to process layer fusion '''
        #     ema = None
        # else :
        #     part_of_data_block = self.data_increase_line[lzc] - self.a_buf_size
        #     x_olp_ratio = self.x_olp_ema_ratio(ttype=self.tile_type)
        #     y_olp_ratio = self.y_olp_ema_ratio(ttype=self.tile_type)
        #     residual_ratio = 1 if self.first_true_id == 0 else 2
        #     if lzc == 1:
        #         ''' cut at residual_tile_data_amount '''
        #         ema = part_of_data_block * residual_ratio \
        #             + (self.x_merging_or_resi_data_amount + self.x_olp_data_amount) * x_olp_ratio \
        #             + (self.y_merging_or_resi_data_amount + self.y_olp_data_amount) * y_olp_ratio \
        #             + self.get_tile_io_ema()
        #     elif lzc == 2:
        #         ''' cut at x_merging_or_resi_data_amount '''
        #         ema = (part_of_data_block + x_olp_ratio) * x_olp_ratio \
        #             + (self.y_merging_or_resi_data_amount + self.y_olp_data_amount) * y_olp_ratio \
        #             + self.get_tile_io_ema()
        #     elif lzc == 3:
        #         ''' cut at next_tile_data_amount '''
        #         ema = self.x_olp_data_amount * x_olp_ratio + (self.y_merging_or_resi_data_amount + self.y_olp_data_amount) * y_olp_ratio \
        #             + self.get_tile_io_ema()
        #     elif lzc == 4:
        #         ''' cut at x_olp_data_amount '''
        #         ema = part_of_data_block * x_olp_ratio + (self.y_merging_or_resi_data_amount + self.y_olp_data_amount) * y_olp_ratio \
        #             + self.get_tile_io_ema()
        #     elif lzc == 5:
        #         ''' cut at y_merging_or_resi_data_amount '''
        #         ema = (part_of_data_block + self.y_olp_data_amount) * y_olp_ratio + self.get_tile_io_ema()
        #     else :
        #         ''' cut at y_olp_data_amount '''
        #         ema = part_of_data_block * y_olp_ratio + self.get_tile_io_ema()

        # self.ema = ema
    def get_oversize_y_part_ratio(self, part_of_data_block, data_type):
        if data_type == 1: # type_1 for y_merging_or_resi
            return part_of_data_block / self.y_merging_or_resi_data_amount
        elif data_type == 2: # type_2 for y_olp
            return part_of_data_block / self.y_olp_data_amount
        else:
            sys.exit("error: WRONG DATA_TYPE FOR -- get_oversize_y_part_ratio")


    def get_tile_io_ema(self):
        return self.in_tile_data_amount_lst[0] + self.out_tile_data_amount_lst[-1]

    def x_olp_ema_ratio(self, ttype):
        if ttype in ['LU', 'L', 'LD', 'RU', 'R', 'RD', 'HL', 'HR']:
            return 1
        elif ttype in ['U', 'M', 'D', 'HM']:
            return 2
        else:
            return 0
        
        
    def y_olp_ema_ratio(self, ttype):
        if ttype in ['LU', 'U', 'LD', 'RU', 'D', 'RD', 'WU', 'WD']:
            return 1
        elif ttype in ['L', 'M', 'R', 'WM']:
            return 2
        else:
            return 0


    def multiply_tmp_unroll(self, i) -> int:
        ''' i is layer_id in current stack '''
        tmp_unroll  = ceil(self.out_tile_h_lst[i] / self.dla.u_h) \
                    * ceil(self.out_tile_w_lst[i] / self.dla.u_w) \
                    * ceil(self.stack.och_per_layer[i] / self.dla.u_oc) \
                    * ceil(self.stack.ich_per_layer[i] / self.dla.u_ic) \
                    * ceil(self.stack.kernel_size[i] / self.dla.u_fx) \
                    * ceil(self.stack.kernel_size[i] / self.dla.u_fy)
        return tmp_unroll


    def calc_en(self):
        e_mac = 0.05    # pJ ?  energy unit of single mac
        e_ema = 500     # pJ ?  energy unit of single ema
        self.tmp_unroll_of_stack = sum([self.multiply_tmp_unroll(i) for i in range(self.stack.stack_len)])
        if self.ema is not None:
            en_of_macs = self.tmp_unroll_of_stack * self.dla.number_of_mac * e_mac #number_of_mac用ceil有冗余
            en_of_datas = self.ema * e_ema
        else:
            en_of_macs = 0
            en_of_datas = 0
        self.en_of_macs = en_of_macs
        self.en_of_datas = en_of_datas
        self.en = self.en_of_macs + self.en_of_datas


    def calc_la(self):
        if self.ema is not None:
            la_of_macs = self.tmp_unroll_of_stack
            la_of_datas = ceil(self.ema / self.dla.dram.bw)
        else:
            la_of_macs = 0
            la_of_datas = 0
        self.la_of_macs = la_of_macs
        self.la_of_datas = la_of_datas
        self.la = max(self.la_of_macs, self.la_of_datas) # 这里是当前块的访存和计算比 理应是当前块的计算和下一块的访存比 即：访存 计算+预防存 计算+预防存 计算+预防存 ......


    def times_tile_number(self):
        if self.ema is not None:
            self.ema *= self.tile_number_of_current_type
        self.en  *= self.tile_number_of_current_type
        self.la  *= self.tile_number_of_current_type
        self.edp = self.en * self.la    # recalculating edp, edp cannot times number_of_type


    def __str__(self):
        return f"cme(stack={self.stack}, tsize={self.tile_size}, ttype={self.tile_type}, edp={self.edp}, en={self.en}, la={self.la}, ema={self.ema})"
    
    
    def __repr__(self) -> str:
        return str(self)
    
    
    def __add__(self, other):
        sum = pickle_deepcopy(self)

        # EDP
        sum.edp = (sum.en + other.en) * (sum.la + other.la)   # edp 不可以直接累加

        # EMA
        if (sum.ema is None) or (other.ema is None):
            sum.ema = None
        else:
            sum.ema += other.ema

        # EN
        sum.en_of_macs += other.en_of_macs
        sum.en_of_datas += other.en_of_datas
        sum.en += other.en

        # LA
        sum.la += other.la

        # Stack
        if type(sum.stack) != list:
            sum.stack = [sum.stack.id]
        if type(other.stack) != list:
            other_stack = [other.stack.id]
        else:
            other_stack = other.stack
        if sum.stack != other_stack:
            sum.stack += other_stack

        # Tile type
        if type(sum.tile_type) != list:
            sum.tile_type = [sum.tile_type]
        if type(other.tile_type) != list:
            other_tile_type = [other.tile_type]
        else:
            other_tile_type = other.tile_type
        sum.tile_type += other_tile_type

        # Tile size
        if sum.tile_size != other.tile_size:
            sum.tile_size = (-1, -1)

        # a_buf_size
        if sum.a_buf_size != other.a_buf_size:
            raise TypeError(f'cmes with different a_buf_size cannot add !!!')
        
        # Not Addable
        func = [
            "calc_data_amount",
            "calc_minimal_abuf_for_stack_under_tsize",
            "backpropagation_tile_data_amount",
            "number_of_tile",
            "calc_residual_tile_data_amount",
            "calc_merging_length_or_resi_shift",
            "calc_merging_or_residual_data_amount",
            "residual_shift",
            "calc_next_tile_data_amount",
            "calc_olp_data_amount",
            "calc_edp",
            "calc_ema",
            "x_olp_ema_ratio",
            "y_olp_ema_ratio",
            "multiply_tmp_unroll",
            "calc_en",
            "calc_la",
            "times_tile_number",
            "get_oversize_y_part_ratio",
            "get_tile_io_ema",
            "current_tile_y_merging_or_resi_data_amount",
        ]
        add_attr = [
            "ema",
            "en",
            "en_of_macs",
            "en_of_datas",
            "la",
            "edp",
            "tile_type",
            "stack",
            "tile_size",
            "a_buf_size",
        ]

        for attr in dir(sum):
            if attr not in (func + add_attr) and attr[0] != "_":
                delattr(sum, attr)

        return sum


    def __jsonrepr__(self):
        """
        JSON representation used for saving this object to a complete json file.
        """
        return {
            # "stack": self.stack,
            # "a_buf_size": self.a_buf_size,
            "EDP": self.edp,
            "energy": {
                "total_en": self.en,
                "macs_en": self.en_of_macs,
                "data_move_en": self.en_of_datas,
            },
            # "latency": {
            #     "total_la": self.la,
            #     "macs_la": self.la_of_macs,
            #     "external_la": self.la_of_datas,
            # },
            "latency": self.la,
            "EMA": self.ema,
            # "other_info": {
            #     "every_data_amount": self.every_data_amount,
            #     "tile_size": self.tile_size,
            # },
        }






if __name__ == '__main__':
    from residse.classes.hardware.HardwareGenerator import HardwareGenerator
    from utils import sum_cme
    
    st2_di =  { 7: {'op': 'conv', 'stride': 2, 'in_resb': True, 'dim': [100, 100, 16, 64, 3, 3]}, 8: {'op': 'conv', 'stride': 1, 'in_resb': True, 'dim': [100, 100, 16, 16, 3, 3]}}
    st3_di =  { 7: {'op': 'conv', 'stride': 2, 'in_resb': True, 'dim': [28, 28, 128, 64, 3, 3]}, 8: {'op': 'conv', 'stride': 1, 'in_resb': True, 'dim': [28, 28, 128, 128, 3, 3]}}
    st4_di =  { 7: {'op': 'conv', 'stride': 1, 'in_resb': True, 'dim': [270, 480, 64, 64, 3, 3]}, 8: {'op': 'conv', 'stride': 1, 'in_resb': True, 'dim': [270, 480, 64, 64, 3, 3]}}    
    dla = HardwareGenerator(json_hw="residse/inputs/HW/srgan_1.json").get_dla()
    buf_lst = dla.a_buf.get_size_list()
    
    #: test difference between merge and not-merge
    cme_merge        = CostModelEvaluation(dla=dla, a_buf_size=50, stack=Stack(1, st4_di), tile_size=(4, 4), tile_type='M', is_feature_merging=True, is_rda=True)
    cme_notme        = CostModelEvaluation(dla=dla, a_buf_size=50, stack=Stack(1, st4_di), tile_size=(4, 4), tile_type='M', is_feature_merging=False, is_rda=True)
    cme_notme_notrda = CostModelEvaluation(dla=dla, a_buf_size=50, stack=Stack(1, st4_di), tile_size=(4, 4), tile_type='M', is_feature_merging=False, is_rda=False)
    print(cme_merge.ema)
    print(cme_merge.en)
    print(cme_merge.la)
    print(cme_merge.edp)
    print(cme_merge.every_data_amount)
    print(cme_merge.ratio)
    print()
    print(cme_notme.ema)
    print(cme_notme.en)
    print(cme_notme.la)
    print(cme_notme.edp)
    print(cme_notme.every_data_amount)
    print(cme_notme.ratio)
    print()
    print(cme_notme_notrda.ema)
    print(cme_notme_notrda.en)
    print(cme_notme_notrda.la)
    print(cme_notme_notrda.edp)
    print(cme_notme_notrda.every_data_amount)
    print(cme_notme_notrda.ratio)
    # print()
    # print(cme_merge.out_tile_h_lst)
    # print(cme_merge.out_tile_w_lst)
    # print(cme_merge.in_tile_h_lst)
    # print(cme_merge.in_tile_w_lst)
    # print(cme_merge.stack.ifm_w_per_layer)
    # print(cme_merge.stack.ifm_h_per_layer)
    # print(cme_merge.stack.ofm_w_per_layer)
    # print(cme_merge.stack.ofm_h_per_layer)
    #: test cme performance change with a_buf_size
    # for buf in buf_lst:
    #     cme = CostModelEvaluation(dla=dla, a_buf_size=buf, stack=Stack(1, st3_di), tile_size=(4, 4), tile_type='M', is_feature_merging=True)
    #     print('---'*30)
    #     print("tile_area                    :", cme.tile_area)
    #     print("out_tile_area_lst            :", cme.out_tile_area_lst)
    #     print("in_tile_area_lst             :", cme.in_tile_area_lst)
    #     print("out_tile_data_amount_lst     :", cme.out_tile_data_amount_lst)
    #     print("in_tile_data_amount_lst      :", cme.in_tile_data_amount_lst)
    #     print("in_out_tile_data_amount_lst  :", cme.in_out_tile_data_amount_lst)
    #     print("minimal_abuf_for_stack_under_tsize:", cme.minimal_abuf_for_stack_under_tsize)
    #     print("residual_tile_data_amount    :", cme.residual_tile_data_amount)

    #     print("a_buf_size                   :", cme.a_buf_size/1024)    
    #     print("first_true_id                :", cme.first_true_id              )           
    #     print("x_merging_or_resi_data_amount        :", cme.x_merging_or_resi_data_amount      )                   
    #     print("y_merging_or_resi_data_amount        :", cme.y_merging_or_resi_data_amount      )                   
    #     print("number_of_tile_in_row        :", cme.number_of_tile_in_row      )                   
    #     print("number_of_tile_in_col        :", cme.number_of_tile_in_col      )                   
    #     print("tile_number_of_current_type  :", cme.tile_number_of_current_type)                           
    #     print("merging_length_or_resi_shift               :", cme.merging_length_or_resi_shift             )               
    #     print("x_olp_data_amount            :", cme.x_olp_data_amount          )               
    #     print("y_olp_data_amount            :", cme.y_olp_data_amount          )               
    #     print("every_data_amount            :", cme.every_data_amount          )    
    #     print("data_increase_line           :", cme.data_increase_line         )    
    #     print("ratio                        :", cme.ratio                      )           
    #     print("ema                          :", cme.ema                        )   
    #     print("en                           :", cme.en                         )   
    #     print("la                           :", cme.la                         )   
    #     print("edp                          :", cme.edp                        )   
        
    #: test summation of two cmes
    # cme1 = CostModelEvaluation(dla=dla, a_buf_size=10, stack=Stack(1, st3_di), tile_size=(4, 4), tile_type='M', is_feature_merging=True)
    # cme2 = CostModelEvaluation(dla=dla, a_buf_size=10, stack=Stack(1, st3_di), tile_size=(4, 4), tile_type='L', is_feature_merging=True)
    # cme3 = CostModelEvaluation(dla=dla, a_buf_size=10, stack=Stack(1, st2_di), tile_size=(4, 4), tile_type='L', is_feature_merging=True)
    # print(cme1.ema)
    # print(cme1.en)
    # print(cme1.la)
    # print(cme1.edp)
    # print()
    # print(cme2.ema)
    # print(cme2.en)
    # print(cme2.la)
    # print(cme2.edp)
    # cme_s = sum_cme([cme1, cme2, cme3])
    # print()
    # print(cme_s.ema)
    # print(cme_s.en)
    # print(cme_s.la)
    # print(cme_s.edp)
    # assert cme_s.en == cme1.en + cme2.en + cme3.en
    # assert cme_s.la == cme1.la + cme2.la + cme3.la
    # assert cme_s.ema == cme1.ema + cme2.ema + cme3.ema
    # assert cme_s.edp == cme_s.en * cme_s.la
    # print('good')
    
    
    
    