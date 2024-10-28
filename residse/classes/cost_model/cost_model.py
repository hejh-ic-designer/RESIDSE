import logging
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
    input/output tile (1 layer) > residual tile (residual stack) > x merging part (a tile) > next input tile (a tile) > xolp (except merging part) > y merging part (n+1 tiles) > yolp (except merging part)

    if not is_feature_merging:
    input/output tile (1 layer) > residual tile (residual stack) > x residual shift (a tile) > next input tile (a tile) > xolp > y residual shift (n+1 tiles) > yolp
    
    in coding:
    lower_limit_of_abuf_for_stack > residual_tile_data_amount > x_merging_data_amount > next_tile_data_amount > x_olp_data_amount > y_merging_data_amount > y_olp_data_amount
    """
    def __init__(self, *, dla: Dla, a_buf_size: int, stack: Stack, tile_size: Tuple[int], tile_type: str, is_feature_merging: bool):
        self.a_buf_size = a_buf_size * 1024     # Byte
        self.dla = dla
        self.stack = stack
        self.tile_type = tile_type
        self.tile_size = tile_size  #? re-calculating tile size according tile_type?
        self.tile_h = tile_size[0]
        self.tile_w = tile_size[1]
        self.is_feature_merging = is_feature_merging
        self.first_true_id = find_first_true_index(self.stack.in_resb)
        self.calc_data_amount()
        self.calc_edp()


    def calc_data_amount(self):
        """
        calculate every data amount
        """
        self.calc_lower_limit_of_abuf_for_stack()
        self.calc_next_tile_data_amount()
        self.calc_olp_data_amount()

        if self.first_true_id is None:  
            # this isn't a residual block
            self.residual_tile_data_amount = 0
            self.x_merging_data_amount = 0
            self.y_merging_data_amount = 0
        else:    
            self.calc_residual_tile_data_amount()
            self.calc_merging_length()
            self.calc_merging_data_amount()


    def calc_edp(self):
        self.calc_ema()
        if self.stack.has_outer_add():
            self.ema += self.stack.ofm_w * self.stack.ofm_h * self.stack.och_per_layer[-1]  # outer_add is big residual feature map
        self.calc_en()
        self.calc_la()
        self.edp = self.en * self.la
        self.times_tile_number()


    def calc_lower_limit_of_abuf_for_stack(self):
        self.backpropagation_tile_data_amount()
        self.number_of_tile()
        self.lower_limit_of_abuf_for_stack = max(self.in_out_tile_data_amount_lst)    # element or byte
        # logger.info(f'lower limit of abuf for stack_{self.stack.id} of tile_size {self.tile_size} is {self.lower_limit_of_abuf_for_stack}')


    def backpropagation_tile_data_amount(self):
        self.tile_area = prod(self.tile_size)
        tile_h_per_layer = generate_tile_sequence(out_len=self.tile_h, stride=self.stack.stride_per_layer, power=1)
        tile_w_per_layer = generate_tile_sequence(out_len=self.tile_w, stride=self.stack.stride_per_layer, power=1)
        tile_h_all_layer = [tile_h_per_layer[0] * prod(self.stack.stride_per_layer)] + tile_h_per_layer        # from ifm to ofm of a stack, number = stack_len + 1
        tile_w_all_layer = [tile_w_per_layer[0] * prod(self.stack.stride_per_layer)] + tile_w_per_layer        # from ifm to ofm of a stack, number = stack_len + 1

        # calc tile h and w
        self.out_tile_h_lst = tile_h_all_layer[1:]
        self.out_tile_w_lst = tile_w_all_layer[1:]
        self.in_tile_h_lst = tile_h_all_layer[:-1]
        self.in_tile_w_lst = tile_w_all_layer[:-1]

        # calc tile area and data amount
        self.out_tile_area_lst = [h * w for h, w in zip(self.out_tile_h_lst, self.out_tile_w_lst)]
        self.in_tile_area_lst  = [h * w for h, w in zip(self.in_tile_h_lst, self.in_tile_w_lst)]
        self.out_tile_data_amount_lst = [area * ch for area, ch in zip(self.out_tile_area_lst, self.stack.och_per_layer)]
        self.in_tile_data_amount_lst  = [area * ch for area, ch in zip(self.in_tile_area_lst, self.stack.ich_per_layer)]
        self.in_out_tile_data_amount_lst = [a + b for a, b in zip(self.out_tile_data_amount_lst, self.in_tile_data_amount_lst)]


    def number_of_tile(self):
        """
        calc numbers of current tile according tile_type
        """
        if self.tile_type in ['LU', 'U', 'RU', 'L', 'M', 'R', 'LD', 'D', 'RD']:
            number_of_tile_in_row = floor(self.stack.ofm_w / self.tile_w + 1)
            number_of_tile_in_col = floor(self.stack.ofm_h / self.tile_h + 1)
            if self.tile_type in ['LU', 'RU', 'LD', 'RD']:
                tile_number_of_current_type = 1
            elif self.tile_type in ['U', 'D']:
                tile_number_of_current_type = number_of_tile_in_row - 2
            elif self.tile_type in ['L', 'R']:
                tile_number_of_current_type = number_of_tile_in_col - 2
            else:
                tile_number_of_current_type = (number_of_tile_in_row - 2) * (number_of_tile_in_col - 2)
        elif self.tile_type in ['HL', 'HM', 'HR']:
            number_of_tile_in_row = floor(self.stack.ofm_w / self.tile_w + 1)
            number_of_tile_in_col = 1
            if self.tile_type in ['HL', 'HR']:
                tile_number_of_current_type = 1
            else:
                tile_number_of_current_type = number_of_tile_in_row - 2
        elif self.tile_type in ['WU', 'WM', 'WD']:
            number_of_tile_in_row = 1
            number_of_tile_in_col = floor(self.stack.ofm_h / self.tile_h + 1)
            if self.tile_type in ['WU', 'WD']:
                tile_number_of_current_type = 1
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


    def calc_residual_tile_data_amount(self):
        self.residual_tile_data_amount = self.in_tile_data_amount_lst[self.first_true_id]
    
    
    def calc_merging_length(self):
        residual_shift = self.residual_shift(in_resb=self.stack.in_resb, kernel_size=self.stack.kernel_size)
        olp_to_merging = self.stack.kernel_size[self.first_true_id] - 1
        if self.is_feature_merging:
            merging_length = max(residual_shift, olp_to_merging)
        else:
            merging_length = residual_shift
        self.merging_length = merging_length

    
    def calc_merging_data_amount(self):
        cu_tile_h = self.in_tile_h_lst[self.first_true_id]
        cu_tile_w = self.in_tile_w_lst[self.first_true_id]
        self.x_merging_data_amount = self.merging_length * cu_tile_h * self.stack.ich_per_layer[self.first_true_id]
        self.y_merging_data_amount = self.merging_length * (cu_tile_w + self.merging_length) * self.stack.ich_per_layer[self.first_true_id] * (self.number_of_tile_in_row + 1)


    def residual_shift(self, in_resb: List[bool], kernel_size: List[int]):
        res_shift_per_layer = [(k-1)/2 for k, in_res_block in zip(kernel_size, in_resb) if in_res_block]
        return sum(res_shift_per_layer)
    
    
    def calc_next_tile_data_amount(self):
        self.next_tile_data_amount = self.tile_area * self.stack.ich_per_layer[0]
    
    
    def calc_olp_data_amount(self):
        olp_length = [k - 1 for k in self.stack.kernel_size]
        x_olp_data_amount = [h * x_olp * ich for h, x_olp, ich in zip(self.in_tile_h_lst, olp_length, self.stack.ich_per_layer)]
        y_olp_data_amount = [(w + olp) * olp * ich * (self.number_of_tile_in_row + 1) for w, olp, ich in zip(self.in_tile_w_lst, olp_length, self.stack.ich_per_layer)]

        if self.is_feature_merging and (self.first_true_id is not None):
            del x_olp_data_amount[self.first_true_id]
            del y_olp_data_amount[self.first_true_id]

        self.x_olp_data_amount = sum(x_olp_data_amount)
        self.y_olp_data_amount = sum(y_olp_data_amount)


    def calc_y_ema_ratio_each_tile(self, part_data_amount, total_data_amount, ofm_w, ofm_h, tile_h, tile_w, n):
        part = part_data_amount * (n+1) * tile_w / total_data_amount
        length = (n+1) * tile_w
        blue = length - part
        y_num = ceil(ofm_h / tile_h) - 1
        total_length = y_num * ofm_w
        remain_len = total_length - floor(total_length / length) * length
        last_red = 0 if (remain_len < blue) else (remain_len - blue)
        total_red = part * floor(total_length / length) + last_red
        ratio = (total_red / total_length) / (n+1)
        if ratio > 1:
            raise NameError
        logger.info(f'ratio: {ratio}')
        return ratio


    def calc_ema(self):
        self.every_data_amount = [
            self.lower_limit_of_abuf_for_stack ,
            self.residual_tile_data_amount     ,
            self.x_merging_data_amount         ,
            self.next_tile_data_amount         ,
            self.x_olp_data_amount             ,
            self.y_merging_data_amount         ,
            self.y_olp_data_amount              
            ]
        self.data_increase_line = cumulative_sum(self.every_data_amount)
        self.ratio = [0 if dt <= self.a_buf_size else 1 for dt in self.data_increase_line]
        lzc = find_lzc(self.ratio)
        
        if lzc is None:
            ''' the a_buf is big enough to store all data on chip '''
            ema = self.lower_limit_of_abuf_for_stack
        elif lzc == 0:
            ''' the a_buf size is too small to process layer fusion '''
            ema = None
        else :
            part_of_data_block = self.data_increase_line[lzc] - self.a_buf_size
            x_olp_ratio = self.x_olp_ema_ratio(ttype=self.tile_type)
            y_olp_ratio = self.y_olp_ema_ratio(ttype=self.tile_type)
            residual_ratio = 1 if self.first_true_id == 0 else 2
            if lzc == 1:
                ''' cut at residual_tile_data_amount '''
                ema = part_of_data_block * residual_ratio \
                    + (self.x_merging_data_amount + self.x_olp_data_amount) * x_olp_ratio \
                    + (self.y_merging_data_amount + self.y_olp_data_amount) * y_olp_ratio / (self.number_of_tile_in_row + 1) \
                    + self.lower_limit_of_abuf_for_stack
            elif lzc == 2:
                ''' cut at x_merging_data_amount '''
                ema = (part_of_data_block + x_olp_ratio) * x_olp_ratio \
                    + (self.y_merging_data_amount + self.y_olp_data_amount) * y_olp_ratio / (self.number_of_tile_in_row + 1)  \
                    + self.lower_limit_of_abuf_for_stack
            elif lzc == 3:
                ''' cut at next_tile_data_amount '''
                ema = self.x_olp_data_amount * x_olp_ratio + (self.y_merging_data_amount + self.y_olp_data_amount) * y_olp_ratio / (self.number_of_tile_in_row + 1)  \
                    + self.lower_limit_of_abuf_for_stack
            elif lzc == 4:
                ''' cut at x_olp_data_amount '''
                ema = part_of_data_block * x_olp_ratio + (self.y_merging_data_amount + self.y_olp_data_amount) * y_olp_ratio / (self.number_of_tile_in_row + 1)  \
                    + self.lower_limit_of_abuf_for_stack
            elif lzc == 5:
                ''' cut at y_merging_data_amount '''
                y_ema_ratio_each_tile = self.calc_y_ema_ratio_each_tile(part_of_data_block, self.y_merging_data_amount, self.stack.ofm_w, self.stack.ofm_h, self.tile_h, self.tile_w, self.number_of_tile_in_row)
                ema = (part_of_data_block + self.y_olp_data_amount) * y_olp_ratio * y_ema_ratio_each_tile  + self.lower_limit_of_abuf_for_stack
            else :
                ''' cut at y_olp_data_amount '''
                y_ema_ratio_each_tile = self.calc_y_ema_ratio_each_tile(part_of_data_block, self.y_olp_data_amount, self.stack.ofm_w, self.stack.ofm_h, self.tile_h, self.tile_w, self.number_of_tile_in_row)
                ema = part_of_data_block * y_olp_ratio * y_ema_ratio_each_tile  + self.lower_limit_of_abuf_for_stack

        self.ema = ema


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
            en_of_macs = self.tmp_unroll_of_stack * self.dla.number_of_mac * e_mac
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
        self.la = max(self.la_of_macs, self.la_of_datas)


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
            "calc_lower_limit_of_abuf_for_stack",
            "backpropagation_tile_data_amount",
            "number_of_tile",
            "calc_residual_tile_data_amount",
            "calc_merging_length",
            "calc_merging_data_amount",
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
            "calc_y_ema_ratio_each_tile",
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
    dla = HardwareGenerator(json_hw="residse/inputs/HW/res18_1.json").get_dla()
    buf_lst = dla.a_buf.get_size_list()
    
    #: test difference between merge and not-merge
    cme_merge = CostModelEvaluation(dla=dla, a_buf_size=10, stack=Stack(1, st3_di), tile_size=(4, 4), tile_type='M', is_feature_merging=True)
    cme_notme = CostModelEvaluation(dla=dla, a_buf_size=10, stack=Stack(1, st3_di), tile_size=(4, 4), tile_type='M', is_feature_merging=False)
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
    #     print("lower_limit_of_abuf_for_stack:", cme.lower_limit_of_abuf_for_stack)
    #     print("residual_tile_data_amount    :", cme.residual_tile_data_amount)

    #     print("a_buf_size                   :", cme.a_buf_size/1024)    
    #     print("first_true_id                :", cme.first_true_id              )           
    #     print("x_merging_data_amount        :", cme.x_merging_data_amount      )                   
    #     print("y_merging_data_amount        :", cme.y_merging_data_amount      )                   
    #     print("number_of_tile_in_row        :", cme.number_of_tile_in_row      )                   
    #     print("number_of_tile_in_col        :", cme.number_of_tile_in_col      )                   
    #     print("tile_number_of_current_type  :", cme.tile_number_of_current_type)                           
    #     print("merging_length               :", cme.merging_length             )               
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
    
    
    
    