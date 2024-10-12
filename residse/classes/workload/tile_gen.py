from typing import Generator, Callable, List, Tuple, Any
from residse.classes.workload.stack import Stack
from itertools import product
import logging
logger = logging.getLogger(__name__)

class TileSizeGenerator:

    def __init__(self, fixed_tile_size: Tuple[int] | None, stack: Stack, nb_of_points: Tuple[int]):
        self.fixed_tile_size = fixed_tile_size
        self.ofm_h = stack.ofm_h
        self.ofm_w = stack.ofm_w
        self.h_points = nb_of_points[0] if nb_of_points is not None else 10  #* default tile size profiling points
        self.w_points = nb_of_points[1] if nb_of_points is not None else 10  #* default tile size profiling points


    def run(self):
        if self.fixed_tile_size is not None:
            size_gen_lst = [self.fixed_tile_size]
        else:
            t_h_lst = self.generate_halves(self.ofm_h, self.h_points)   #todo 采用了比较简单的除以2算法
            t_w_lst = self.generate_halves(self.ofm_w, self.w_points)   #todo 采用了比较简单的除以2算法
            t_h_lst = list(filter(lambda x: x != 0, t_h_lst))   # remove 0
            t_w_lst = list(filter(lambda x: x != 0, t_w_lst))   # remove 0
            logger.debug(f'tile size profiling list: h in {t_h_lst}, w in {t_w_lst}')
            size_gen_lst = list(product(t_h_lst, t_w_lst))
        self.size_gen_lst = size_gen_lst
        
        for tile_size in self.size_gen_lst:
            yield tile_size


    @staticmethod
    def generate_halves(number, n):
        return [int(number / (2 ** i)) for i in range(n)]
    
    
class TileTypeGenerator:
    
    def __init__(self, tile_size: Tuple[int], stack: Stack):
        self.ofm_h = stack.ofm_h
        self.ofm_w = stack.ofm_w
        self.tile_size = tile_size
        
    def run(self):
        if self.tile_size == (self.ofm_h, self.ofm_w):
            return ['F']
        elif (self.tile_size[0] == self.ofm_h):
            return ['HL', 'HM', 'HR']
        elif (self.tile_size[1] == self.ofm_w):
            return ['WU', 'WM', 'WD']
        else:
            return ['LU', 'U', 'RU', 'L', 'M', 'R', 'LD', 'D', 'RD']

        
        
        


if __name__ == '__main__':
    st3_di =  { 7: {'op': 'conv', 'stride': 2, 'in_resb': True, 'dim': [28, 28, 128, 64, 3, 3]}, 
                8: {'op': 'conv', 'stride': 1, 'in_resb': True, 'dim': [270, 480, 128, 128, 3, 3]}}
    eg = TileSizeGenerator(fixed_tile_size=None, stack=Stack(1, st3_di), nb_of_points=[10, 10])
    poss = eg.run()
    print(list(poss))
    
    
    
    