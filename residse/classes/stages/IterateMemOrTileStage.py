from typing import Generator, Callable, List, Tuple, Any
from residse.classes.stages.Stage import Stage
from residse.classes.hardware.HardwareGenerator import HardwareGenerator
from residse.classes.hardware.dla import Dla
from utils import sum_cme
from residse.classes.cost_model.cost_model import CostModelEvaluation
from residse.classes.workload.stack import Stack
from residse.classes.workload.tile_gen import TileSizeGenerator
import sys
import logging

logger = logging.getLogger(__name__)

class IterateMemOrTileStage(Stage):
    def __init__(self, list_of_callables, dla: Dla, is_fixed_tsize, is_fixed_memsize, fixed_mem_size, fixed_tile_size, stacks: List[Stack], **kwargs):
        super().__init__(list_of_callables, **kwargs)
        self.dla = dla
        self.is_fixed_tsize = is_fixed_tsize
        self.is_fixed_memsize = is_fixed_memsize
        self.fixed_mem_size = fixed_mem_size
        self.fixed_tile_size = fixed_tile_size
        self.stacks = stacks
        if is_fixed_tsize and not is_fixed_memsize:
            self.a_buf_size_list = dla.a_buf.get_size_list()

    def run(self):

        if self.is_fixed_tsize and not self.is_fixed_memsize:
        # 固定tile size迭代不同mem size
            logger.info(f'Running a buf size at: {self.a_buf_size_list}')
            for a_buf_size in self.a_buf_size_list:
                self.cme_of_stacks = [] # all stack cmes in a list
                self.sum_cme = None     # sum of all stack cmes     
                kwargs = self.kwargs.copy()
                kwargs['a_buf_size'] = a_buf_size
                kwargs['dla'] = self.dla
                kwargs['tile_size'] = self.fixed_tile_size
                kwargs['stacks'] = self.stacks
                sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
                logger.info(f'Start running a_buf size at {a_buf_size} '+'---'*30)
                for cme, extra_info in sub_stage.run():
                    self.cme_of_stacks.append(cme)
                    if a_buf_size == 51:
                        print(cme.ema)
                        print(cme.la)
                        print(cme.edp)


                # any stack is not-able to Layer Fusion --> sum_cme = None
                self.sum_cme = sum_cme(self.cme_of_stacks)   
                if self.sum_cme is None:
                    logger.info(f"skip a buf size at {a_buf_size}")
                yield self.sum_cme, (self.cme_of_stacks, a_buf_size, self.fixed_tile_size, extra_info)

        elif not self.is_fixed_tsize and self.is_fixed_memsize:
            self.gen = TileSizeGenerator(fixed_tile_size=self.fixed_tile_size, stacks=self.stacks)
            for tile_size in self.gen.run():
                self.cme_of_stacks = [] # all stack cmes in a list
                self.sum_cme = None     # sum of all stack cmes            
                kwargs = self.kwargs.copy()
                kwargs['a_buf_size'] = self.fixed_mem_size
                kwargs['dla'] = self.dla
                kwargs['tile_size'] = tile_size
                kwargs['stacks'] = self.stacks
                sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
                for cme, extra_info in sub_stage.run():
                    self.cme_of_stacks.append(cme)

                # any stack is not-able to Layer Fusion --> sum_cme = None
                self.sum_cme = sum_cme(self.cme_of_stacks)
                if self.sum_cme is None:
                    sys.exit("error: self.sum_cme is None -- 固定mem size迭代tile size实验中, mem size设置太低不够layer fusion存储io tile")
                yield self.sum_cme, (self.cme_of_stacks, self.fixed_mem_size, tile_size, extra_info)