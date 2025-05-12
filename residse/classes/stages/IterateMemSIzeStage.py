from typing import Generator, Callable, List, Tuple, Any
from residse.classes.stages.Stage import Stage
from residse.classes.hardware.HardwareGenerator import HardwareGenerator
from residse.classes.hardware.dla import Dla
from utils import sum_cme
import logging
logger = logging.getLogger(__name__)


class IterateMemSizeStage(Stage):
    def __init__(self, list_of_callables, dla: Dla, is_fixed_tsize, is_fixed_memsize, fixed_mem_size, **kwargs):
        super().__init__(list_of_callables, **kwargs)
        self.dla = dla
        self.is_fixed_tsize = is_fixed_tsize
        self.is_fixed_memsize = is_fixed_memsize
        self.fixed_mem_size = fixed_mem_size
        if is_fixed_tsize and not is_fixed_memsize:
            self.a_buf_size_list = dla.a_buf.get_size_list()
        elif not is_fixed_tsize and is_fixed_memsize:
            self.a_buf_size_list = [fixed_mem_size]

    def run(self):
        logger.info(f'Running a buf size at: {self.a_buf_size_list}')
        for a_buf_size in self.a_buf_size_list:
            # self.cme_of_stacks = [] # all stack cmes in a list
            # self.sum_cme = None     # sum of all stack cmes
            kwargs = self.kwargs.copy()
            kwargs['a_buf_size'] = a_buf_size
            kwargs['dla'] = self.dla
            sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
            logger.info(f'Start running a_buf size at {a_buf_size} '+'---'*30)
            for cme, extra_info in sub_stage.run():
                self.cme_of_stacks.append(cme)

            # any stack is not-able to Layer Fusion --> sum_cme = None
            self.sum_cme = sum_cme(self.cme_of_stacks)   
            if self.sum_cme is None:
                logger.info(f"skip a buf size at {a_buf_size}")
            yield self.sum_cme, (self.cme_of_stacks, a_buf_size, extra_info)

