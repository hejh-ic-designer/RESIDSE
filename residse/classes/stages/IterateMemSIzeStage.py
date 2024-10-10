from typing import Generator, Callable, List, Tuple, Any
from residse.classes.stages.Stage import Stage
from residse.classes.hardware.HardwareGenerator import HardwareGenerator
from residse.classes.hardware.dla import Dla
import logging
logger = logging.getLogger(__name__)


class IterateMemSizeStage(Stage):
    def __init__(self, list_of_callables, dla: Dla, **kwargs):
        super().__init__(list_of_callables, **kwargs)
        self.dla = dla
        self.a_buf_size_list = dla.a_buf.get_size_list()
        self.sum_cme = None     # sum of all stack cmes
        self.cme_of_stacks = [] # all stack cmes in a list
    
    
    def run(self):
        for a_buf_size in self.a_buf_size_list:
            kwargs = self.kwargs.copy()
            kwargs['a_buf_size'] = a_buf_size
            kwargs['dla'] = self.dla
            sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
            logger.info(f'Start running a_buf size at {a_buf_size} ...')
            for cme, extra_info in sub_stage.run():
                self.cme_of_stacks.append(cme)
                if self.sum_cme is None:
                    self.sum_cme = cme
                else:
                    self.sum_cme += cme
            yield self.sum_cme, (self.cme_of_stacks, extra_info)

