from typing import Generator, Callable, List, Tuple, Any
from residse.classes.stages.Stage import Stage
from residse.classes.cost_model.cost_model import CostModelEvaluation
from residse.classes.workload.stack import Stack
from residse.classes.workload.tile_gen import TileSizeGenerator
from utils import sum_cme
import logging

logger = logging.getLogger(__name__)


class IterateTileSizeStage(Stage):
    # def __init__(self, list_of_callables, fixed_tile_size, stack: Stack, nb_of_points, **kwargs):
    def __init__(self, list_of_callables, fixed_tile_size, stack: Stack, **kwargs):
        super().__init__(list_of_callables, **kwargs)
        self.fixed_tile_size = fixed_tile_size
        self.stack = stack
        # self.nb_of_points = nb_of_points


    def run(self):
        # self.gen = TileSizeGenerator(fixed_tile_size=self.fixed_tile_size, stack=self.stack, nb_of_points=self.nb_of_points)
        self.gen = TileSizeGenerator(fixed_tile_size=self.fixed_tile_size, stack=self.stack)
        for tile_size in self.gen.run():
            self.cme_of_stacks = [] # all stack cmes in a list
            self.sum_cme = None     # sum of all stack cmes            
            kwargs = self.kwargs.copy()
            kwargs['tile_size'] = tile_size
            kwargs['stack'] = self.stack
            sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
            for cme, extra_info in sub_stage.run():
                # yield cme, (tile_size, extra_info)
                self.cme_of_stacks.append(cme)

            # any stack is not-able to Layer Fusion --> sum_cme = None
            self.sum_cme = sum_cme(self.cme_of_stacks)   
            yield self.sum_cme, (self.cme_of_stacks, tile_size, extra_info)