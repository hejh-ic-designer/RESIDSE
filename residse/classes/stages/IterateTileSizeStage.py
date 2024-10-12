from typing import Generator, Callable, List, Tuple, Any
from residse.classes.stages.Stage import Stage
from residse.classes.cost_model.cost_model import CostModelEvaluation
from residse.classes.workload.stack import Stack
from residse.classes.workload.tile_gen import TileSizeGenerator
import logging

logger = logging.getLogger(__name__)


class IterateTileSizeStage(Stage):
    def __init__(self, list_of_callables, fixed_tile_size, stack: Stack, nb_of_points, **kwargs):
        super().__init__(list_of_callables, **kwargs)
        self.fixed_tile_size = fixed_tile_size
        self.stack = stack
        self.nb_of_points = nb_of_points


    def run(self):
        self.gen = TileSizeGenerator(fixed_tile_size=self.fixed_tile_size, stack=self.stack, nb_of_points=self.nb_of_points)
        for tile_size in self.gen.run():
            kwargs = self.kwargs.copy()
            kwargs['tile_size'] = tile_size
            kwargs['stack'] = self.stack
            sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
            for cme, extra_info in sub_stage.run():
                yield cme, (tile_size, extra_info)

