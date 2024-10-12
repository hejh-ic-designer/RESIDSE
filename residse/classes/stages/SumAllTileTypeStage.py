from typing import Generator, Callable, List, Tuple, Any
from residse.classes.stages.Stage import Stage
from residse.classes.cost_model.cost_model import CostModelEvaluation
from residse.classes.workload.stack import Stack
from residse.classes.workload.tile_gen import TileSizeGenerator, TileTypeGenerator
from utils import sum_cme
import logging

logger = logging.getLogger(__name__)


class SumAllTileTypeStage(Stage):
    def __init__(self, list_of_callables, tile_size, stack: Stack, **kwargs):
        super().__init__(list_of_callables, **kwargs)
        self.tile_size = tile_size
        self.stack = stack
        self.type_lst = TileTypeGenerator(self.tile_size, self.stack).run()

    def run(self):
        self.cme_of_types = []
        self.sum_cme = None
        for ttype in self.type_lst:
            kwargs = self.kwargs.copy()
            kwargs["tile_type"] = ttype
            kwargs["tile_size"] = self.tile_size
            kwargs["stack"] = self.stack
            substage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
            for cme in substage.run():
                if cme.ema is None:
                    self.cme_of_types.append(None)
                else:
                    self.cme_of_types.append(cme)

        self.sum_cme = sum_cme(self.cme_of_types)       
        yield self.sum_cme, self.cme_of_types
