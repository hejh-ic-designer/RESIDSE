from typing import Generator, Callable, List, Tuple, Any
from residse.classes.stages.Stage import Stage
from residse.classes.cost_model.cost_model import CostModelEvaluation
import logging

logger = logging.getLogger(__name__)


class ResidseCostModelStage(Stage):
    def __init__(
        self,
        list_of_callables: List[Callable],
        *,
        dla,
        a_buf_size,
        stack,
        tile_size,
        tile_type,
        is_feature_merging,
        is_rda,
        **kwargs
    ):
        super().__init__(list_of_callables, **kwargs)
        (
            self.dla,
            self.a_buf_size,
            self.stack,
            self.tile_size,
            self.tile_type,
            self.is_feature_merging,
            self.is_rda,
        ) = (dla, a_buf_size, stack, tile_size, tile_type, is_feature_merging, is_rda)

    def run(self):
        self.cme = CostModelEvaluation(
            dla=self.dla,
            a_buf_size=self.a_buf_size,
            stack=self.stack,
            tile_size=self.tile_size,
            tile_type=self.tile_type,
            is_feature_merging=self.is_feature_merging,
            is_rda=self.is_rda,
        )
        yield self.cme

    def is_leaf(self) -> bool:
        return True
