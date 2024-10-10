from typing import Generator, Callable, List, Tuple, Any
from residse.classes.stages.Stage import Stage
from residse.classes.workload.stack import Stack
import logging

logger = logging.getLogger(__name__)


class IterateStackStage(Stage):
    def __init__(self, list_of_callables, stacks: List[Stack], **kwargs):
        super().__init__(list_of_callables, **kwargs)
        self.stacks = stacks
    
    
    def run(self):
        for stack in self.stacks:
            kwargs = self.kwargs.copy()
            kwargs['stack'] = stack
            sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
            logger.info(f'Running stack of {stack} ...')
            for cme, extra_info in sub_stage.run():
                yield cme, extra_info

    














