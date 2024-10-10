from typing import Generator, Callable, List, Tuple, Any
from residse.classes.stages.Stage import Stage
from residse.classes.hardware.HardwareGenerator import HardwareGenerator
from residse.classes.workload.WorkloadParser import WorkloadParser
import logging
logger = logging.getLogger(__name__)


class HardwareParserStage(Stage):
    def __init__(self, list_of_callables: List[Callable], hw_path: str, **kwargs):
        super().__init__(list_of_callables, **kwargs)
        self.dla = HardwareGenerator(json_hw=hw_path).get_dla()
        
    def run(self):
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], dla=self.dla, **self.kwargs)
        for cme, extra_info in sub_stage.run():
            yield cme, extra_info


class WorkloadParserStage(Stage):
    def __init__(self, list_of_callables: List[Callable], workload_path: str, **kwargs):
        super().__init__(list_of_callables, **kwargs)
        self.stacks = WorkloadParser(yaml_path=workload_path).get_stacks()

    def run(self):
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], stacks=self.stacks, **self.kwargs)
        for cme, extra_info in sub_stage.run():
            yield cme, extra_info
    
