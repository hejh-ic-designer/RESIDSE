import json
import logging
import os
import pickle
from typing import Any, Callable, Generator, List, Tuple

import numpy as np

from residse.classes.cost_model.cost_model import CostModelEvaluation
from residse.classes.stages.Stage import Stage

logger = logging.getLogger(__name__)


class CompleteSaveStage(Stage):
    """
    Class that passes through all results yielded by substages, but saves the results as a json list to a file
    at the end of the iteration.
    """

    def __init__(self, list_of_callables, *, dump_filename_pattern, **kwargs):
        super().__init__(list_of_callables, **kwargs)
        self.dump_filename_pattern = dump_filename_pattern

    def run(self):
        """
        Run the complete save stage by running the substage and saving the CostModelEvaluation json representation.
        """
        self.kwargs["dump_filename_pattern"] = self.dump_filename_pattern
        substage = self.list_of_callables[0](self.list_of_callables[1:], **self.kwargs)

        for cme, extra_info in substage.run():
            cme: CostModelEvaluation
            a_buf_size = extra_info[1]  # take a_buf_size from extra_info of next stage
            filename = self.dump_filename_pattern.replace("?", f"abuf_{a_buf_size}")
            self.save_to_json(cme, filename=filename)

            # print log
            if cme is not None:
                logger.info(f"Saved cme with energy {cme.en}, latency {cme.la} and edp {cme.edp} to {filename}")
            yield cme, extra_info

    def save_to_json(self, obj, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as fp:
            json.dump(obj, fp, default=self.complexHandler, indent=4)

    @staticmethod
    def complexHandler(obj):
        if hasattr(obj, "__jsonrepr__"):
            return obj.__jsonrepr__()
        elif obj is None:
            return {"EDP": 0}
        else:
            raise TypeError(
                f"Object of type {type(obj)} is not serializable. Create a __jsonrepr__ method."
            )


class SimpleSaveStage(Stage):
    """
    Class that passes through results yielded by substages, but saves the results as a json list to a file
    at the end of the iteration.
    In this simple version, only the energy total and latency total are saved.
    """

    def __init__(self, list_of_callables, *, dump_filename_pattern, **kwargs):
        """
        :param list_of_callables: see Stage
        :param dump_filename_pattern: filename string formatting pattern, which can use named field whose values will be
        in kwargs (thus supplied by higher level runnables)
        :param kwargs: any kwargs, passed on to substages and can be used in dump_filename_pattern
        """
        super().__init__(list_of_callables, **kwargs)
        self.dump_filename_pattern = dump_filename_pattern

    def run(self):
        """
        Run the simple save stage by running the substage and saving the CostModelEvaluation simple json representation.
        """
        self.kwargs["dump_filename_pattern"] = self.dump_filename_pattern
        substage = self.list_of_callables[0](self.list_of_callables[1:], **self.kwargs)

        for id, (cme, extra_info) in enumerate(substage.run()):
            cme: CostModelEvaluation
            if type(cme.layer) == list:
                filename = self.dump_filename_pattern.replace("?", "overall_simple")
            else:
                filename = self.dump_filename_pattern.replace(
                    "?", f"{cme.layer}_simple"
                )
            self.save_to_json(cme, filename=filename)
            logger.info(
                f"Saved {cme} with energy {cme.energy_total:.3e} and latency {cme.latency_total2:.3e} to {filename}"
            )
            yield cme, extra_info

    def save_to_json(self, obj, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as fp:
            json.dump(obj, fp, default=self.complexHandler, indent=4)

    @staticmethod
    def complexHandler(obj):
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, np.int32):
            return int(obj)
        if hasattr(obj, "__simplejsonrepr__"):
            return obj.__simplejsonrepr__()
        else:
            raise TypeError(
                f"Object of type {type(obj)} is not serializable. Create a __simplejsonrepr__ method."
            )


class PickleSaveStage(Stage):
    """
    Class that dumps all received CMEs into a list and saves that list to a pickle file.
    """

    def __init__(self, list_of_callables, *, dump_filename_pattern, is_fixed_tsize, is_fixed_memsize, **kwargs):
        """
        :param list_of_callables: see Stage
        :param dump_filename_pattern: output pickle filename pattern
        :param is_fixed_tsize: whether tile size is fixed
        :param is_fixed_memsize: whether memory size is fixed
        :param kwargs: any kwargs, passed on to substages and can be used in dump_filename_pattern
        """
        super().__init__(list_of_callables, **kwargs)
        self.dump_filename_pattern = dump_filename_pattern
        self.pickle_file_name = self.dump_filename_pattern.replace("?.json", "all_cmes.pickle")
        self.is_fixed_tsize = is_fixed_tsize
        self.is_fixed_memsize = is_fixed_memsize

    def run(self):
        """
        Run the simple save stage by running the substage and saving the CostModelEvaluation simple json representation.
        This should be placed above a ReduceStage such as the SumStage, as we assume the list of CMEs is passed as extra_info
        """
        self.kwargs["dump_filename_pattern"] = self.dump_filename_pattern
        self.kwargs["is_fixed_tsize"] = self.is_fixed_tsize
        self.kwargs["is_fixed_memsize"] = self.is_fixed_memsize

        substage = self.list_of_callables[0](self.list_of_callables[1:], **self.kwargs)
        all_cmes = []
        for cme, extra_info in substage.run():
            all_cmes.append(cme)
            yield cme, extra_info
        
        #固定tile_size，迭代mem_size时，保存pickle文件用于绘制曲线图
        if self.is_fixed_tsize and not self.is_fixed_memsize:
            os.makedirs(os.path.dirname(self.pickle_file_name), exist_ok=True)
            with open(self.pickle_file_name, "wb") as handle:
                pickle.dump(all_cmes, handle, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(
                f"Saved pickled list of {len(all_cmes)} CMEs to {self.pickle_file_name}."
            )
