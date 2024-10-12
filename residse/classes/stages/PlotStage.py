import os
import logging
from typing import Generator, Any, List, Tuple
from residse.classes.cost_model.cost_model import CostModelEvaluation
from residse.classes.stages.Stage import Stage
from residse.visualization.plot_cme import plot_cme_edp, plot_cme_ema
logger = logging.getLogger(__name__)

class PlotStage(Stage):
    def __init__(self, list_of_callables, *, dump_filename_pattern, **kwargs):
        super().__init__(list_of_callables, **kwargs)
        self.dump_filename_pattern = dump_filename_pattern


    def run(self):
        # self.plot_path = self.dump_filename_pattern.replace("?.json", "")
        self.kwargs["dump_filename_pattern"] = self.dump_filename_pattern
        substage = self.list_of_callables[0](self.list_of_callables[1:], **self.kwargs)
        self.cmes = []
        for cme, extra_info in substage.run():
            self.cmes.append(cme)
            yield cme, extra_info

        logger.info(f'Charting all result information in directory, Please Wait......')
        self.plot_total()

    def plot_total(self):
        
        # edp
        file_bd_edp = self.dump_filename_pattern.replace("?.json", "edp.png")
        os.makedirs(os.path.dirname(file_bd_edp), exist_ok=True)
        plot_cme_edp(self.cmes, file_bd_edp)
        
        # ema
        file_bd_ema = self.dump_filename_pattern.replace("?.json", "ema.png")
        os.makedirs(os.path.dirname(file_bd_ema), exist_ok=True)
        plot_cme_ema(self.cmes, file_bd_ema)

        logger.info(f'plot CME breakdown at path {file_bd_ema}, {file_bd_edp}')


