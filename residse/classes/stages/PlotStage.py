import os
import logging
from typing import Generator, Any, List, Tuple
from residse.classes.cost_model.cost_model import CostModelEvaluation
from residse.classes.stages.Stage import Stage
from residse.visualization.plot_cme import plot_cme_edp, plot_cme_ema, plot_cme_tileiter
logger = logging.getLogger(__name__)

class PlotStage(Stage):
    def __init__(self, list_of_callables, *, dump_filename_pattern, is_fixed_tsize, is_fixed_memsize, **kwargs):
        super().__init__(list_of_callables, **kwargs)
        self.dump_filename_pattern = dump_filename_pattern
        self.is_fixed_tsize = is_fixed_tsize
        self.is_fixed_memsize = is_fixed_memsize

    def run(self):
        # self.plot_path = self.dump_filename_pattern.replace("?.json", "")
        self.kwargs["dump_filename_pattern"] = self.dump_filename_pattern
        self.kwargs["is_fixed_tsize"] = self.is_fixed_tsize
        self.kwargs["is_fixed_memsize"] = self.is_fixed_memsize

        substage = self.list_of_callables[0](self.list_of_callables[1:], **self.kwargs)
        self.cmes = []
        for cme, extra_info in substage.run():
            self.cmes.append(cme)
            yield cme, extra_info

        logger.info(f'Charting all result information in directory, Please Wait......')
        self.plot_total()

    def plot_total(self):
        
        if self.is_fixed_tsize and not self.is_fixed_memsize:
        # 固定tile size迭代mem size，绘制不同merge/rda策略下，edp/ema随着mem size变化曲线
            # edp
            file_bd_edp = self.dump_filename_pattern.replace("?.json", "edp.png")
            os.makedirs(os.path.dirname(file_bd_edp), exist_ok=True)
            plot_cme_edp(self.cmes, file_bd_edp)
            
            # ema
            file_bd_ema = self.dump_filename_pattern.replace("?.json", "ema.png")
            os.makedirs(os.path.dirname(file_bd_ema), exist_ok=True)
            plot_cme_ema(self.cmes, file_bd_ema)

            logger.info(f'plot CME breakdown at path {file_bd_ema}, {file_bd_edp}')
        
        elif not self.is_fixed_tsize and self.is_fixed_memsize:
        # 固定mem size迭代tile size，绘制merge+rda策略下，edp随着tile size变化热力图
            file_bd_tileiter = self.dump_filename_pattern.replace("?.json", "tileiter.png")
            os.makedirs(os.path.dirname(file_bd_tileiter), exist_ok=True)
            plot_cme_tileiter(self.cmes, file_bd_tileiter)

            logger.info(f'plot CME breakdown at path {file_bd_tileiter}')
