import argparse
import logging as _logging
from residse.classes.stages import *

_logging_level = _logging.INFO
_logging_format = (
    "%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
)
_logging.basicConfig(level=_logging_level, format=_logging_format)
logger = _logging.getLogger(__name__)

# set arg parser
parser = argparse.ArgumentParser(description="Setup residse inputs")
parser.add_argument( "--nn", metavar="Network name", required=True, help="module name to networks, e.g. resnet18")
parser.add_argument( "--hw", metavar="Hardware name", required=True, help="file name to the user-defined json accelerator, e.g. example_1")
parser.add_argument( "--merge", action='store_true', help="bool, using feature merging stategy?")
parser.add_argument( "--tile_size", metavar="Fixed Tile Size", type=int, nargs='+', required=False, help="use fixed tile size [h, w], e.g. [4, 6]" )
parser.add_argument( "--tile_points", metavar="Tile Size Points", required=False, help="tile size points to explore, e.g. [10, 10] will run 100 points" )
args = parser.parse_args()

# set experiment id
if args.tile_size:
    experiment_id = f"{args.hw}--{args.nn}--{args.merge}--fix_tsize{args.tile_size[0]}x{args.tile_size[1]}"
else:
    experiment_id = f"{args.hw}--{args.nn}--{args.merge}"


# set stage pipeline
StagesPipeline = [
    HardwareParserStage,
    WorkloadParserStage,
    CompleteSaveStage,
    PlotStage,
    IterateMemSizeStage,
    IterateStackStage,
    MinimalEDPStage,
    IterateTileSizeStage,
    SumAllTileTypeStage,
    ResidseCostModelStage,
]
mainstage = MainStage(
    list_of_callables=StagesPipeline,
    hw_path=f"residse/inputs/HW/{args.hw}.json",
    workload_path=f"residse/inputs/WL/{args.nn}.yml",
    dump_filename_pattern=f"outputs/{experiment_id}/?.json",
    is_feature_merging=args.merge,
    fixed_tile_size=args.tile_size,  # [h, w]
    nb_of_points=args.tile_points,  # [nb_h, nb_w]
)

# start run
logger.info(f"Runing ResiDSE Experiment: ......")
mainstage.run()


"""
loop 1: mem size
loop 2: stack
loop 3: tile size
loop 4: tile type
"""