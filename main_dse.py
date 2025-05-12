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
parser.add_argument( "--rda", action='store_true', help="bool, using reuse-distance-aware caching strategy?")
parser.add_argument( "--tile_size", metavar="Fixed Tile Size", type=int, nargs='+', required=False, help="use fixed tile size [h, w], e.g. 4 6. If provided, --mem_size cannot be used." )
parser.add_argument( "--mem_size", metavar="Fixed Memory Size", type=int, required=False, help="specify memory size (KB) for iteration with tile size. If provided, --tile_size cannot be used.")
# parser.add_argument( "--tile_points", metavar="Tile Size Points", type=int, nargs='+', required=False, help="tile size points to explore, e.g. 10 10 will run 100 points" )
args = parser.parse_args()

# 要么固定tile_size，迭代mem_size，绘制曲线图（所有模型通用）
# 要么固定mem_size，迭代tile_size，绘制热力图（只针对没有下采样的图像增强网络有意义）
if args.tile_size and args.mem_size:
    parser.error("Cannot provide both --tile_size and --mem_size arguments at the same time.")
elif not args.tile_size and not args.mem_size:
    parser.error("Either --tile_size or --mem_size must be provided, but not both.")

if args.merge and not args.rda:
    parser.error("Feature merging must be used along with reuse-distance-aware caching strategy.")

# 当 nn 是 'resnet18' 不做tile_size迭代实验
if args.nn == 'resnet18' and args.mem_size and not args.tile_size:
    parser.error("When nn is 'resnet18', iterating tile_size is not supported.")

# set experiment id
if args.tile_size:
    experiment_id = f"{args.hw}--{args.nn}--merge_{args.merge}--rda_{args.rda}--fix_tile_size{args.tile_size[0]}x{args.tile_size[1]}"
elif args.mem_size:
    experiment_id = f"{args.hw}--{args.nn}--merge_{args.merge}--rda_{args.rda}--fix_mem_size_{args.mem_size}KB"

# 当固定 mem_size 迭代 tile_size 时，必须同时使用 --merge 和 --rda
if args.mem_size and not args.tile_size:
    if not (args.merge and args.rda):
        parser.error("When iterating tile_size with a fixed mem_size, both feature merging and reuse-distance-aware caching strategy must be used.")

# set stage pipeline
StagesPipeline = [
    HardwareParserStage,    # 解析硬件
    WorkloadParserStage,    # 解析网络模型
    PickleSaveStage,        # 保存评估结果到pickle文件
    PlotStage,              # 绘图
    # IterateMemSizeStage,    # 迭代 mem size
    # IterateStackStage,      # 迭代各个 stack
    # MinimalEDPStage,        # 针对不同tile size, 筛选最小 EDP 设计点
    # IterateTileSizeStage,   # 迭代tile size设计点
    IterateMemOrTileStage,
    IterateStackStage,      # 迭代各个 stack
    SumAllTileTypeStage,    # 累加一个stack中所有layer所有tile type的评估结果
    ResidseCostModelStage,  # 固定mem size, stack, tile size, tile type，评估其EDP
]
mainstage = MainStage(
    list_of_callables=StagesPipeline,
    hw_path=f"residse/inputs/HW/{args.hw}.json",
    workload_path=f"residse/inputs/WL/{args.nn}.yml",
    dump_filename_pattern=f"outputs/{experiment_id}/?.json",
    is_feature_merging=args.merge,
    is_rda=args.rda,
    is_fixed_tsize=True if args.tile_size else False,
    is_fixed_memsize=True if args.mem_size else False,
    fixed_tile_size=args.tile_size,  # [h, w]
    fixed_mem_size=args.mem_size,  # [h, w]
    # nb_of_points=args.tile_points,  # [nb_h, nb_w]
)

# start run
logger.info(f"Runing ResiDSE Experiment: ......")
mainstage.run()


"""
loop 1: mem/tile size
loop 2: stack
loop 3: tile type
"""
