# Installation

1. install [Anaconda](https://docs.anaconda.com/anaconda/install/index.html) environment

2. clone整个目录文件 (`git clone` this repo)

3. Use a terminal or an Anaconda Prompt for the following steps:

   -  `cd` to RESIDUAL-DSE repo
   -  create environment form `environment.yml`
       ```
       conda env create -f environment.yml
       ```
   -  activate new enviroment:
       ```
       conda activate RESIenv
       ```

# Quick Start

1. `cd` to RESIDUAL-DSE repo

2. run `python main_dse.py --nn sesr --hw sesr_1 --merge --tile_size 32 32` to get EDP proling when feature merging

3. run `python main_dse.py --nn sesr --hw sesr_1 --tile_size 32 32` to get EDP proling when not feature merging

4. set `experiment_id` in `plot_two_lines.py` to `sesr_1--sesr--True--fix_tsize32x32`

5. run `python plot_two_lines.py` to get EDP difference between merging and not-merging


# set your experiment

1. workload is defined at path `residse/inputs/WL`

2. hardware is defined at path `residse/inputs/HW`

3. terminal command prompt: 
   1. --nn : workload name
   2. --hw : hardware name
   3. --merge : feature merging, default means not-feature merging
   4. --tile_size : fixed tile size at all stacks, e.g. `--tile_size 32 32` will fix tile_size to h=32, w=32
   5. --tile_points : tile size points to explore, e.g. [10, 10] will run 100 tile_size points


# customization protocol

## workload

1. 使用 --- 来切割stack
2. 必选条目: op, in_resb(表示当前层是否位于residual block内部), dim
3. 可选条目: stride，缺省时默认为1
4. 目前可支持的op: conv, pool, fc, outer_add(表示大残差)
5. dim 列表必须是六个元素，按照顺序分别代表 h, w, oc, ic, fx, fy
6. 一个stack内，不支持超过一个 stride = 2 的卷积层

## hardware 

1. 硬件定义文件中的buffer容量单位均为 KB
2. dram bandwidth 单位是 Byte/cycle


