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

# cal version

## srgan 

1. `python main_dse.py --nn srgan --hw srgan_1 --merge --rda --tile_size 32 32`
2. `python main_dse.py --nn srgan --hw srgan_1 --rda --tile_size 32 32`
3. `python main_dse.py --nn srgan --hw srgan_1 --tile_size 32 32`
4. `python plot_compare_lines.py --id srgan_1--srgan --tsize 32 32`

5. `python main_dse.py --nn srgan --hw srgan_1 --merge --rda --tile_size 16 16`
6. `python main_dse.py --nn srgan --hw srgan_1 --rda --tile_size 16 16`
7. `python main_dse.py --nn srgan --hw srgan_1 --tile_size 16 16`
8. `python plot_compare_lines.py --id srgan_1--srgan --tsize 16 16`

9. `python main_dse.py --nn srgan --hw srgan_1 --merge --rda --tile_size 8 8`
10. `python main_dse.py --nn srgan --hw srgan_1 --rda --tile_size 8 8`
11. `python main_dse.py --nn srgan --hw srgan_1 --tile_size 8 8`
12. `python plot_compare_lines.py --id srgan_1--srgan --tsize 8 8`

13. `python main_dse.py --nn srgan --hw srgan_1 --merge --rda --tile_size 4 4`
14. `python main_dse.py --nn srgan --hw srgan_1 --rda --tile_size 4 4`
15. `python main_dse.py --nn srgan --hw srgan_1 --tile_size 4 4`
16. `python plot_compare_lines.py --id srgan_1--srgan --tsize 4 4`

17. `python main_dse.py --nn srgan --hw srgan_1 --merge --rda --tile_size 1 480`
18. `python main_dse.py --nn srgan --hw srgan_1 --rda --tile_size 1 480`
19. `python main_dse.py --nn srgan --hw srgan_1 --tile_size 1 480`
20. `python plot_compare_lines.py --id srgan_1--srgan --tsize 1 480`

## resnet18 

21. `python main_dse.py --nn resnet18 --hw res18_1 --merge --rda --tile_size 4 4`
22. `python main_dse.py --nn resnet18 --hw res18_1 --rda --tile_size 4 4`
23. `python main_dse.py --nn resnet18 --hw res18_1 --tile_size 4 4`
24. `python plot_compare_lines.py --id res18_1--resnet18 --tsize 4 4`

25. `python main_dse.py --nn resnet18 --hw res18_1 --merge --rda --tile_size 1 112`
26. `python main_dse.py --nn resnet18 --hw res18_1 --rda --tile_size 1 112`
27. `python main_dse.py --nn resnet18 --hw res18_1 --tile_size 1 112`
28. `python plot_compare_lines.py --id res18_1--resnet18 --tsize 1 112`

29. `python main_dse.py --nn resnet18 --hw res18_1 --merge --rda --tile_size 3 3`
30. `python main_dse.py --nn resnet18 --hw res18_1 --rda --tile_size 3 3`
31. `python main_dse.py --nn resnet18 --hw res18_1 --tile_size 3 3`
32. `python plot_compare_lines.py --id res18_1--resnet18 --tsize 3 3`

33. `python main_dse.py --nn resnet18 --hw res18_1 --merge --rda --tile_size 2 2`
34. `python main_dse.py --nn resnet18 --hw res18_1 --rda --tile_size 2 2`
35. `python main_dse.py --nn resnet18 --hw res18_1 --tile_size 2 2`
36. `python plot_compare_lines.py --id res18_1--resnet18 --tsize 2 2`