# Installation

# Quick Start

# set your experiment

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