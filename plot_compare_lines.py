# from residse.visualization.plot_cme import plot_two_lines_ema, plot_two_lines_edp, plot_two_lines_tsize_edp, plot_two_lines_tsize_ema
from residse.visualization.plot_cme import plot_compare_lines_edp
import argparse

def main():
    parser = argparse.ArgumentParser(description='生成模型配置字符串')
    parser.add_argument('--id', required=True, help='模型标识符 (例: srgan_1--srgan)')
    parser.add_argument('--tsize', nargs=2, type=int, required=True,
                      help='分块尺寸 (例: 32 32)')
    
    args = parser.parse_args()

    # 参数解析校验
    if '--' not in args.id:
        raise ValueError("--id格式错误，应包含双连字符(--)")
    prefix, model = args.id.split('--', 1)  # 确保只分割一次
    
    # 构建尺寸字符串
    tsize_str = f"{args.tsize[0]}x{args.tsize[1]}"

    # 直接变量赋值
    string_merge_rda = f"{prefix}--{model}--merge_True--rda_True--fix_tile_size{tsize_str}"
    string_nomerge_rda = f"{prefix}--{model}--merge_False--rda_True--fix_tile_size{tsize_str}"
    string_nomerge_norda = f"{prefix}--{model}--merge_False--rda_False--fix_tile_size{tsize_str}"

    # # 直接输出变量值
    # print(string_merge_rda)
    # print(string_nomerge_rda)
    # print(string_nomerge_norda)
    pkl_path_1 = f'outputs/{string_merge_rda}/all_cmes.pickle'
    pkl_path_2 = f'outputs/{string_nomerge_rda}/all_cmes.pickle'
    pkl_path_3 = f'outputs/{string_nomerge_norda}/all_cmes.pickle'
    save_path_edp = f'outputs/{string_merge_rda}/two_lines_edp.png'

    # plot merge and not-merge
    # plot_two_lines_ema(pkl_paths=[pkl_path_1, pkl_path_2], save_path=save_path_ema) # 只要edp
    plot_compare_lines_edp(pkl_paths=[pkl_path_1, pkl_path_2, pkl_path_3], save_path=save_path_edp)
    print(f'##### PLOT DONE @ path: {string_merge_rda} and {save_path_edp}')

if __name__ == "__main__":
    main()




# def plot_2_merge():
#     """
#     feature merging VS. non-feature merging
#     """
#     #= set experiment_id
#     experiment_id = 'res18_1--resnet18--True--fix_tsize4x2'
#     # experiment_id = 'sesr_1--sesr--True--fix_tsize32x4'
#     # experiment_id = 'srgan_1--srgan--True--fix_tsize32x4'

#     # extract path
#     other_experiment_id = experiment_id.replace("True", "False")
#     pkl_path_1 = f'outputs/{experiment_id}/all_cmes.pickle'
#     pkl_path_2 = f'outputs/{other_experiment_id}/all_cmes.pickle'
#     save_path_ema = f'outputs/{experiment_id}/two_lines_ema.png'
#     save_path_edp = f'outputs/{experiment_id}/two_lines_edp.png'

#     # plot merge and not-merge
#     # plot_two_lines_ema(pkl_paths=[pkl_path_1, pkl_path_2], save_path=save_path_ema) # 只要edp
#     plot_two_lines_edp(pkl_paths=[pkl_path_1, pkl_path_2], save_path=save_path_edp)
#     print(f'##### PLOT DONE @ path: {save_path_ema} and {save_path_edp}')


# def plot_2_tsize():
#     """
#     fixed tile size VS. free tile size
#     """
#     #= set experiment_id
#     experiment_id = 'res18_1--resnet18--True--fix_tsize4x2'
#     # experiment_id = 'sesr_1--sesr--True--fix_tsize32x4'
#     # experiment_id = 'srgan_1--srgan--True--fix_tsize32x4'

#     # extract path
#     other_experiment_id = experiment_id.replace("--fix_tsize4x2", "")   #! 注意在这里改 nxn
#     pkl_path_1 = f'outputs/{experiment_id}/all_cmes.pickle'
#     pkl_path_2 = f'outputs/{other_experiment_id}/all_cmes.pickle'
#     save_path_ema = f'outputs/{experiment_id}/two_lines_tsize_ema.png'
#     save_path_edp = f'outputs/{experiment_id}/two_lines_tsize_edp.png'


#     # plot free-size and fixed-size
#     # plot_two_lines_tsize_ema(pkl_paths=[pkl_path_1, pkl_path_2], save_path=save_path_ema)
#     plot_two_lines_tsize_edp(pkl_paths=[pkl_path_1, pkl_path_2], save_path=save_path_edp)
#     print(f'##### PLOT DONE @ path: {save_path_ema}')



# if __name__ == '__main__':
#     plot_2_tsize()
#     plot_2_merge()
