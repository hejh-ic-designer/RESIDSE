from residse.visualization.plot_cme import plot_two_lines_ema, plot_two_lines_edp, plot_two_lines_tsize_edp, plot_two_lines_tsize_ema

def plot_2_merge():
    """
    feature merging VS. non-feature merging
    """
    #= set experiment_id
    experiment_id = 'res18_1--resnet18--True--fix_tsize4x2'
    # experiment_id = 'sesr_1--sesr--True--fix_tsize32x4'
    # experiment_id = 'srgan_1--srgan--True--fix_tsize32x4'

    # extract path
    other_experiment_id = experiment_id.replace("True", "False")
    pkl_path_1 = f'outputs/{experiment_id}/all_cmes.pickle'
    pkl_path_2 = f'outputs/{other_experiment_id}/all_cmes.pickle'
    save_path_ema = f'outputs/{experiment_id}/two_lines_ema.png'
    save_path_edp = f'outputs/{experiment_id}/two_lines_edp.png'

    # plot merge and not-merge
    # plot_two_lines_ema(pkl_paths=[pkl_path_1, pkl_path_2], save_path=save_path_ema) # 只要edp
    plot_two_lines_edp(pkl_paths=[pkl_path_1, pkl_path_2], save_path=save_path_edp)
    print(f'##### PLOT DONE @ path: {save_path_ema} and {save_path_edp}')


def plot_2_tsize():
    """
    fixed tile size VS. free tile size
    """
    #= set experiment_id
    experiment_id = 'res18_1--resnet18--True--fix_tsize4x2'
    # experiment_id = 'sesr_1--sesr--True--fix_tsize32x4'
    # experiment_id = 'srgan_1--srgan--True--fix_tsize32x4'

    # extract path
    other_experiment_id = experiment_id.replace("--fix_tsize4x2", "")   #! 注意在这里改 nxn
    pkl_path_1 = f'outputs/{experiment_id}/all_cmes.pickle'
    pkl_path_2 = f'outputs/{other_experiment_id}/all_cmes.pickle'
    save_path_ema = f'outputs/{experiment_id}/two_lines_tsize_ema.png'
    save_path_edp = f'outputs/{experiment_id}/two_lines_tsize_edp.png'


    # plot free-size and fixed-size
    # plot_two_lines_tsize_ema(pkl_paths=[pkl_path_1, pkl_path_2], save_path=save_path_ema)
    plot_two_lines_tsize_edp(pkl_paths=[pkl_path_1, pkl_path_2], save_path=save_path_edp)
    print(f'##### PLOT DONE @ path: {save_path_ema}')



if __name__ == '__main__':
    plot_2_tsize()
    plot_2_merge()
