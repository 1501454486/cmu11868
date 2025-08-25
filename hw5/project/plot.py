import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import numpy as np
import math
import os

def plot(means, stds, labels, fig_name, y_label):
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(means)), means, yerr=stds,
           align='center', alpha=0.5, ecolor='red', capsize=10, width=0.6)
    ax.set_ylabel(y_label)
    ax.set_xticks(np.arange(len(means)))
    ax.set_xticklabels(labels)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close(fig)

# Fill the data points here
if __name__ == '__main__':
    """
    epoch1:
    device1_time: 18.3255
    device2_time: 18.8361
    device1_throughput: 99171.7
    device2_throughput: 99391.2
    
    epoch2:
    1_time: 18.6210
    2_time: 18.4131
    1_throughput: 98798.5
    2_throughput: 98984.4
    
    epoch3:
    1_time: 18.6930
    2_time: 18.4571
    1_throughput: 98526.4
    2_throughput: 98767.6
    
    epoch4:
    18.7780
    18.4900
    98216.2
    98691.3
    
    
    Single:
    epoch1:
    time: 35.0487
    throughput: 224168.4
    
    epoch2:
    35.3361
    222788.9
    
    epoch3:
    35.3747
    221264.8
    
    epoch4:
    35.3812
    221662.3
    """
    output_dir = 'submit_figures'
    os.makedirs(output_dir, exist_ok=True)
    
    single_mean, single_std = 35.2852, 0.1589
    device0_mean, device0_std =  18.6044, 0.1967
    device1_mean, device1_std =  18.5491, 0.1939
    plot([device0_mean, device1_mean, single_mean],
        [device0_std, device1_std, single_std],
        ['Data Parallel - GPU0', 'Data Parallel - GPU1', 'Single GPU'],
        f'{output_dir}/training_time_comparison.png',
        y_label = 'GPT2 Execution Time (Second)')
    
    single_tp_mean, single_tp_std = 222471.1, 1302.7076
    device0_tp_mean, device0_tp_std = 98678.2, 405.9974
    device1_tp_mean, device1_tp_std = 98958.625, 313.9738
    total_dp_tp_mean = device0_tp_mean + device1_tp_mean
    total_dp_tp_std = math.sqrt(device0_tp_std ** 2 + device1_tp_std ** 2)
    plot(
        means = [total_dp_tp_mean, single_tp_mean],
        stds = [total_dp_tp_std, single_tp_std],
        labels = ['Data Parallel (2GPUs)', 'Single GPU'],
        fig_name = f'{output_dir}/throughput_comparison.png',
        y_label = 'GPT2 Throughput (Tokens per Second)'
    )
    

    mp_time_mean, mp_time_std = 39.8867, 0.2852
    mp_tp_mean, mp_tp_std = 16046.3, 113.5
    pp_time_mean, pp_time_std = 33.9733, 0.5037
    pp_tp_mean, pp_tp_std = 18842.4, 274.06

    plot(
        means = [pp_time_mean, mp_time_mean],
        stds = [pp_time_std, mp_time_std],
        labels = ['Pipeline Parallel', 'Model Parallel'],
        fig_name = f'{output_dir}/pp_vs_mp_time.png',
        y_label = 'GPT2 Execution Time (second)'
    )
    
    plot(
        means = [pp_tp_mean, mp_tp_mean],
        stds = [pp_tp_std, mp_tp_std],
        labels = ['Pipeline Parallel', 'Model Parallel'],
        fig_name = f'{output_dir}/pp_vs_mp_throughput.png',
        y_label = 'GPT2 Throughput (tokens per second)'
    )