import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import numpy as np

def plot(means, stds, labels, fig_name):
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(means)), means, yerr=stds,
           align='center', alpha=0.5, ecolor='red', capsize=10, width=0.6)
    ax.set_ylabel('GPT2 Execution Time (Second)')
    ax.set_xticks(np.arange(len(means)))
    ax.set_xticklabels(labels)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close(fig)

# Fill the data points here
if __name__ == '__main__':
    single_mean, single_std = None, None
    device0_mean, device0_std =  None, None
    device1_mean, device1_std =  None, None
    plot([mp0_mean, mp1_mean, rn_mean],
        [mp0_std, mp1_std, rn_std],
        ['Data Parallel - GPU0', 'Data Parallel - GPU1', 'Single GPU'],
        'ddp_vs_rn.png')

    pp_mean, pp_std = None, None
    mp_mean, mp_std = None, None
    plot([pp_mean, mp_mean],
        [pp_std, mp_std],
        ['Pipeline Parallel', 'Model Parallel'],
        'pp_vs_mp.png')