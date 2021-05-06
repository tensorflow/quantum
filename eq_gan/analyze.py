import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm, rcParams
import os
import sys

suffix = sys.argv[1]


def find_best_params(overlap_record, g_weights, d_loss, smoothing_period=1):
    """Get overlap and parameters at minimum generator loss."""
    # simple moving average
    flattened_loss = np.array(d_loss).flatten()
    if smoothing_period > 1:
        smoothed = np.convolve(flattened_loss, np.ones(smoothing_period), 'valid') / smoothing_period
    else:
        smoothed = flattened_loss
    # find when the discriminator loss is lowest in the second half of training
    # this corresponds to when it is fooled the best
    n_episodes = len(g_weights)
    best_ind = n_episodes//2 + np.argmin(smoothed)
    best_ind += smoothing_period // 2
    if best_ind >= n_episodes:
        best_ind = n_episodes - 1
    return best_ind, g_weights[best_ind], overlap_record[best_ind]

def find_best_overlaps(data):
    overlap_record = data['overlap_record']
    g_weights = data['g_weights']
    g_loss = data['g_loss']
    
    overlap = []
    for i in range(len(overlap_record)):
        _, _, o = find_best_params(overlap_record[i], g_weights[i], g_loss[i])
        overlap.append(o)
    return overlap

if __name__ == '__main__':
    plt.rcParams['font.family'] = 'Times New Roman'
    
    out_adv = np.load('out-all-adversarial_swap-' + suffix + '.npz', allow_pickle=True)
    out_perf = np.load('out-all-perfect_swap-' + suffix + '.npz', allow_pickle=True)
    
    # plot one example
    gan_data = out_adv['overlap_record'][0]
    swap_data = out_perf['overlap_record'][0]
    
    adv_g_loss = out_adv['g_loss'][0]
    adv_d_loss = out_adv['d_loss'][0]
    adv_g_weights = out_adv['g_weights'][0]
    adv_best_ind, _, _ = find_best_params(gan_data, adv_g_weights, adv_d_loss)
    
    perf_g_loss = out_perf['g_loss'][0]
    perf_d_loss = out_perf['d_loss'][0]
    perf_g_weights = out_perf['g_weights'][0]
#     perf_best_ind, _, _ = find_best_params(swap_data, perf_g_weights, perf_d_loss)
    
    # plot 
    plt.figure(figsize=(5, 3.9))
    plt.plot(swap_data, 'C1', label='Supervised learner')
    plt.plot(gan_data, 'C2', label='EQ-GAN')
#     plt.plot([0, len(swap_data)], [1, 1], c='black', linestyle='--')
    plt.legend(fontsize=12)
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('State fidelity w.r.t. true data', fontsize=14)
#     plt.xlim(50, len(swap_data))
#     plt.ylim(0.995, 1.0)
#     plt.axvline(x=perf_best_ind, c='C1', linestyle='--')
    print(adv_best_ind)
#     plt.axvline(x=adv_best_ind, c='C2', linestyle='--')

    plt.tight_layout()
    plt.savefig('./plots/err-' + suffix + '.pdf')
    plt.show()
    
    
    # plot generator loss
    plt.figure(figsize=(5, 3.9))
    plt.plot(perf_g_loss, 'C1', label='Supervised loss')
    plt.plot(adv_g_loss, 'C2', label='EQ-GAN loss')
    plt.xlabel('Iteration', fontsize=14)
    plt.xlim(len(swap_data)//2, len(swap_data))
    plt.ylabel('Generator loss', fontsize=14)
#     plt.axvline(x=perf_best_ind, c='C1', linestyle='--')
    plt.axvline(x=adv_best_ind, c='C2', linestyle='--')
    plt.legend(fontsize=12)
    plt.savefig('./plots/loss_g-' + suffix + '.pdf')
    plt.show()
    
    
    # plot discriminator loss
    plt.figure(figsize=(5, 3.9))
    plt.plot(np.concatenate((np.zeros(len(swap_data)//2), adv_d_loss)), 'C2', label='EQ-GAN loss')
    plt.xlabel('Iteration', fontsize=14)
    plt.xlim(len(swap_data)//2, len(swap_data))
    plt.ylabel('Discriminator loss', fontsize=14)
    plt.axvline(x=adv_best_ind, c='C2', linestyle='--')
    plt.legend(fontsize=12)
    plt.savefig('./plots/loss_d-' + suffix + '.pdf')
    plt.show()
    
    
    # plot histogram of all runs
    perf_overlaps = find_best_overlaps(out_perf)
    adv_overlaps = find_best_overlaps(out_adv)
    
    print(perf_overlaps)
    print(adv_overlaps)
    min_bin = min(np.amin(perf_overlaps), np.amin(adv_overlaps))
    bins = np.linspace(min_bin, 1, num=20)
    
    plt.figure(figsize=(5, 3.9))
    plt.hist(perf_overlaps, color='C1', alpha=0.5, label='Supervised learner', bins=bins)
    plt.hist(adv_overlaps, color='C2', alpha=0.5, label='EQ-GAN', bins=bins)
    plt.legend(fontsize=12)
    plt.xlabel('State fidelity w.r.t. true data', fontsize=14)
    plt.tight_layout()
    plt.savefig('./plots/overlaps-' + suffix + '.pdf')
    plt.show()