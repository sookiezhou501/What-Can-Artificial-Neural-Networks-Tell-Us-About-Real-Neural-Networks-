# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 24,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 20,
    'figure.titlesize': 26
})


depths = [3, 5, 8, 10]
bp_times = [0.03, 0.04, 0.07, 0.10]
pcn_std_times = [0.81, 1.84, 3.29, 4.27]
pcn_inc_times = [0.07, 0.16, 0.29, 0.32]

bp_acc = [0.1500, 0.1700, 0.1700, 0.1700]
pcn_std_acc = [0.0600, 0.0900, 0.0600, 0.0900]
pcn_inc_acc = [0.0900, 0.1100, 0.0900, 0.0900]

steps = [1, 2, 3, 5, 10, 20, 30]
step_times = [0.03, 0.03, 0.06, 0.08, 0.14, 0.24, 0.36]
step_acc = [0.0500, 0.0333, 0.0500, 0.0500, 0.0333, 0.0333, 0.0500]

colors = {
    'BP': '#1f77b4',
    'PCN_Standard': '#ff7f0e',
    'PCN_Incremental': '#2ca02c'
}


def create_figure_training_accuracy(save_path='figure_training_accuracy.png'):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
    
    ax1.plot(depths, bp_times, 'o-', label='BP', 
             linewidth=3, markersize=12, color=colors['BP'])
    ax1.plot(depths, pcn_std_times, 's-', label='Standard PCN (T=50)', 
             linewidth=3, markersize=12, color=colors['PCN_Standard'])
    ax1.plot(depths, pcn_inc_times, '^-', label='Incremental PCN (T=3)', 
             linewidth=3, markersize=12, color=colors['PCN_Incremental'])
    
    ax1.set_xlabel('Network Depth (Layers)', fontsize=26, fontweight='bold')
    ax1.set_ylabel('Training Time (seconds)', fontsize=26, fontweight='bold')
    ax1.set_title('(a) Training Time vs Network Depth', fontsize=24, fontweight='bold')

    ax1.legend(loc='center left', bbox_to_anchor=(0.7,0.85 ), 
               fontsize=18, framealpha=0.9, facecolor='lightgray', edgecolor='black')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.tick_params(axis='both', labelsize=18)
    ax1.set_yscale('log')
    

    for i, depth in enumerate(depths):
        speedup = pcn_std_times[i] / pcn_inc_times[i]
        ax1.annotate(f'{speedup:.1f}x', 
                    xy=(depth, pcn_inc_times[i]),
                    xytext=(depth + 0.3, pcn_inc_times[i] * 1.5),
                    fontsize=16, color='#2ca02c', fontweight='bold')
    
 
    ax2.plot(depths, bp_acc, 'o-', label='BP', 
             linewidth=3, markersize=12, color=colors['BP'])
    ax2.plot(depths, pcn_std_acc, 's-', label='Standard PCN (T=50)', 
             linewidth=3, markersize=12, color=colors['PCN_Standard'])
    ax2.plot(depths, pcn_inc_acc, '^-', label='Incremental PCN (T=3)', 
             linewidth=3, markersize=12, color=colors['PCN_Incremental'])
    
    ax2.set_xlabel('Network Depth (Layers)', fontsize=22, fontweight='bold')
    ax2.set_ylabel('Test Accuracy', fontsize=22, fontweight='bold')
    ax2.set_title('(b) Accuracy vs Network Depth', fontsize=24, fontweight='bold')

    ax2.legend(loc='center left', bbox_to_anchor=(0.7, 0.85), 
               fontsize=18, framealpha=0.9, facecolor='lightgray', edgecolor='black')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.tick_params(axis='both', labelsize=18)
    ax2.set_ylim([0, 0.25])
    
    plt.suptitle('Incremental PCN: Training Time and Accuracy Comparison', 
                 fontsize=28, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {save_path}")
    plt.show()
    
    return fig


def create_figure_efficiency_analysis(save_path='figure_efficiency_analysis.png'):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
    

    bp_ratio = [pcn_inc_times[i] / bp_times[i] for i in range(len(depths))]
    speedup_over_std = [pcn_std_times[i] / pcn_inc_times[i] for i in range(len(depths))]
    time_saved = [(pcn_std_times[i] - pcn_inc_times[i]) / pcn_std_times[i] * 100 
                  for i in range(len(depths))]
    
    x_pos = np.arange(len(depths))
    width = 0.35
    

    bars1 = ax1.bar(x_pos - width/2, bp_ratio, width, label='vs BP (lower=better)', 
                    color='#1f77b4', alpha=0.8, edgecolor='black', linewidth=2)
    bars2 = ax1.bar(x_pos + width/2, speedup_over_std, width, label='vs Standard PCN (higher=better)', 
                    color='#2ca02c', alpha=0.8, edgecolor='black', linewidth=2)
    
    ax1.set_xlabel('Network Depth (Layers)', fontsize=22, fontweight='bold')
    ax1.set_ylabel('Efficiency Ratio', fontsize=22, fontweight='bold')
    ax1.set_title('(a) Efficiency Gain Analysis', fontsize=24, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(depths, fontsize=18)
    ax1.set_ylim([0, 15])
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    

    ax1.legend(loc='center left', bbox_to_anchor=(0.7, 0.85), 
               fontsize=18, framealpha=0.9, facecolor='lightgray', edgecolor='black')
    

    for bar, val in zip(bars1, bp_ratio):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                f'{val:.1f}x', ha='center', va='bottom', fontsize=16, fontweight='bold')
    for bar, val in zip(bars2, speedup_over_std):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}x', ha='center', va='bottom', fontsize=16, fontweight='bold')
    

    bars = ax2.bar(depths, time_saved, color='#ff7f0e', alpha=0.8, 
                   edgecolor='black', linewidth=2, width=0.6, label='Time Saved')
    ax2.set_xlabel('Network Depth (Layers)', fontsize=22, fontweight='bold')
    ax2.set_ylabel('Time Saved (%)', fontsize=22, fontweight='bold')
    ax2.set_title('(b) Computational Cost Reduction', fontsize=24, fontweight='bold')
    ax2.tick_params(axis='both', labelsize=18)
    ax2.set_ylim([0, 100])
    ax2.grid(True, alpha=0.3, axis='y')
    

    ax2.legend(loc='center left', bbox_to_anchor=(0.7, 0.85), 
               fontsize=18, framealpha=0.9, facecolor='lightgray', edgecolor='black')
    
    # 添加数值标签
    for bar, val in zip(bars, time_saved):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=18, fontweight='bold')
    
    plt.suptitle('Incremental PCN: Efficiency Gains and Cost Reduction', 
                 fontsize=28, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {save_path}")
    plt.show()
    
    return fig


def create_figure_inference_steps_analysis(save_path='figure_inference_steps_analysis.png'):

    fig, ax = plt.subplots(figsize=(16, 10))
    
    color_time = 'tab:blue'
    color_acc = 'tab:orange'
    
    line1 = ax.plot(steps, step_times, 'o-', color=color_time, 
                     linewidth=3, markersize=12, label='Training Time')
    ax.set_xlabel('Inference Steps T', fontsize=26, fontweight='bold')
    ax.set_ylabel('Training Time (seconds)', color=color_time, fontsize=26, fontweight='bold')
    ax.tick_params(axis='y', labelcolor=color_time, labelsize=24)
    ax.tick_params(axis='x', labelsize=24)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, linestyle='--')
    

    efficiency_scores = [step_acc[i] / step_times[i] if step_times[i] > 0 else 0 
                         for i in range(len(steps))]
    efficiency_scores = np.array(efficiency_scores)
    efficiency_scores = efficiency_scores / efficiency_scores.max()
    
    ax_twin = ax.twinx()
    line2 = ax_twin.plot(steps, step_acc, 's-', color=color_acc, 
                          linewidth=3, markersize=12, label='Accuracy')
    line3 = ax_twin.plot(steps, efficiency_scores, 'd-', color='#9467bd',
                          linewidth=3, markersize=10, label='Efficiency Score', alpha=0.7)
    ax_twin.set_ylabel('Test Accuracy / Efficiency Score', color=color_acc, fontsize=22, fontweight='bold')
    ax_twin.tick_params(axis='y', labelcolor=color_acc, labelsize=18)
    

    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='center left', bbox_to_anchor=(0.6, 0.85), 
              fontsize=22, framealpha=0.9, facecolor='lightgray', edgecolor='black')
    

    ax.axvspan(2.5, 5.5, alpha=0.2, color='green')
    ax.text(3.8, max(step_times[:5]) * 0.6, 'Optimal Region\n(T=3-5)', 
            ha='center', va='center', fontsize=22, fontweight='bold', color='darkgreen')
    
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {save_path}")
    plt.show()
    
    return fig


def main():

    print("=" * 60)
    print("Generating Figures for Simulation 9")
    print("=" * 60)
    
    print("\n1. Creating training time and accuracy figure...")
    fig1 = create_figure_training_accuracy('figure_training_accuracy.png')
    
    print("\n2. Creating efficiency analysis figure...")
    fig2 = create_figure_efficiency_analysis('figure_efficiency_analysis.png')
    
    print("\n3. Creating inference steps analysis figure...")
    fig3 = create_figure_inference_steps_analysis('figure_inference_steps_analysis.png')
    
    print("\n" + "=" * 60)
    print("All figures generated successfully:")
    print("  • figure_training_accuracy.png")
    print("  • figure_efficiency_analysis.png")
    print("  • figure_inference_steps_analysis.png")
    print("=" * 60)


if __name__ == "__main__":
    main()