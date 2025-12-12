"""
visualize_results.py - Create comprehensive analysis plots
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from typing import List, Dict
from pprint import pprint


# Configuration
RESULTS_DIR = "./new_results"
PLOTS_DIR = "./new_plots"
SUMMARY_FILE = "decryption_summary.csv"

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 100


def load_metrics(results_dir: str, summary_file: str) -> List[Dict]:
    """Load metrics from summary CSV file."""
    summary_path = os.path.join(results_dir, summary_file)
    
    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            f"Summary file not found: {summary_path}\n"
            f"Please run decrypt_results.py first to generate the summary."
        )
    
    df = pd.read_csv(summary_path)
    metrics = df.to_dict('records')
    
    print(f"Loaded {len(metrics)} configurations from {summary_file}")
    print(f"Rounds: {sorted(df['rounds'].unique())}")
    print(f"Nonlinearity levels: {sorted(df['nl'].unique())}")
    print(f"Plaintext types: {df['plaintext_type'].unique().tolist()}")
    return metrics


def plot_1_learnability_vs_nonlinearity(metrics: List[Dict], save_dir: str):
    """
    PLOT 1: Learnability vs. S-box Nonlinearity
    
    Shows how S-box nonlinearity (confusion property) affects neural network's 
    ability to predict plaintext bits.
    """
    print("\nGenerating Plot 1: Learnability vs. Nonlinearity...")
    
    # Group by rounds and nonlinearity (use text-based for primary analysis)
    data_by_rounds = defaultdict(lambda: defaultdict(list))
    
    for m in metrics:
        if m["plaintext_type"] == "text" and m["hidden_size"]==1024:
            data_by_rounds[m["rounds"]][m["nl"]].append(m["bit_acc"])
    
    rounds_to_plot = sorted(data_by_rounds.keys())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1a: Line plot - accuracy degradation with rounds
    nl_values = [0, 2, 4]
    nl_labels = ["Linear (NL=0)", "Medium (NL=2)", "Default (NL=4)"]
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    
    for nl, label, color in zip(nl_values, nl_labels, colors):
        rounds_list = []
        acc_list = []
        for r in rounds_to_plot:
            if nl in data_by_rounds[r]:
                rounds_list.append(r)
                acc_list.append(np.mean(data_by_rounds[r][nl]))
        
        if rounds_list:
            ax1.plot(rounds_list, acc_list, marker='o', linewidth=2.5, 
                    markersize=8, label=label, color=color, alpha=0.9)
    
    ax1.set_xlabel("Number of Rounds", fontweight='bold')
    ax1.set_ylabel("Bit-Level Accuracy (%)", fontweight='bold')
    ax1.set_title("Neural Network Performance vs S-box Nonlinearity (Text Plaintext, NN hidden size=1024)", fontweight='bold', fontsize=13)
    ax1.legend(loc='best', framealpha=0.95)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=45)
    ax1.axhline(y=50, color='red', linestyle=':', linewidth=1, alpha=0.5, label='Random Guess')
    
    # Plot 1b: Bar comparison for selected rounds
    selected_rounds = [r for r in [2, 8, 16] if r in rounds_to_plot]
    if not selected_rounds:
        selected_rounds = rounds_to_plot[:3]
    
    x_pos = np.arange(len(nl_values))
    width = 0.25
    
    for i, r in enumerate(selected_rounds):
        if r in data_by_rounds:
            accs = [np.mean(data_by_rounds[r].get(nl, [50])) for nl in nl_values]
            bars = ax2.bar(x_pos + i*width, accs, width, 
                          label=f'{r} rounds', alpha=0.8)
            
            # Add value labels on bars
            for bar, acc in zip(bars, accs):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{acc:.1f}', ha='center', va='bottom', fontsize=8)
    
    ax2.set_xlabel("S-box Nonlinearity Level", fontweight='bold')
    ax2.set_ylabel("Bit-Level Accuracy (%)", fontweight='bold')
    ax2.set_title("Impact of Nonlinearity on Learnability (Text Plaintext, NN hidden size=1024)", fontweight='bold', fontsize=13)
    ax2.set_xticks(x_pos + width)
    ax2.set_xticklabels(nl_labels)
    ax2.legend(loc='best', framealpha=0.95)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(bottom=45)
    
    plt.suptitle("How does S-box linearity affect neural network prediction?", 
                 fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "1_learnability_vs_nonlinearity.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: 1_learnability_vs_nonlinearity.png")


def plot_2_learnability_vs_rounds(metrics: List[Dict], save_dir: str):
    """
    PLOT 2: Learnability vs. Number of Cipher Rounds
    
    Shows how increasing cipher rounds (diffusion) impacts prediction accuracy.
    Compares across different nonlinearity levels and plaintext types.
    """
    print("\nGenerating Plot 2: Learnability vs. Rounds...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    nl_values = [0, 2, 4]
    nl_labels = ["Linear (NL=0)", "Medium (NL=2)", "Default (NL=4)"]
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    
    for ax, ptype, title in zip(axes, ["text", "random"], 
                                 ["Text-based Plaintexts", "Random Plaintexts"]):
        
        for nl, label, color in zip(nl_values, nl_labels, colors):
            data = defaultdict(list)
            for m in metrics:
                if m["plaintext_type"] == ptype and m["nl"] == nl:
                    data[m["rounds"]].append(m["bit_acc"])
            
            if not data:
                continue
                
            rounds_list = sorted(data.keys())
            acc_means = [np.mean(data[r]) for r in rounds_list]
            acc_stds = [np.std(data[r]) if len(data[r]) > 1 else 0 for r in rounds_list]
            
            ax.errorbar(rounds_list, acc_means, yerr=acc_stds, 
                       marker='o', linewidth=2.5, markersize=8, 
                       label=label, color=color, capsize=5, alpha=0.9)
        
        ax.set_xlabel("Number of Cipher Rounds", fontweight='bold')
        ax.set_ylabel("Bit-Level Accuracy (%)", fontweight='bold')
        ax.set_title(title, fontweight='bold', fontsize=13)
        ax.legend(loc='best', framealpha=0.95)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=45)
        ax.axhline(y=50, color='red', linestyle=':', linewidth=1, alpha=0.5)
    
    plt.suptitle("Research Question 2: How does cipher round number impact bit prediction accuracy?", 
                 fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "2_learnability_vs_rounds.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: 2_learnability_vs_rounds.png")


def plot_3_network_size_impact(metrics: List[Dict], save_dir: str):
    """
    PLOT 3: Neural Network Architecture Impact
    
    Shows how hidden layer size affects the network's ability to learn
    cipher structure. Tests if larger networks can better model complexity.
    """
    print("\nGenerating Plot 3: Network Size Impact...")
    
    # Filter for metrics with hidden_size information
    data_by_hs = defaultdict(lambda: defaultdict(list))
    
    for m in metrics:
        if m["hidden_size"] is not None and m["plaintext_type"] == "text":
            data_by_hs[m["hidden_size"]][m["rounds"]].append(m["bit_acc"])
    
    if not data_by_hs:
        print("  Warning: No hidden size data found, skipping plot 3")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 3a: Line plot - accuracy vs hidden size for different rounds
    hidden_sizes = sorted(data_by_hs.keys())
    rounds_available = set()
    for hs_data in data_by_hs.values():
        rounds_available.update(hs_data.keys())
    
    # rounds_to_plot = sorted(list(rounds_available))[:7]
    rounds_to_plot = [1,2,3,4,5,6,16,32]
    colors_rounds = plt.cm.viridis(np.linspace(0.1, 0.9, len(rounds_to_plot)))
    
    for r, color in zip(rounds_to_plot, colors_rounds):
        hs_list = []
        acc_list = []
        for hs in hidden_sizes:
            if r in data_by_hs[hs]:
                hs_list.append(hs)
                acc_list.append(np.mean(data_by_hs[hs][r]))
        
        if hs_list:
            ax1.plot(hs_list, acc_list, marker='s', linewidth=2.5,
                    markersize=8, label=f'{r} rounds', color=color)
    
    ax1.set_xlabel("Hidden Layer Size", fontweight='bold')
    ax1.set_ylabel("Bit-Level Accuracy (%)", fontweight='bold')
    ax1.set_title("Hidden Layer Size vs Performance (Text Data)", fontweight='bold', fontsize=13)
    ax1.legend(loc='best', framealpha=0.95)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)
    ax1.set_ylim(bottom=45)
    
    # Plot 3b: Heatmap showing all combinations
    rounds_sorted = sorted(rounds_available)
    heatmap_data = np.full((len(rounds_sorted), len(hidden_sizes)), np.nan)
    
    for i, r in enumerate(rounds_sorted):
        for j, hs in enumerate(hidden_sizes):
            if r in data_by_hs[hs]:
                heatmap_data[i, j] = np.mean(data_by_hs[hs][r])
    
    im = ax2.imshow(heatmap_data, aspect='auto', cmap='RdYlGn', vmin=50, vmax=100)
    ax2.set_xticks(range(len(hidden_sizes)))
    ax2.set_xticklabels([str(hs) for hs in hidden_sizes], rotation=45, ha='right')
    ax2.set_yticks(range(len(rounds_sorted)))
    ax2.set_yticklabels(rounds_sorted)
    ax2.set_xlabel("Hidden Layer Size", fontweight='bold')
    ax2.set_ylabel("Number of Rounds", fontweight='bold')
    ax2.set_title("Hidden Layer Size vs Rounds \n Accuracy Heatmap (Text Data)", fontweight='bold', fontsize=13)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Bit Accuracy (%)', rotation=270, labelpad=20)
    
    plt.suptitle("How does neural network size impact performance?", 
                 fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "3_network_size_impact.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: 3_network_size_impact.png")


def plot_4_text_vs_random(metrics: List[Dict], save_dir: str):
    """
    PLOT 4: Text-based vs Random Bit Plaintexts
    
    Compares whether structured (English text) plaintexts are easier to predict
    than random bitstrings. Tests if plaintext structure aids learning.
    """
    print("\nGenerating Plot 4: Text vs Random Comparison...")
    
    # Separate data by plaintext type
    data_text = defaultdict(lambda: defaultdict(list))
    data_random = defaultdict(lambda: defaultdict(list))
    
    for m in metrics:
        target = data_text if m["plaintext_type"] == "text" else data_random
        target[m["rounds"]][m["nl"]].append(m["bit_acc"])
        
    rounds_common = sorted(set(data_text.keys()) & set(data_random.keys()))
    
    if not rounds_common:
        print("  Warning: No common rounds between text and random, skipping plot 4")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), 
                            gridspec_kw={'height_ratios': [2, 1]})
    
    nl_values = [0, 2, 4]
    nl_labels = ["Linear (NL=0)", "Medium (NL=2)", "Default (NL=4)"]
    
    # Plot each nonlinearity separately
    for idx, (nl, label) in enumerate(zip(nl_values, nl_labels)):
        # Collect data for text and random
        rounds_text = []
        mean_text = []
        std_text = []
        
        rounds_random = []
        mean_random = []
        std_random = []
        
        for r in rounds_common:
            if nl in data_text[r] and len(data_text[r][nl]) > 0:
                rounds_text.append(r)
                mean_text.append(np.mean(data_text[r][nl]))
                std_text.append(np.std(data_text[r][nl]))
            
            if nl in data_random[r] and len(data_random[r][nl]) > 0:
                rounds_random.append(r)
                mean_random.append(np.mean(data_random[r][nl]))
                std_random.append(np.std(data_random[r][nl]))
        
        # TOP ROW: Accuracy plots
        # Plot text data
        if rounds_text:
            axes[0, idx].errorbar(rounds_text, mean_text, yerr=std_text,
                              marker='o', linewidth=2.5, markersize=8,
                              capsize=5, capthick=2, label='Text Plaintext',
                              color='#3498db', alpha=0.8)
        
        # Plot random data
        if rounds_random:
            axes[0, idx].errorbar(rounds_random, mean_random, yerr=std_random,
                              marker='s', linewidth=2.5, markersize=8,
                              capsize=5, capthick=2, label='Random Plaintext',
                              color='#e67e22', alpha=0.8)
        
        axes[0, idx].set_xlabel("Number of Rounds", fontweight='bold', fontsize=11)
        axes[0, idx].set_ylabel("Bit-Level Accuracy (%)", fontweight='bold', fontsize=11)
        axes[0, idx].set_title(label, fontweight='bold', fontsize=12)
        axes[0, idx].legend(loc='best', framealpha=0.95, fontsize=10)
        axes[0, idx].grid(True, alpha=0.3)
        axes[0, idx].set_ylim(bottom=45)
        
        # Add 50% baseline (random guessing)
        axes[0, idx].axhline(y=50, color='red', linestyle='--', linewidth=1.5, 
                         alpha=0.5, label='Random Baseline')
        
        # BOTTOM ROW: Advantage plots (Text - Random)
        # Calculate advantage where we have both text and random data
        rounds_advantage = []
        advantage = []
        advantage_err = []
        
        for r in rounds_common:
            if nl in data_text[r] and nl in data_random[r]:
                if len(data_text[r][nl]) > 0 and len(data_random[r][nl]) > 0:
                    # Calculate pairwise differences if possible
                    text_vals = data_text[r][nl]
                    random_vals = data_random[r][nl]
                    
                    # Mean advantage
                    mean_advantage = np.mean(text_vals) - np.mean(random_vals)
                    
                    # Error propagation: std of difference
                    std_advantage = np.sqrt(np.std(text_vals)**2 + np.std(random_vals)**2)
                    
                    rounds_advantage.append(r)
                    advantage.append(mean_advantage)
                    advantage_err.append(std_advantage)
        
        if rounds_advantage:
            # Color based on advantage: green if positive, red if negative
            colors_adv = ['#27ae60' if a > 0 else '#c0392b' for a in advantage]
            
            axes[1, idx].errorbar(rounds_advantage, advantage, yerr=advantage_err,
                                 marker='D', linewidth=2.5, markersize=8,
                                 capsize=5, capthick=2, 
                                 color='#9b59b6', alpha=0.8,
                                 label='Text Advantage')
                    
        axes[1, idx].axhline(y=0, color='black', linestyle='-', linewidth=2, alpha=0.7)
        axes[1, idx].set_xlabel("Number of Rounds", fontweight='bold', fontsize=11)
        axes[1, idx].set_ylabel("Advantage (Text - Random) %", fontweight='bold', fontsize=11)
        axes[1, idx].set_title(f"Text Advantage for {label}", fontweight='bold', fontsize=11)
        axes[1, idx].legend(loc='best', framealpha=0.95, fontsize=9)
        axes[1, idx].grid(True, alpha=0.3)
    
    plt.suptitle("Research Question 4: Text-based vs Random Plaintexts\n(Averaged across hidden sizes with error bars)", 
                 fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "4_text_vs_random.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: 4_text_vs_random.png")



def main():
    """Main execution function."""
    print("="*80)
    print("COMPREHENSIVE VISUALIZATION TOOL")
    print("="*80)
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Plots will be saved to: {PLOTS_DIR}")
    
    # Create plots directory
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # Load metrics
    try:
        metrics = load_metrics(RESULTS_DIR, SUMMARY_FILE)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return
    pprint(metrics)
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)
    
    plot_1_learnability_vs_nonlinearity(metrics, PLOTS_DIR)
    plot_2_learnability_vs_rounds(metrics, PLOTS_DIR)
    plot_3_network_size_impact(metrics, PLOTS_DIR)
    plot_4_text_vs_random(metrics, PLOTS_DIR)

if __name__ == "__main__":
    main()