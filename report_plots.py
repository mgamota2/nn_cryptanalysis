"""
report_plots.py - Create comprehensive analysis plots for report
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


def load_metrics(results_dir: str, summary_file: str) -> pd.DataFrame:
    """Load metrics from summary CSV file."""
    summary_path = os.path.join(results_dir, summary_file)
    
    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            f"Summary file not found: {summary_path}\n"
            f"Please run decrypt_results.py first to generate the summary."
        )
    
    df = pd.read_csv(summary_path)
    
    print(f"Loaded {len(df)} configurations from {summary_file}")
    print(f"Rounds: {sorted(df['rounds'].unique())}")
    print(f"Nonlinearity levels: {sorted(df['nl'].unique())}")
    print(f"Plaintext types: {df['plaintext_type'].unique().tolist()}")
    print(f"Avalanche effects: {sorted(df['avalanche_effect'].unique())}")
    print(f"Hidden sizes: {sorted(df['hidden_size'].unique())}")
    return df


def plot_1_bit_acc_vs_avalanche(df: pd.DataFrame, save_dir: str):
    """
    PLOT 1: Bit-Level Accuracy vs. Avalanche Effect
    
    Shows how avalanche effect impacts neural network's ability to predict plaintext bits.
    Fixed: NN size = 1024, text plaintext, default P-box
    """
    print("\nGenerating Plot 1: Bit Accuracy vs. Avalanche Effect...")
    
    # Filter data
    data = df[(df['hidden_size'] == 1024) & 
              (df['plaintext_type'] == 'text') & 
              (df['pbox'] == 'Default')]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1a: Scatter plot with trend line
    rounds_to_plot = [2, 4, 8, 16]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(rounds_to_plot)))
    
    for r, color in zip(rounds_to_plot, colors):
        subset = data[data['rounds'] == r]
        if len(subset) > 0:
            ax1.scatter(subset['avalanche_effect'], subset['bit_acc'], 
                       s=80, alpha=0.7, color=color, label=f'{r} rounds', edgecolors='black', linewidth=0.5)
            
            # Add trend line
            if len(subset) > 1:
                z = np.polyfit(subset['avalanche_effect'], subset['bit_acc'], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(subset['avalanche_effect'].min(), subset['avalanche_effect'].max(), 100)
                ax1.plot(x_trend, p(x_trend), color=color, linestyle='--', alpha=0.5, linewidth=2)
    
    ax1.set_xlabel("Avalanche Effect", fontweight='bold')
    ax1.set_ylabel("Bit-Level Accuracy (%)", fontweight='bold')
    ax1.set_title("Bit Accuracy vs Avalanche Effect\n(NN=1024, Text, Default P-box)", fontweight='bold', fontsize=13)
    ax1.legend(loc='best', framealpha=0.95)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=50, color='red', linestyle=':', linewidth=1, alpha=0.5, label='Random Guess')
    
    # Plot 1b: Heatmap of avalanche vs rounds
    pivot_data = data.pivot_table(values='bit_acc', index='rounds', columns='avalanche_effect', aggfunc='mean')
    
    im = ax2.imshow(pivot_data.values, aspect='auto', cmap='RdYlGn', vmin=50, vmax=100)
    ax2.set_xticks(range(len(pivot_data.columns)))
    ax2.set_xticklabels([f'{x:.1f}' for x in pivot_data.columns], rotation=45, ha='right')
    ax2.set_yticks(range(len(pivot_data.index)))
    ax2.set_yticklabels(pivot_data.index)
    ax2.set_xlabel("Avalanche Effect", fontweight='bold')
    ax2.set_ylabel("Number of Rounds", fontweight='bold')
    ax2.set_title("Accuracy Heatmap:\nAvalanche vs Rounds", fontweight='bold', fontsize=13)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Bit Accuracy (%)', rotation=270, labelpad=20)
    
    plt.suptitle("How does Avalanche Effect impact bit prediction accuracy?", 
                 fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "1_bit_acc_vs_avalanche.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: 1_bit_acc_vs_avalanche.png")


def plot_2_avalanche_vs_rounds_nonlinearity(df: pd.DataFrame, save_dir: str):
    """
    PLOT 2: Avalanche Effect vs Rounds and Nonlinearity
    
    Shows the relationship between avalanche effect, rounds, and nonlinearity.
    Fixed: NN size = 1024, text plaintext
    """
    print("\nGenerating Plot 2: Avalanche vs Rounds and Nonlinearity...")
    
    # Filter data
    data = df[(df['hidden_size'] == 1024) & 
              (df['plaintext_type'] == 'text') & 
              (df['pbox'] == 'Default')]
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    
    nl_values = sorted(data['nl'].unique())[:4]  # Take first 4 NL values
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(nl_values)))
    
    for nl, color in zip(nl_values, colors):
        subset = data[data['nl'] == nl].groupby('rounds').agg({
            'avalanche_effect': 'mean',
            'bit_acc': 'mean'
        }).reset_index()
        
        if len(subset) > 0:
            ax.plot(subset['rounds'], subset['avalanche_effect'], 
                    marker='o', linewidth=2.5, markersize=8, 
                    label=f'NL={nl}', color=color, alpha=0.9)
    
    ax.set_xlabel("Number of Rounds", fontweight='bold')
    ax.set_ylabel("Avalanche Effect", fontweight='bold')
    ax.set_title("Avalanche Effect vs Rounds\nfor Different Nonlinearity Levels", fontweight='bold', fontsize=13)
    ax.legend(loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle("Avalanche Effect vs Number of Rounds and Nonlinearity (NN=1024, Text, Default P-box)", 
                 fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "2_avalanche_vs_rounds_nonlinearity.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: 2_avalanche_vs_rounds_nonlinearity.png")


def plot_3_nn_size_impact(df: pd.DataFrame, save_dir: str):
    """
    PLOT 3: Neural Network Size Impact on Accuracy
    
    Shows how different NN sizes perform across rounds.
    Fixed: text plaintext, default P-box, default S-box
    """
    print("\nGenerating Plot 3: NN Size Impact...")
    
    # Filter data
    data = df[(df['plaintext_type'] == 'text') & 
              (df['pbox'] == 'Default') & 
              (df['nl'] == 4)]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 3a: Line plot - accuracy vs rounds for different NN sizes
    hidden_sizes = sorted(data['hidden_size'].unique())
    colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(hidden_sizes)))
    
    for hs, color in zip(hidden_sizes, colors):
        subset = data[data['hidden_size'] == hs].groupby('rounds')['bit_acc'].mean().reset_index()
        if len(subset) > 0:
            ax1.plot(subset['rounds'], subset['bit_acc'], 
                    marker='s', linewidth=2.5, markersize=8, 
                    label=f'NN={hs}', color=color, alpha=0.9)
    
    ax1.set_xlabel("Number of Rounds", fontweight='bold')
    ax1.set_ylabel("Bit-Level Accuracy (%)", fontweight='bold')
    ax1.set_title("NN Size Impact on Accuracy", fontweight='bold', fontsize=13)
    ax1.legend(loc='best', framealpha=0.95)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=50, color='red', linestyle=':', linewidth=1, alpha=0.5)
    
    # Plot 3b: Bar chart comparing NN sizes at specific rounds
    selected_rounds = [2, 8, 16]
    x_pos = np.arange(len(hidden_sizes))
    width = 0.25
    
    for i, r in enumerate(selected_rounds):
        subset = data[data['rounds'] == r]
        accs = [subset[subset['hidden_size'] == hs]['bit_acc'].mean() for hs in hidden_sizes]
        bars = ax2.bar(x_pos + i*width, accs, width, 
                      label=f'{r} rounds', alpha=0.8)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accs):
            if not np.isnan(acc):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{acc:.1f}', ha='center', va='bottom', fontsize=8)
    
    ax2.set_xlabel("Hidden Layer Size", fontweight='bold')
    ax2.set_ylabel("Bit-Level Accuracy (%)", fontweight='bold')
    ax2.set_title("NN Size Comparison at Key Rounds", fontweight='bold', fontsize=13)
    ax2.set_xticks(x_pos + width)
    ax2.set_xticklabels(hidden_sizes)
    ax2.legend(loc='best', framealpha=0.95)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle("How does Neural Network Size impact performance? (Text, Default P-box, Default S-Box)", 
                 fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "3_nn_size_impact.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: 3_nn_size_impact.png")


def plot_4_pbox_comparison(df: pd.DataFrame, save_dir: str):
    """
    PLOT 4: P-box Configuration Impact
    
    Compares different P-box configurations.
    Fixed: NN size = 1024, text plaintext, default S-box
    """
    print("\nGenerating Plot 4: P-box Comparison...")
    
    # Filter data
    data = df[(df['hidden_size'] == 1024) & 
              (df['plaintext_type'] == 'text') & 
              (df['nl'] == 4)]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 4a: Line plot - accuracy vs rounds for different P-boxes
    pbox_types = data['pbox'].unique()
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    
    for pbox, color in zip(pbox_types, colors):
        subset = data[data['pbox'] == pbox].groupby('rounds')['bit_acc'].mean().reset_index()
        if len(subset) > 0:
            ax1.plot(subset['rounds'], subset['bit_acc'], 
                    marker='o', linewidth=2.5, markersize=8, 
                    label=f'{pbox} P-box', color=color, alpha=0.9)
    
    ax1.set_xlabel("Number of Rounds", fontweight='bold')
    ax1.set_ylabel("Bit-Level Accuracy (%)", fontweight='bold')
    ax1.set_title("P-box Configuration Impact", fontweight='bold', fontsize=13)
    ax1.legend(loc='best', framealpha=0.95)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=50, color='red', linestyle=':', linewidth=1, alpha=0.5)
    
    # Plot 4b: Grouped bar chart
    selected_rounds = [2, 8, 16]
    x_pos = np.arange(len(pbox_types))
    width = 0.25
    
    for i, r in enumerate(selected_rounds):
        subset = data[data['rounds'] == r]
        accs = [subset[subset['pbox'] == pb]['bit_acc'].mean() for pb in pbox_types]
        bars = ax2.bar(x_pos + i*width, accs, width, 
                      label=f'{r} rounds', alpha=0.8)
        
        for bar, acc in zip(bars, accs):
            if not np.isnan(acc):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{acc:.1f}', ha='center', va='bottom', fontsize=8)
    
    ax2.set_xlabel("P-box Type", fontweight='bold')
    ax2.set_ylabel("Bit-Level Accuracy (%)", fontweight='bold')
    ax2.set_title("P-box Comparison at Key Rounds", fontweight='bold', fontsize=13)
    ax2.set_xticks(x_pos + width)
    ax2.set_xticklabels(pbox_types)
    ax2.legend(loc='best', framealpha=0.95)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle("P-box Configuration Impact on Accuracy (NN=1024, Text, Default S-box)", 
                 fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "4_pbox_comparison.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: 4_pbox_comparison.png")


def plot_5_text_vs_random(df: pd.DataFrame, save_dir: str):
    """
    PLOT 5: Text vs Random Plaintext Comparison
    
    Compares structured vs random plaintexts.
    Fixed: NN size = 1024, default P-box, default S-box
    """
    print("\nGenerating Plot 5: Text vs Random Plaintext...")
    
    # Filter data
    data = df[(df['hidden_size'] == 1024) & 
              (df['pbox'] == 'Default') & 
              (df['nl'] == 4)]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 5a: Line plot comparing text vs random
    for ptype, color, marker in [('text', '#3498db', 'o'), ('random', '#e67e22', 's')]:
        subset = data[data['plaintext_type'] == ptype].groupby('rounds')['bit_acc'].mean().reset_index()
        if len(subset) > 0:
            ax1.plot(subset['rounds'], subset['bit_acc'], 
                    marker=marker, linewidth=2.5, markersize=8, 
                    label=f'{ptype.capitalize()} Plaintext', color=color, alpha=0.9)
    
    ax1.set_xlabel("Number of Rounds", fontweight='bold')
    ax1.set_ylabel("Bit-Level Accuracy (%)", fontweight='bold')
    ax1.set_title("Text vs Random Plaintext", fontweight='bold', fontsize=13)
    ax1.legend(loc='best', framealpha=0.95)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=50, color='red', linestyle=':', linewidth=1, alpha=0.5)
    
    # Plot 5b: Advantage plot (Text - Random)
    rounds = sorted(data['rounds'].unique())
    advantages = []
    
    for r in rounds:
        text_acc = data[(data['rounds'] == r) & (data['plaintext_type'] == 'text')]['bit_acc'].mean()
        rand_acc = data[(data['rounds'] == r) & (data['plaintext_type'] == 'random')]['bit_acc'].mean()
        if not np.isnan(text_acc) and not np.isnan(rand_acc):
            advantages.append((r, text_acc - rand_acc))
    
    if advantages:
        rounds_adv, adv_vals = zip(*advantages)
        colors_adv = ['#27ae60' if a > 0 else '#c0392b' for a in adv_vals]
        ax2.bar(rounds_adv, adv_vals, color=colors_adv, alpha=0.8, edgecolor='black')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=2, alpha=0.7)
    
    ax2.set_xlabel("Number of Rounds", fontweight='bold')
    ax2.set_ylabel("Advantage (Text - Random) %", fontweight='bold')
    ax2.set_title("Text Advantage over Random", fontweight='bold', fontsize=13)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle("Text-based vs Random Plaintexts (NN=1024, Default P-box, Default S-Box)", 
                 fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "5_text_vs_random.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: 5_text_vs_random.png")


def plot_6_nonlinearity_impact(df: pd.DataFrame, save_dir: str):
    """
    PLOT 6: Nonlinearity Impact on Accuracy
    
    Shows how S-box nonlinearity affects prediction.
    Fixed: NN size = 1024, text plaintext, Default P-box
    """
    print("\nGenerating Plot 6: Nonlinearity Impact...")
    
    # Filter data
    data = df[(df['hidden_size'] == 1024) & 
              (df['plaintext_type'] == 'text') & 
              (df['pbox'] == 'Default')]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 6a: Line plot - accuracy vs rounds for different NL
    nl_values = sorted(data['nl'].unique())[:5]  # Top 5 NL values
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(nl_values)))
    
    for nl, color in zip(nl_values, colors):
        subset = data[data['nl'] == nl].groupby('rounds')['bit_acc'].mean().reset_index()
        if len(subset) > 0:
            ax1.plot(subset['rounds'], subset['bit_acc'], 
                    marker='o', linewidth=2.5, markersize=8, 
                    label=f'NL={nl}', color=color, alpha=0.9)
    
    ax1.set_xlabel("Number of Rounds", fontweight='bold')
    ax1.set_ylabel("Bit-Level Accuracy (%)", fontweight='bold')
    ax1.set_title("Nonlinearity Impact on Accuracy", fontweight='bold', fontsize=13)
    ax1.legend(loc='best', framealpha=0.95)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=50, color='red', linestyle=':', linewidth=1, alpha=0.5)
    
    # Plot 6b: Heatmap
    pivot_data = data.pivot_table(values='bit_acc', index='rounds', columns='nl', aggfunc='mean')
    
    im = ax2.imshow(pivot_data.values, aspect='auto', cmap='RdYlGn', vmin=50, vmax=100)
    ax2.set_xticks(range(len(pivot_data.columns)))
    ax2.set_xticklabels([f'{x}' for x in pivot_data.columns], rotation=45, ha='right')
    ax2.set_yticks(range(len(pivot_data.index)))
    ax2.set_yticklabels(pivot_data.index)
    ax2.set_xlabel("Nonlinearity Level", fontweight='bold')
    ax2.set_ylabel("Number of Rounds", fontweight='bold')
    ax2.set_title("Accuracy Heatmap:\nNonlinearity vs Rounds", fontweight='bold', fontsize=13)
    
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Bit Accuracy (%)', rotation=270, labelpad=20)
    
    plt.suptitle("S-box Nonlinearity Impact on Prediction (NN=1024, Text, Default P-box)", 
                 fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "6_nonlinearity_impact.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: 6_nonlinearity_impact.png")


def plot_7_cosine_vs_rounds_sbox(df: pd.DataFrame, save_dir: str):
    """
    PLOT 7: Cosine Similarity vs Rounds for Different S-box Nonlinearity
    
    Shows how cosine similarity changes with rounds for different nonlinearity levels.
    Fixed: NN size = 1024, text plaintext, default P-box
    """
    print("\nGenerating Plot 7: Cosine Similarity vs Rounds (S-box)...")
    
    # Filter data
    data = df[(df['hidden_size'] == 1024) & 
              (df['plaintext_type'] == 'text') & 
              (df['pbox'] == 'Default')]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 7a: Cosine similarity vs rounds for different NL
    nl_values = sorted(data['nl'].unique())[:5]  # Top 5 NL values
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(nl_values)))
    
    for nl, color in zip(nl_values, colors):
        subset = data[data['nl'] == nl].groupby('rounds')['avg_cos'].mean().reset_index()
        if len(subset) > 0:
            ax1.plot(subset['rounds'], subset['avg_cos'], 
                    marker='o', linewidth=2.5, markersize=8, 
                    label=f'NL={nl}', color=color, alpha=0.9)
    
    ax1.set_xlabel("Number of Rounds", fontweight='bold')
    ax1.set_ylabel("Average Cosine Similarity", fontweight='bold')
    ax1.set_title("Cosine Similarity vs Rounds\nfor Different S-box Nonlinearity", fontweight='bold', fontsize=13)
    ax1.legend(loc='best', framealpha=0.95)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.65, 1.05])
    
    # Plot 7b: Heatmap of cosine similarity
    pivot_data = data.pivot_table(values='avg_cos', index='rounds', columns='nl', aggfunc='mean')
    
    im = ax2.imshow(pivot_data.values, aspect='auto', cmap='RdYlGn', vmin=0.7, vmax=1.0)
    ax2.set_xticks(range(len(pivot_data.columns)))
    ax2.set_xticklabels([f'{x}' for x in pivot_data.columns], rotation=45, ha='right')
    ax2.set_yticks(range(len(pivot_data.index)))
    ax2.set_yticklabels(pivot_data.index)
    ax2.set_xlabel("Nonlinearity Level", fontweight='bold')
    ax2.set_ylabel("Number of Rounds", fontweight='bold')
    ax2.set_title("Cosine Similarity Heatmap:\nNonlinearity vs Rounds", fontweight='bold', fontsize=13)
    
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Avg Cosine Similarity', rotation=270, labelpad=20)
    
    plt.suptitle("Cosine Similarity vs Rounds for Different S-box Nonlinearity (NN=1024, Text, Default P-box)", 
                 fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "7_cosine_vs_rounds_sbox.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: 7_cosine_vs_rounds_sbox.png")


def plot_8_cosine_vs_rounds_pbox(df: pd.DataFrame, save_dir: str):
    """
    PLOT 8: Cosine Similarity vs Rounds for Different P-box Configurations
    
    Shows how cosine similarity changes with rounds for different P-box types.
    Fixed: NN size = 1024, text plaintext, default S-box (nl=4)
    """
    print("\nGenerating Plot 8: Cosine Similarity vs Rounds (P-box)...")
    
    # Filter data
    data = df[(df['hidden_size'] == 1024) & 
              (df['plaintext_type'] == 'text') & 
              (df['nl'] == 4)]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 8a: Cosine similarity vs rounds for different P-boxes
    pbox_types = data['pbox'].unique()
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    
    for pbox, color in zip(pbox_types, colors):
        subset = data[data['pbox'] == pbox].groupby('rounds')['avg_cos'].mean().reset_index()
        if len(subset) > 0:
            ax1.plot(subset['rounds'], subset['avg_cos'], 
                    marker='o', linewidth=2.5, markersize=8, 
                    label=f'{pbox} P-box', color=color, alpha=0.9)
    
    ax1.set_xlabel("Number of Rounds", fontweight='bold')
    ax1.set_ylabel("Average Cosine Similarity", fontweight='bold')
    ax1.set_title("Cosine Similarity vs Rounds\nfor Different P-box Configurations", fontweight='bold', fontsize=13)
    ax1.legend(loc='best', framealpha=0.95)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.65, 1.05])
    
    # Plot 8b: Grouped bar chart for selected rounds
    selected_rounds = [2, 8, 16]
    x_pos = np.arange(len(pbox_types))
    width = 0.25
    
    for i, r in enumerate(selected_rounds):
        subset = data[data['rounds'] == r]
        cos_vals = [subset[subset['pbox'] == pb]['avg_cos'].mean() for pb in pbox_types]
        bars = ax2.bar(x_pos + i*width, cos_vals, width, 
                      label=f'{r} rounds', alpha=0.8)
        
        for bar, cos_val in zip(bars, cos_vals):
            if not np.isnan(cos_val):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{cos_val:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax2.set_xlabel("P-box Type", fontweight='bold')
    ax2.set_ylabel("Average Cosine Similarity", fontweight='bold')
    ax2.set_title("P-box Comparison at Key Rounds", fontweight='bold', fontsize=13)
    ax2.set_xticks(x_pos + width)
    ax2.set_xticklabels(pbox_types)
    ax2.legend(loc='best', framealpha=0.95)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0.65, 1.05])
    
    plt.suptitle("Cosine Similarity vs Rounds for Different P-box Configurations (NN=1024, Text, Default S-box)", 
                 fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "8_cosine_vs_rounds_pbox.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: 8_cosine_vs_rounds_pbox.png")


def plot_9_avalanche_vs_rounds_nonlinearity(df: pd.DataFrame, save_dir: str):
    """
    PLOT 9: Avalanche Effect vs Rounds and Nonlinearity
    
    Shows the relationship between avalanche effect, rounds, and nonlinearity.
    Fixed: NN size = 1024, text plaintext
    """
    print("\nGenerating Plot 9: Avalanche vs Rounds and Nonlinearity...")
    
    # Filter data
    data = df[(df['hidden_size'] == 1024) & 
              (df['plaintext_type'] == 'text') & 
              (df['pbox'] == 'Weak')]
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    
    nl_values = sorted(data['nl'].unique())[:4]  # Take first 4 NL values
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(nl_values)))
    
    for nl, color in zip(nl_values, colors):
        subset = data[data['nl'] == nl].groupby('rounds').agg({
            'avalanche_effect': 'mean',
            'bit_acc': 'mean'
        }).reset_index()
        
        if len(subset) > 0:
            ax.plot(subset['rounds'], subset['avalanche_effect'], 
                    marker='o', linewidth=2.5, markersize=8, 
                    label=f'NL={nl}', color=color, alpha=0.9)
    
    ax.set_xlabel("Number of Rounds", fontweight='bold')
    ax.set_ylabel("Avalanche Effect", fontweight='bold')
    ax.set_title("Avalanche Effect vs Rounds\nfor Different Nonlinearity Levels", fontweight='bold', fontsize=13)
    ax.legend(loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle("Avalanche Effect vs Number of Rounds and Nonlinearity (NN=1024, Text, Weak P-box)", 
                 fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "9_avalanche_vs_rounds_nonlinearity.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: 9_avalanche_vs_rounds_nonlinearity.png")


def plot_10_avalanche_vs_rounds_nonlinearity(df: pd.DataFrame, save_dir: str):
    """
    PLOT 10: Avalanche Effect vs Rounds and Nonlinearity
    
    Shows the relationship between avalanche effect, rounds, and nonlinearity.
    Fixed: NN size = 1024, text plaintext
    """
    print("\nGenerating Plot 10: Avalanche vs Rounds and Nonlinearity...")
    
    # Filter data
    data = df[(df['hidden_size'] == 1024) & 
              (df['plaintext_type'] == 'text') & 
              (df['pbox'] == 'Trivial')]
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    
    nl_values = sorted(data['nl'].unique())[:4]  # Take first 4 NL values
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(nl_values)))
    
    for nl, color in zip(nl_values, colors):
        subset = data[data['nl'] == nl].groupby('rounds').agg({
            'avalanche_effect': 'mean',
            'bit_acc': 'mean'
        }).reset_index()
        
        if len(subset) > 0:
            ax.plot(subset['rounds'], subset['avalanche_effect'], 
                    marker='o', linewidth=2.5, markersize=8, 
                    label=f'NL={nl}', color=color, alpha=0.9)
    
    ax.set_xlabel("Number of Rounds", fontweight='bold')
    ax.set_ylabel("Avalanche Effect", fontweight='bold')
    ax.set_title("Avalanche Effect vs Rounds\nfor Different Nonlinearity Levels", fontweight='bold', fontsize=13)
    ax.legend(loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle("Avalanche Effect vs Number of Rounds and Nonlinearity (NN=1024, Text, Trivial P-box)", 
                 fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "10_avalanche_vs_rounds_nonlinearity.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: 10_avalanche_vs_rounds_nonlinearity.png")


def main():
    """Main execution function."""
    print("="*80)
    print("COMPREHENSIVE VISUALIZATION TOOL - 8 PLOTS")
    print("="*80)
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Plots will be saved to: {PLOTS_DIR}")
    
    # Create plots directory
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # Load metrics
    try:
        df = load_metrics(RESULTS_DIR, SUMMARY_FILE)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return
    
    print("\n" + "="*80)
    print("GENERATING 8 PLOTS")
    print("="*80)
    
    # Generate all 8 plots
    plot_1_bit_acc_vs_avalanche(df, PLOTS_DIR)
    plot_2_avalanche_vs_rounds_nonlinearity(df, PLOTS_DIR)
    plot_3_nn_size_impact(df, PLOTS_DIR)
    plot_4_pbox_comparison(df, PLOTS_DIR)
    plot_5_text_vs_random(df, PLOTS_DIR)
    plot_6_nonlinearity_impact(df, PLOTS_DIR)
    plot_7_cosine_vs_rounds_sbox(df, PLOTS_DIR)
    plot_8_cosine_vs_rounds_pbox(df, PLOTS_DIR)
    plot_9_avalanche_vs_rounds_nonlinearity(df, PLOTS_DIR)
    plot_10_avalanche_vs_rounds_nonlinearity(df, PLOTS_DIR)
    
    print("\n" + "="*80)
    print("ALL PLOTS GENERATED SUCCESSFULLY!")
    print("="*80)


if __name__ == "__main__":
    main()