"""
Comprehensive Comparison Analysis: Pure LLM vs Neural Network
For Formula Discovery Across All Domains
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import pandas as pd

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_results(llm_file: str, nn_file: str) -> tuple:
    """Load both result files."""
    try:
        with open(llm_file, 'r') as f:
            llm_results = json.load(f)
        print(f"✅ Loaded LLM results from: {llm_file}")
        
        with open(nn_file, 'r') as f:
            nn_results = json.load(f)
        print(f"✅ Loaded NN results from: {nn_file}")
        
        return llm_results, nn_results
    except FileNotFoundError as e:
        print(f"❌ Error: File not found - {e}")
        raise
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON format - {e}")
        raise

def create_comparison_tables(llm_results: List, nn_results: List):
    """Create comprehensive comparison tables using pandas."""
    
    # Handle different JSON structures (dict vs list)
    if isinstance(llm_results, dict):
        # Extract test results from report structure
        llm_list = []
        nn_list = []
        
        # Try to find the results array in the dict
        for key in ['results', 'test_results', 'cases']:
            if key in llm_results:
                llm_list = llm_results[key]
                break
        
        for key in ['results', 'test_results', 'cases']:
            if key in nn_results:
                nn_list = nn_results[key]
                break
        
        # If still empty, the dict itself might be a single result
        if not llm_list:
            llm_list = [llm_results]
        if not nn_list:
            nn_list = [nn_results]
            
        llm_results = llm_list
        nn_results = nn_list
    
    # Build main comparison dataframe
    data = []
    for i, (llm_r, nn_r) in enumerate(zip(llm_results, nn_results)):
        # Handle both dict and object-like structures
        if isinstance(llm_r, dict):
            desc = llm_r.get('description', f'Test case {i+1}')[:50]
            domain = llm_r.get('domain', llm_r.get('metadata', {}).get('domain', 'unknown'))
            formula_type = llm_r.get('metadata', {}).get('formula_type', 
                                                          llm_r.get('formula_type', 'unknown'))
            
            llm_eval = llm_r.get('evaluation', {})
            nn_eval = nn_r.get('evaluation', {})
            
            llm_r2 = llm_eval.get('r2')
            nn_r2 = nn_eval.get('r2')
            llm_rmse = llm_eval.get('rmse')
            nn_rmse = nn_eval.get('rmse')
            
            extrap = llm_r.get('metadata', {}).get('extrapolation_test', False)
            if not extrap:
                extrap = llm_r.get('extrapolation_test', False)
        else:
            # Handle string or other types
            desc = f'Test case {i+1}'
            domain = 'unknown'
            formula_type = 'unknown'
            llm_r2 = None
            nn_r2 = None
            llm_rmse = None
            nn_rmse = None
            extrap = False
        
        data.append({
            'Description': desc,
            'Domain': domain,
            'Formula Type': formula_type,
            'LLM R²': llm_r2,
            'NN R²': nn_r2,
            'LLM RMSE': llm_rmse,
            'NN RMSE': nn_rmse,
            'Extrapolation': extrap
        })
    
    df = pd.DataFrame(data)
    
    # Calculate winner (only for valid R² values)
    def determine_winner(row):
        llm_r2 = row['LLM R²']
        nn_r2 = row['NN R²']
        
        if llm_r2 is None or nn_r2 is None:
            return 'N/A'
        if pd.isna(llm_r2) or pd.isna(nn_r2):
            return 'N/A'
        
        diff = abs(llm_r2 - nn_r2)
        if diff < 0.001:  # Essentially tied
            return 'Tie'
        elif llm_r2 > nn_r2:
            return 'LLM'
        else:
            return 'NN'
    
    df['Winner'] = df.apply(determine_winner, axis=1)
    
    # Calculate R² difference (handle None/NaN values)
    def calc_diff(row):
        llm = row['LLM R²']
        nn = row['NN R²']
        if llm is None or nn is None or pd.isna(llm) or pd.isna(nn):
            return np.nan
        return llm - nn
    
    df['R² Difference'] = df.apply(calc_diff, axis=1)
    
    return df


def plot_overall_comparison(df: pd.DataFrame, output_dir: Path):
    """Create overall comparison visualizations."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Pure LLM vs Neural Network: Overall Comparison', fontsize=16, fontweight='bold')
    
    # 1. R² Distribution comparison
    ax1 = axes[0, 0]
    llm_r2 = df['LLM R²'].dropna()
    nn_r2 = df['NN R²'].dropna()
    
    ax1.hist(llm_r2, bins=20, alpha=0.6, label='LLM', color='blue', edgecolor='black')
    ax1.hist(nn_r2, bins=20, alpha=0.6, label='NN', color='red', edgecolor='black')
    ax1.axvline(llm_r2.mean(), color='blue', linestyle='--', linewidth=2, label=f'LLM Mean: {llm_r2.mean():.3f}')
    ax1.axvline(nn_r2.mean(), color='red', linestyle='--', linewidth=2, label=f'NN Mean: {nn_r2.mean():.3f}')
    ax1.set_xlabel('R² Score', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('R² Distribution', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Head-to-head scatter plot
    ax2 = axes[0, 1]
    ax2.scatter(df['LLM R²'], df['NN R²'], alpha=0.6, s=100, edgecolors='black', linewidth=1)
    
    # Add diagonal line
    min_val = min(df['LLM R²'].min(), df['NN R²'].min())
    max_val = max(df['LLM R²'].max(), df['NN R²'].max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=2, label='Equal Performance')
    
    ax2.set_xlabel('LLM R²', fontsize=12)
    ax2.set_ylabel('NN R²', fontsize=12)
    ax2.set_title('Head-to-Head R² Comparison', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Win rate pie chart
    ax3 = axes[1, 0]
    winner_counts = df['Winner'].value_counts()
    colors = {'LLM': 'blue', 'NN': 'red', 'Tie': 'gray', 'N/A': 'lightgray'}
    pie_colors = [colors[w] for w in winner_counts.index]
    
    wedges, texts, autotexts = ax3.pie(winner_counts.values, 
                                         labels=winner_counts.index,
                                         autopct='%1.1f%%',
                                         colors=pie_colors,
                                         startangle=90,
                                         textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax3.set_title('Win Rate Distribution', fontsize=14, fontweight='bold')
    
    # 4. Performance by difficulty
    ax4 = axes[1, 1]
    
    # Success rate (R² > 0.95) by method
    llm_excellent = (df['LLM R²'] > 0.95).sum()
    llm_good = ((df['LLM R²'] > 0.80) & (df['LLM R²'] <= 0.95)).sum()
    llm_poor = (df['LLM R²'] <= 0.80).sum()
    
    nn_excellent = (df['NN R²'] > 0.95).sum()
    nn_good = ((df['NN R²'] > 0.80) & (df['NN R²'] <= 0.95)).sum()
    nn_poor = (df['NN R²'] <= 0.80).sum()
    
    categories = ['Excellent\n(R² > 0.95)', 'Good\n(0.80 < R² ≤ 0.95)', 'Poor\n(R² ≤ 0.80)']
    llm_values = [llm_excellent, llm_good, llm_poor]
    nn_values = [nn_excellent, nn_good, nn_poor]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, llm_values, width, label='LLM', color='blue', alpha=0.7, edgecolor='black')
    bars2 = ax4.bar(x + width/2, nn_values, width, label='NN', color='red', alpha=0.7, edgecolor='black')
    
    ax4.set_ylabel('Number of Cases', fontsize=12)
    ax4.set_title('Performance Quality Distribution', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(categories)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'overall_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'overall_comparison.png'}")
    plt.close()


def plot_domain_comparison(df: pd.DataFrame, output_dir: Path):
    """Create domain-by-domain comparison."""
    
    # Group by domain
    domain_stats = df.groupby('Domain').agg({
        'LLM R²': ['mean', 'std', 'count'],
        'NN R²': ['mean', 'std']
    }).round(4)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Performance by Domain', fontsize=16, fontweight='bold')
    
    # 1. Mean R² by domain
    ax1 = axes[0]
    domains = domain_stats.index
    x = np.arange(len(domains))
    width = 0.35
    
    llm_means = domain_stats['LLM R²']['mean'].values
    nn_means = domain_stats['NN R²']['mean'].values
    llm_stds = domain_stats['LLM R²']['std'].values
    nn_stds = domain_stats['NN R²']['std'].values
    
    bars1 = ax1.bar(x - width/2, llm_means, width, yerr=llm_stds, 
                     label='LLM', color='blue', alpha=0.7, 
                     capsize=5, edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x + width/2, nn_means, width, yerr=nn_stds,
                     label='NN', color='red', alpha=0.7,
                     capsize=5, edgecolor='black', linewidth=1.5)
    
    ax1.set_ylabel('Mean R² Score', fontsize=12)
    ax1.set_title('Mean R² by Domain (with standard deviation)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(domains, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0.95, color='green', linestyle='--', alpha=0.5, linewidth=2, label='Excellent threshold')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 2. Heatmap of R² differences
    ax2 = axes[1]
    
    # Create matrix for heatmap
    pivot_data = []
    for domain in df['Domain'].unique():
        domain_df = df[df['Domain'] == domain]
        row_data = domain_df['R² Difference'].values
        pivot_data.append(row_data)
    
    # Create heatmap data
    max_len = max(len(row) for row in pivot_data)
    heatmap_data = np.full((len(pivot_data), max_len), np.nan)
    for i, row in enumerate(pivot_data):
        heatmap_data[i, :len(row)] = row
    
    im = ax2.imshow(heatmap_data, cmap='RdBu_r', aspect='auto', vmin=-0.5, vmax=0.5)
    ax2.set_yticks(np.arange(len(domains)))
    ax2.set_yticklabels(domains)
    ax2.set_xlabel('Test Case Index', fontsize=12)
    ax2.set_title('R² Difference (LLM - NN) Heatmap\nBlue = LLM Better, Red = NN Better', 
                  fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('R² Difference', rotation=270, labelpad=20, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'domain_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'domain_comparison.png'}")
    plt.close()
    
    return domain_stats


def plot_formula_type_comparison(df: pd.DataFrame, output_dir: Path):
    """Create formula type comparison."""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Performance by Formula Type', fontsize=16, fontweight='bold')
    
    # Group by formula type
    formula_stats = df.groupby('Formula Type').agg({
        'LLM R²': ['mean', 'count'],
        'NN R²': 'mean',
        'R² Difference': 'mean'
    }).round(4)
    
    # 1. Mean R² by formula type
    ax1 = axes[0]
    formula_types = formula_stats.index
    x = np.arange(len(formula_types))
    width = 0.35
    
    llm_means = formula_stats['LLM R²']['mean'].values
    nn_means = formula_stats['NN R²']['mean'].values
    
    bars1 = ax1.bar(x - width/2, llm_means, width, label='LLM', 
                     color='blue', alpha=0.7, edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x + width/2, nn_means, width, label='NN',
                     color='red', alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax1.set_ylabel('Mean R² Score', fontsize=12)
    ax1.set_title('Mean R² by Formula Type', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(formula_types, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0.95, color='green', linestyle='--', alpha=0.5, linewidth=2)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 2. Advantage by formula type
    ax2 = axes[1]
    differences = formula_stats['R² Difference']['mean'].values
    colors = ['blue' if d > 0 else 'red' for d in differences]
    
    bars = ax2.barh(formula_types, differences, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=2)
    ax2.set_xlabel('Mean R² Difference (LLM - NN)', fontsize=12)
    ax2.set_title('LLM Advantage by Formula Type\nPositive = LLM Better', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width + 0.01 if width > 0 else width - 0.01
        ax2.text(label_x_pos, bar.get_y() + bar.get_height()/2,
                f'{width:.3f}',
                ha='left' if width > 0 else 'right',
                va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'formula_type_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'formula_type_comparison.png'}")
    plt.close()
    
    return formula_stats


def plot_extrapolation_analysis(df: pd.DataFrame, output_dir: Path):
    """Analyze extrapolation performance."""
    
    extrap_df = df[df['Extrapolation'] == True].copy()
    
    if len(extrap_df) == 0:
        print("⚠️ No extrapolation test cases found")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Extrapolation Performance Analysis', fontsize=16, fontweight='bold')
    
    # 1. Extrapolation vs non-extrapolation
    ax1 = axes[0]
    
    extrap_llm = extrap_df['LLM R²'].mean()
    extrap_nn = extrap_df['NN R²'].mean()
    
    non_extrap_df = df[df['Extrapolation'] == False]
    non_extrap_llm = non_extrap_df['LLM R²'].mean()
    non_extrap_nn = non_extrap_df['NN R²'].mean()
    
    categories = ['Extrapolation\nTests', 'Standard\nTests']
    llm_values = [extrap_llm, non_extrap_llm]
    nn_values = [extrap_nn, non_extrap_nn]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, llm_values, width, label='LLM',
                     color='blue', alpha=0.7, edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x + width/2, nn_values, width, label='NN',
                     color='red', alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax1.set_ylabel('Mean R² Score', fontsize=12)
    ax1.set_title('Extrapolation vs Standard Test Performance', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0.95, color='green', linestyle='--', alpha=0.5, linewidth=2)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 2. Individual extrapolation cases
    ax2 = axes[1]
    
    case_names = [desc[:25] for desc in extrap_df['Description']]
    y_pos = np.arange(len(case_names))
    
    ax2.barh(y_pos - 0.2, extrap_df['LLM R²'], 0.4, label='LLM',
             color='blue', alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.barh(y_pos + 0.2, extrap_df['NN R²'], 0.4, label='NN',
             color='red', alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(case_names, fontsize=9)
    ax2.set_xlabel('R² Score', fontsize=12)
    ax2.set_title('Individual Extrapolation Test Cases', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.axvline(x=0.95, color='green', linestyle='--', alpha=0.5, linewidth=2)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'extrapolation_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'extrapolation_analysis.png'}")
    plt.close()


def create_summary_tables(df: pd.DataFrame, domain_stats: pd.DataFrame, 
                          formula_stats: pd.DataFrame, output_dir: Path):
    """Create and save summary tables as text files."""
    
    with open(output_dir / 'summary_tables.txt', 'w') as f:
        f.write("="*100 + "\n")
        f.write("COMPREHENSIVE COMPARISON SUMMARY TABLES\n")
        f.write("="*100 + "\n\n")
        
        # Overall statistics
        f.write("1. OVERALL STATISTICS\n")
        f.write("-"*100 + "\n")
        f.write(f"{'Metric':<40} {'LLM':>25} {'NN':>25}\n")
        f.write("-"*100 + "\n")
        
        llm_r2 = df['LLM R²'].dropna()
        nn_r2 = df['NN R²'].dropna()
        
        f.write(f"{'Total Cases':<40} {len(llm_r2):>25} {len(nn_r2):>25}\n")
        f.write(f"{'Mean R²':<40} {llm_r2.mean():>25.4f} {nn_r2.mean():>25.4f}\n")
        f.write(f"{'Median R²':<40} {llm_r2.median():>25.4f} {nn_r2.median():>25.4f}\n")
        f.write(f"{'Std Dev R²':<40} {llm_r2.std():>25.4f} {nn_r2.std():>25.4f}\n")
        f.write(f"{'Min R²':<40} {llm_r2.min():>25.4f} {nn_r2.min():>25.4f}\n")
        f.write(f"{'Max R²':<40} {llm_r2.max():>25.4f} {nn_r2.max():>25.4f}\n")
        f.write(f"{'Excellent (R² > 0.99)':<40} {(llm_r2 > 0.99).sum():>25} {(nn_r2 > 0.99).sum():>25}\n")
        f.write(f"{'Good (R² > 0.95)':<40} {(llm_r2 > 0.95).sum():>25} {(nn_r2 > 0.95).sum():>25}\n")
        f.write(f"{'Moderate (0.80 < R² ≤ 0.95)':<40} {((llm_r2 > 0.80) & (llm_r2 <= 0.95)).sum():>25} {((nn_r2 > 0.80) & (nn_r2 <= 0.95)).sum():>25}\n")
        f.write(f"{'Poor (R² ≤ 0.80)':<40} {(llm_r2 <= 0.80).sum():>25} {(nn_r2 <= 0.80).sum():>25}\n")
        
        # Win rates
        f.write("\n2. HEAD-TO-HEAD RESULTS\n")
        f.write("-"*100 + "\n")
        winner_counts = df['Winner'].value_counts()
        total = len(df)
        f.write(f"{'LLM Wins:':<40} {winner_counts.get('LLM', 0)} ({winner_counts.get('LLM', 0)/total*100:.1f}%)\n")
        f.write(f"{'NN Wins:':<40} {winner_counts.get('NN', 0)} ({winner_counts.get('NN', 0)/total*100:.1f}%)\n")
        f.write(f"{'Ties:':<40} {winner_counts.get('Tie', 0)} ({winner_counts.get('Tie', 0)/total*100:.1f}%)\n")
        
        # Domain statistics
        f.write("\n3. PERFORMANCE BY DOMAIN\n")
        f.write("-"*100 + "\n")
        f.write(f"{'Domain':<20} {'Count':>10} {'LLM Mean R²':>15} {'NN Mean R²':>15} {'LLM Advantage':>15}\n")
        f.write("-"*100 + "\n")
        
        for domain in domain_stats.index:
            count = int(domain_stats.loc[domain, ('LLM R²', 'count')])
            llm_mean = domain_stats.loc[domain, ('LLM R²', 'mean')]
            nn_mean = domain_stats.loc[domain, ('NN R²', 'mean')]
            advantage = llm_mean - nn_mean
            
            f.write(f"{domain:<20} {count:>10} {llm_mean:>15.4f} {nn_mean:>15.4f} {advantage:>15.4f}\n")
        
        # Formula type statistics
        f.write("\n4. PERFORMANCE BY FORMULA TYPE\n")
        f.write("-"*100 + "\n")
        f.write(f"{'Formula Type':<25} {'Count':>10} {'LLM Mean R²':>15} {'NN Mean R²':>15} {'LLM Advantage':>15}\n")
        f.write("-"*100 + "\n")
        
        for ftype in formula_stats.index:
            count = int(formula_stats.loc[ftype, ('LLM R²', 'count')])
            llm_mean = formula_stats.loc[ftype, ('LLM R²', 'mean')]
            nn_mean = formula_stats.loc[ftype, ('NN R²', 'mean')]
            advantage = formula_stats.loc[ftype, ('R² Difference', 'mean')]
            
            f.write(f"{ftype:<25} {count:>10} {llm_mean:>15.4f} {nn_mean:>15.4f} {advantage:>15.4f}\n")
        
        # Detailed case-by-case
        f.write("\n5. DETAILED CASE-BY-CASE COMPARISON\n")
        f.write("-"*100 + "\n")
        f.write(f"{'Description':<50} {'LLM R²':>12} {'NN R²':>12} {'Difference':>12} {'Winner':>10}\n")
        f.write("-"*100 + "\n")
        
        for idx, row in df.iterrows():
            desc = row['Description'][:47] + "..."
            llm_r2 = row['LLM R²']
            nn_r2 = row['NN R²']
            diff = row['R² Difference']
            winner = row['Winner']

            llm_r2_str = f"{llm_r2:.4f}" if llm_r2 is not None and not pd.isna(llm_r2) else "N/A"
            nn_r2_str = f"{nn_r2:.4f}" if nn_r2 is not None and not pd.isna(nn_r2) else "N/A"
            diff_str = f"{diff:.4f}" if diff is not None and not pd.isna(diff) else "N/A"
            f.write(f"{desc:<50} {llm_r2_str:>12} {nn_r2_str:>12} {diff_str:>12} {winner:>10}\n")
            
    print(f"✅ Saved: {output_dir / 'summary_tables.txt'}")


def generate_comparison_report(llm_file: str, nn_file: str):
    """Generate comprehensive comparison report with tables and figures."""
    
    # Load results
    llm_results, nn_results = load_results(llm_file, nn_file)
    
    # Create output directory
    output_dir = Path("results/comparison_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*100)
    print("GENERATING COMPREHENSIVE COMPARISON ANALYSIS".center(100))
    print("="*100)
    
    # Create comparison dataframe
    print("\n📊 Creating comparison tables...")
    df = create_comparison_tables(llm_results, nn_results)
    
    # Save raw dataframe
    df.to_csv(output_dir / 'detailed_comparison.csv', index=False)
    print(f"✅ Saved: {output_dir / 'detailed_comparison.csv'}")
    
    # Generate visualizations
    print("\n📈 Generating visualizations...")
    plot_overall_comparison(df, output_dir)
    domain_stats = plot_domain_comparison(df, output_dir)
    formula_stats = plot_formula_type_comparison(df, output_dir)
    plot_extrapolation_analysis(df, output_dir)
    
    # Create summary tables
    print("\n📝 Creating summary tables...")
    create_summary_tables(df, domain_stats, formula_stats, output_dir)
    
    # Print console summary
    print("\n" + "="*100)
    print("SUMMARY STATISTICS".center(100))
    print("="*100)
    
    llm_r2 = df['LLM R²'].dropna()
    nn_r2 = df['NN R²'].dropna()
    
    print(f"\nOverall Performance:")
    print(f"  LLM Mean R²: {llm_r2.mean():.4f} (±{llm_r2.std():.4f})")
    print(f"  NN Mean R²:  {nn_r2.mean():.4f} (±{nn_r2.std():.4f})")
    print(f"  LLM Advantage: {llm_r2.mean() - nn_r2.mean():.4f}")
    
    print(f"\nWin Rates:")
    winner_counts = df['Winner'].value_counts()
    total = len(df)
    print(f"  LLM: {winner_counts.get('LLM', 0)}/{total} ({winner_counts.get('LLM', 0)/total*100:.1f}%)")
    print(f"  NN:  {winner_counts.get('NN', 0)}/{total} ({winner_counts.get('NN', 0)/total*100:.1f}%)")
    print(f"  Tie: {winner_counts.get('Tie', 0)}/{total} ({winner_counts.get('Tie', 0)/total*100:.1f}%)")
    
    print(f"\nPerformance by Quality:")
    print(f"  Excellent (R² > 0.99):")
    print(f"    LLM: {(llm_r2 > 0.99).sum()}/{len(llm_r2)} ({(llm_r2 > 0.99).sum()/len(llm_r2)*100:.1f}%)")
    print(f"    NN:  {(nn_r2 > 0.99).sum()}/{len(nn_r2)} ({(nn_r2 > 0.99).sum()/len(nn_r2)*100:.1f}%)")
    
    print(f"\n  Good (R² > 0.95):")
    print(f"    LLM: {(llm_r2 > 0.95).sum()}/{len(llm_r2)} ({(llm_r2 > 0.95).sum()/len(llm_r2)*100:.1f}%)")
    print(f"    NN:  {(nn_r2 > 0.95).sum()}/{len(nn_r2)} ({(nn_r2 > 0.95).sum()/len(nn_r2)*100:.1f}%)")
    
    # Domain breakdown
    print("\n" + "="*100)
    print("DOMAIN-SPECIFIC ANALYSIS".center(100))
    print("="*100)
    
    for domain in domain_stats.index:
        count = int(domain_stats.loc[domain, ('LLM R²', 'count')])
        llm_mean = domain_stats.loc[domain, ('LLM R²', 'mean')]
        nn_mean = domain_stats.loc[domain, ('NN R²', 'mean')]
        advantage = llm_mean - nn_mean
        
        print(f"\n{domain.upper()}:")
        print(f"  Cases: {count}")
        print(f"  LLM Mean R²: {llm_mean:.4f}")
        print(f"  NN Mean R²:  {nn_mean:.4f}")
        print(f"  Advantage:   {advantage:+.4f} {'(LLM wins)' if advantage > 0 else '(NN wins)' if advantage < 0 else '(Tie)'}")
    
    # Critical insights
    print("\n" + "="*100)
    print("CRITICAL INSIGHTS".center(100))
    print("="*100)
    
    # Best LLM performances
    best_llm = df[df['R² Difference'] > 0.2].sort_values('R² Difference', ascending=False)
    if len(best_llm) > 0:
        print("\n🎯 Cases where LLM dominantly outperforms (R² diff > 0.2):")
        for idx, row in best_llm.iterrows():
            print(f"  • {row['Description'][:60]}")
            print(f"    LLM: {row['LLM R²']:.4f} vs NN: {row['NN R²']:.4f} (Δ = +{row['R² Difference']:.4f})")
    
    # Best NN performances
    best_nn = df[df['R² Difference'] < -0.1].sort_values('R² Difference')
    if len(best_nn) > 0:
        print("\n⚠️  Cases where NN has advantage (R² diff < -0.1):")
        for idx, row in best_nn.iterrows():
            print(f"  • {row['Description'][:60]}")
            print(f"    NN: {row['NN R²']:.4f} vs LLM: {row['LLM R²']:.4f} (Δ = {row['R² Difference']:.4f})")
    
    # Failure analysis
    print("\n" + "="*100)
    print("FAILURE ANALYSIS".center(100))
    print("="*100)
    
    llm_failures = df[df['LLM R²'] < 0.80]
    nn_failures = df[df['NN R²'] < 0.80]
    
    print(f"\nLLM Failures (R² < 0.80): {len(llm_failures)}/{len(df)}")
    if len(llm_failures) > 0:
        for idx, row in llm_failures.iterrows():
            print(f"  • {row['Description'][:60]}")
            print(f"    R² = {row['LLM R²']:.4f}, Domain: {row['Domain']}")
    
    print(f"\nNN Failures (R² < 0.80): {len(nn_failures)}/{len(df)}")
    if len(nn_failures) > 0:
        for idx, row in nn_failures.iterrows():
            print(f"  • {row['Description'][:60]}")
            print(f"    R² = {row['NN R²']:.4f}, Domain: {row['Domain']}")
    
    # Catastrophic failures
    llm_catastrophic = df[df['LLM R²'] < 0]
    nn_catastrophic = df[df['NN R²'] < 0]
    
    if len(llm_catastrophic) > 0 or len(nn_catastrophic) > 0:
        print("\n🚨 CATASTROPHIC FAILURES (R² < 0):")
        if len(llm_catastrophic) > 0:
            print(f"\n  LLM: {len(llm_catastrophic)} cases")
            for idx, row in llm_catastrophic.iterrows():
                print(f"    • {row['Description'][:60]}: R² = {row['LLM R²']:.4f}")
        
        if len(nn_catastrophic) > 0:
            print(f"\n  NN: {len(nn_catastrophic)} cases")
            for idx, row in nn_catastrophic.iterrows():
                print(f"    • {row['Description'][:60]}: R² = {row['NN R²']:.4f}")
    
    # Recommendations
    print("\n" + "="*100)
    print("RECOMMENDATIONS".center(100))
    print("="*100)
    
    print("\n💡 Based on the analysis:")
    
    # Calculate recommendation based on actual results
    if llm_r2.mean() > nn_r2.mean() + 0.1:
        print("\n✅ PREFER LLM APPROACH for:")
        print("  • Overall superior performance across domains")
        print(f"  • {winner_counts.get('LLM', 0)/total*100:.1f}% win rate vs {winner_counts.get('NN', 0)/total*100:.1f}%")
        print("  • Interpretable formulas with mathematical insights")
        print("  • Zero-shot learning without training data")
    elif nn_r2.mean() > llm_r2.mean() + 0.1:
        print("\n✅ PREFER NN APPROACH for:")
        print("  • Overall superior performance across domains")
        print(f"  • {winner_counts.get('NN', 0)/total*100:.1f}% win rate vs {winner_counts.get('LLM', 0)/total*100:.1f}%")
        print("  • Complex non-linear patterns")
        print("  • When interpretability is not required")
    else:
        print("\n⚖️  METHODS ARE COMPARABLE:")
        print("  • Choose based on specific requirements:")
        print("    - LLM: interpretability, zero-shot, mathematical insights")
        print("    - NN: handles complex patterns, large datasets")
    
    # Domain-specific recommendations
    print("\n📊 Domain-specific recommendations:")
    for domain in domain_stats.index:
        llm_mean = domain_stats.loc[domain, ('LLM R²', 'mean')]
        nn_mean = domain_stats.loc[domain, ('NN R²', 'mean')]
        
        if llm_mean > nn_mean + 0.1:
            rec = "LLM ✅"
        elif nn_mean > llm_mean + 0.1:
            rec = "NN ✅"
        else:
            rec = "Either ⚖️"
        
        print(f"  • {domain:20s}: {rec:15s} (LLM: {llm_mean:.3f}, NN: {nn_mean:.3f})")
    
    # Save summary JSON
    summary = {
        "overall": {
            "llm_mean_r2": float(llm_r2.mean()),
            "nn_mean_r2": float(nn_r2.mean()),
            "llm_std_r2": float(llm_r2.std()),
            "nn_std_r2": float(nn_r2.std()),
            "llm_wins": int(winner_counts.get('LLM', 0)),
            "nn_wins": int(winner_counts.get('NN', 0)),
            "ties": int(winner_counts.get('Tie', 0)),
            "total_cases": int(total)
        },
        "by_domain": {},
        "by_formula_type": {}
    }
    
    for domain in domain_stats.index:
        summary["by_domain"][domain] = {
            "count": int(domain_stats.loc[domain, ('LLM R²', 'count')]),
            "llm_mean": float(domain_stats.loc[domain, ('LLM R²', 'mean')]),
            "nn_mean": float(domain_stats.loc[domain, ('NN R²', 'mean')]),
            "advantage": float(domain_stats.loc[domain, ('LLM R²', 'mean')] - 
                             domain_stats.loc[domain, ('NN R²', 'mean')])
        }
    
    for ftype in formula_stats.index:
        summary["by_formula_type"][ftype] = {
            "count": int(formula_stats.loc[ftype, ('LLM R²', 'count')]),
            "llm_mean": float(formula_stats.loc[ftype, ('LLM R²', 'mean')]),
            "nn_mean": float(formula_stats.loc[ftype, ('NN R²', 'mean')]),
            "advantage": float(formula_stats.loc[ftype, ('R² Difference', 'mean')])
        }
    
    with open(output_dir / 'comparison_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*100)
    print(f"✅ All results saved to: {output_dir}")
    print("="*100)
    print("\nGenerated files:")
    print(f"  • detailed_comparison.csv")
    print(f"  • summary_tables.txt")
    print(f"  • comparison_summary.json")
    print(f"  • overall_comparison.png")
    print(f"  • domain_comparison.png")
    print(f"  • formula_type_comparison.png")
    print(f"  • extrapolation_analysis.png")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python comparison_analysis_improved.py <llm_results.json> <nn_results.json>")
        print("\nSearching for latest results files...")
        
        # Try to find latest files
        results_dir = Path("results")
        if not results_dir.exists():
            results_dir = Path(".")
        
        # More flexible pattern matching
        llm_patterns = ["*llm*.json", "*LLM*.json", "baseline_pure_llm*.json", 
                       "report*llm*.json", "experiment_report*.json"]
        nn_patterns = ["*nn*.json", "*NN*.json", "baseline_nn*.json", 
                      "report*nn*.json", "nn_experiment*.json"]
        
        llm_files = []
        for pattern in llm_patterns:
            llm_files.extend(list(results_dir.glob(pattern)))
        
        nn_files = []
        for pattern in nn_patterns:
            nn_files.extend(list(results_dir.glob(pattern)))
        
        # Remove duplicates
        llm_files = list(set(llm_files))
        nn_files = list(set(nn_files))
        
        if llm_files and nn_files:
            llm_file = max(llm_files, key=lambda p: p.stat().st_mtime)
            nn_file = max(nn_files, key=lambda p: p.stat().st_mtime)
            print(f"Found LLM results: {llm_file}")
            print(f"Found NN results: {nn_file}")
            
            # Ask for confirmation
            response = input("\nUse these files? (y/n): ").lower()
            if response == 'y' or response == '':
                generate_comparison_report(str(llm_file), str(nn_file))
            else:
                print("Please specify files manually.")
        else:
            print("❌ Could not find results files automatically.")
            print("Please provide file paths as arguments.")
            print(f"\nLooking in: {results_dir.absolute()}")
            if llm_files:
                print(f"Found LLM files: {[f.name for f in llm_files]}")
            if nn_files:
                print(f"Found NN files: {[f.name for f in nn_files]}")
    else:
        llm_file = sys.argv[1]
        nn_file = sys.argv[2]
        generate_comparison_report(llm_file, nn_file)
