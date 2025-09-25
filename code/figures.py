import seaborn as sns
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec

def plot_heatmap_matrix(self, figsize=(14, 10)):
    """Create heatmap showing performance of all fingerprint-model combinations"""
    df = self.get_summary_statistics()
    if df.empty:
        return
    
    # Separate classification and regression
    class_df = df[df['is_classification'] == True]
    reg_df = df[df['is_classification'] == False]
    
    fig = plt.figure(figsize=figsize)
    
    if not class_df.empty and not reg_df.empty:
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        axes = [ax1, ax2]
        data_types = [('Classification (AUROC)', class_df), ('Regression (RMSE)', reg_df)]
    elif not class_df.empty:
        ax1 = fig.add_subplot(111)
        axes = [ax1]
        data_types = [('Classification (AUROC)', class_df)]
    else:
        ax1 = fig.add_subplot(111)
        axes = [ax1]
        data_types = [('Regression (RMSE)', reg_df)]
    
    for ax, (title, data) in zip(axes, data_types):
        # Create pivot table for heatmap
        pivot_data = data.pivot_table(values='mean', index='fingerprint', columns='model', aggfunc='mean')
        
        # Create heatmap
        if 'Classification' in title:
            cmap = 'Reds'
            fmt = '.3f'
        else:
            # For RMSE, reverse colormap (lower is better)
            cmap = 'Reds_r'
            fmt = '.3f'
        
        sns.heatmap(pivot_data, annot=True, fmt=fmt, cmap=cmap, ax=ax,
                   cbar_kws={'label': 'Performance'}, square=True)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Model')
        ax.set_ylabel('Fingerprint')
    
    plt.suptitle('Performance Heatmap: Fingerprint-Model Combinations', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plot_path = os.path.join(self.save_dir, 'performance_heatmap.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_ranking_chart(self, figsize=(16, 10)):
    """Create ranking chart showing top combinations across all datasets"""
    detailed_df = self.get_detailed_scores()
    if detailed_df.empty:
        return
    
    # Calculate overall performance for each combination
    combo_stats = detailed_df.groupby(['fingerprint', 'model', 'is_classification'])['score'].agg([
        'mean', 'std', 'count'
    ]).reset_index()
    
    combo_stats['combination'] = combo_stats['fingerprint'] + ' + ' + combo_stats['model']
    
    # Separate classification and regression
    class_combos = combo_stats[combo_stats['is_classification'] == True].copy()
    reg_combos = combo_stats[combo_stats['is_classification'] == False].copy()
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Classification ranking (higher AUROC is better)
    if not class_combos.empty:
        class_combos = class_combos.sort_values('mean', ascending=False).head(10)
        
        bars = axes[0].barh(range(len(class_combos)), class_combos['mean'], 
                           xerr=class_combos['std'], capsize=5)
        axes[0].set_yticks(range(len(class_combos)))
        axes[0].set_yticklabels(class_combos['combination'])
        axes[0].set_xlabel('AUROC')
        axes[0].set_title('Top 10 Combinations - Classification', fontweight='bold')
        axes[0].grid(axis='x', alpha=0.3)
        
        # Color bars by fingerprint
        fingerprints = class_combos['fingerprint'].unique()
        colors = plt.cm.Set2(np.linspace(0, 1, len(fingerprints)))
        fp_colors = {fp: colors[i] for i, fp in enumerate(fingerprints)}
        
        for i, (bar, fp) in enumerate(zip(bars, class_combos['fingerprint'])):
            bar.set_color(fp_colors[fp])
    
    # Regression ranking (lower RMSE is better)
    if not reg_combos.empty:
        reg_combos = reg_combos.sort_values('mean', ascending=True).head(10)
        
        bars = axes[1].barh(range(len(reg_combos)), reg_combos['mean'], 
                           xerr=reg_combos['std'], capsize=5)
        axes[1].set_yticks(range(len(reg_combos)))
        axes[1].set_yticklabels(reg_combos['combination'])
        axes[1].set_xlabel('RMSE')
        axes[1].set_title('Top 10 Combinations - Regression', fontweight='bold')
        axes[1].grid(axis='x', alpha=0.3)
        
        # Color bars by fingerprint
        fingerprints = reg_combos['fingerprint'].unique()
        colors = plt.cm.Set2(np.linspace(0, 1, len(fingerprints)))
        fp_colors = {fp: colors[i] for i, fp in enumerate(fingerprints)}
        
        for i, (bar, fp) in enumerate(zip(bars, reg_combos['fingerprint'])):
            bar.set_color(fp_colors[fp])
    
    plt.suptitle('Top Performing Fingerprint-Model Combinations', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plot_path = os.path.join(self.save_dir, 'ranking_chart.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_consistency_vs_performance(self, figsize=(12, 8)):
    """Plot showing performance vs consistency (std) for each combination"""
    df = self.get_summary_statistics()
    if df.empty:
        return
    
    # Create combination labels
    df['combination'] = df['fingerprint'] + ' + ' + df['model']
    
    # Separate by task type
    class_df = df[df['is_classification'] == True]
    reg_df = df[df['is_classification'] == False]
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Classification scatter plot
    if not class_df.empty:
        scatter = axes[0].scatter(class_df['std'], class_df['mean'], 
                                 s=100, alpha=0.7, c=range(len(class_df)), cmap='viridis')
        
        # Annotate points with combination names
        for i, row in class_df.iterrows():
            axes[0].annotate(row['combination'], (row['std'], row['mean']), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        axes[0].set_xlabel('Standard Deviation (Consistency)')
        axes[0].set_ylabel('Mean AUROC (Performance)')
        axes[0].set_title('Performance vs Consistency - Classification', fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Add ideal region (high performance, low std)
        axes[0].axhline(y=class_df['mean'].quantile(0.75), color='red', linestyle='--', alpha=0.5, label='Top 25% Performance')
        axes[0].axvline(x=class_df['std'].quantile(0.25), color='blue', linestyle='--', alpha=0.5, label='Top 25% Consistency')
        axes[0].legend()
    
    # Regression scatter plot
    if not reg_df.empty:
        scatter = axes[1].scatter(reg_df['std'], reg_df['mean'], 
                                 s=100, alpha=0.7, c=range(len(reg_df)), cmap='viridis')
        
        # Annotate points
        for i, row in reg_df.iterrows():
            axes[1].annotate(row['combination'], (row['std'], row['mean']), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        axes[1].set_xlabel('Standard Deviation (Consistency)')
        axes[1].set_ylabel('Mean RMSE (Performance)')
        axes[1].set_title('Performance vs Consistency - Regression', fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        # Add ideal region (low RMSE, low std)
        axes[1].axhline(y=reg_df['mean'].quantile(0.25), color='red', linestyle='--', alpha=0.5, label='Top 25% Performance')
        axes[1].axvline(x=reg_df['std'].quantile(0.25), color='blue', linestyle='--', alpha=0.5, label='Top 25% Consistency')
        axes[1].legend()
    
    plt.suptitle('Performance vs Consistency Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plot_path = os.path.join(self.save_dir, 'performance_vs_consistency.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_winner_summary(self, figsize=(14, 8)):
    """Create a summary plot showing winners for each dataset"""
    if not self.statistical_results or 'dataset_winners' not in self.statistical_results:
        print("Run statistical analysis first!")
        return
    
    winners_data = self.statistical_results['dataset_winners']
    if not winners_data:
        return
    
    # Prepare data
    datasets = []
    winners = []
    scores = []
    significant = []
    task_types = []
    
    df = self.get_summary_statistics()
    
    for dataset, winner_info in winners_data.items():
        datasets.append(dataset)
        winners.append(winner_info['winner'])
        scores.append(winner_info['score'])
        significant.append(winner_info.get('significant', False))
        
        # Determine task type
        dataset_df = df[df['dataset'] == dataset]
        if not dataset_df.empty:
            is_class = dataset_df['is_classification'].iloc[0]
            task_types.append('Classification' if is_class else 'Regression')
        else:
            task_types.append('Unknown')
    
    # Create DataFrame
    winners_df = pd.DataFrame({
        'dataset': datasets,
        'winner': winners,
        'score': scores,
        'significant': significant,
        'task_type': task_types
    })
    
    # Separate by task type
    class_winners = winners_df[winners_df['task_type'] == 'Classification']
    reg_winners = winners_df[winners_df['task_type'] == 'Regression']
    
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    
    # Classification winners
    if not class_winners.empty:
        colors = ['green' if sig else 'orange' for sig in class_winners['significant']]
        bars = axes[0].barh(range(len(class_winners)), class_winners['score'], color=colors)
        axes[0].set_yticks(range(len(class_winners)))
        axes[0].set_yticklabels([f"{row['dataset']}\n{row['winner']}" for _, row in class_winners.iterrows()])
        axes[0].set_xlabel('AUROC')
        axes[0].set_title('Dataset Winners - Classification', fontweight='bold')
        axes[0].grid(axis='x', alpha=0.3)
        
        # Add significance legend
        green_patch = patches.Patch(color='green', label='Significantly better')
        orange_patch = patches.Patch(color='orange', label='Not significantly better')
        axes[0].legend(handles=[green_patch, orange_patch])
    
    # Regression winners
    if not reg_winners.empty:
        colors = ['green' if sig else 'orange' for sig in reg_winners['significant']]
        bars = axes[1].barh(range(len(reg_winners)), reg_winners['score'], color=colors)
        axes[1].set_yticks(range(len(reg_winners)))
        axes[1].set_yticklabels([f"{row['dataset']}\n{row['winner']}" for _, row in reg_winners.iterrows()])
        axes[1].set_xlabel('RMSE')
        axes[1].set_title('Dataset Winners - Regression', fontweight='bold')
        axes[1].grid(axis='x', alpha=0.3)
        
        # Add significance legend
        green_patch = patches.Patch(color='green', label='Significantly better')
        orange_patch = patches.Patch(color='orange', label='Not significantly better')
        axes[1].legend(handles=[green_patch, orange_patch])
    
    plt.suptitle('Best Combination for Each Dataset', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plot_path = os.path.join(self.save_dir, 'dataset_winners.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_component_importance(self, figsize=(12, 6)):
    """Plot showing relative importance of fingerprints vs models"""
    if not self.statistical_results or 'component_anova' not in self.statistical_results:
        print("Run statistical analysis first!")
        return
    
    anova_results = self.statistical_results['component_anova']
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    for i, (task_type, results) in enumerate(anova_results.items()):
        if 'error' in results:
            continue
            
        ax = axes[i] if len(anova_results) > 1 else axes
        
        # Get fingerprint and model means
        fp_means = results['fingerprint_means']
        model_means = results['model_means']
        
        # Create side-by-side comparison
        x_pos = np.arange(max(len(fp_means), len(model_means)))
        width = 0.35
        
        # Plot fingerprints
        fp_bars = ax.bar(x_pos[:len(fp_means)] - width/2, fp_means.values, width, 
                        label='Fingerprints', alpha=0.8, color='skyblue')
        
        # Plot models
        model_bars = ax.bar(x_pos[:len(model_means)] + width/2, model_means.values, width,
                           label='Models', alpha=0.8, color='lightcoral')
        
        # Customize
        ax.set_xlabel('Component Rank')
        ax.set_ylabel('AUROC' if 'Classification' in task_type else 'RMSE')
        ax.set_title(f'{task_type} - Component Performance', fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add component labels
        max_len = max(len(fp_means), len(model_means))
        labels = []
        for j in range(max_len):
            fp_label = fp_means.index[j] if j < len(fp_means) else ''
            model_label = model_means.index[j] if j < len(model_means) else ''
            labels.append(f'{j+1}')
        
        ax.set_xticks(x_pos[:max_len])
        ax.set_xticklabels(labels)
        
        # Add text annotations
        for j, (bar, name) in enumerate(zip(fp_bars, fp_means.index)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   name, ha='center', va='bottom', rotation=45, fontsize=8)
        
        for j, (bar, name) in enumerate(zip(model_bars, model_means.index)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   name, ha='center', va='bottom', rotation=45, fontsize=8)
    
    plt.suptitle('Component Performance Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plot_path = os.path.join(self.save_dir, 'component_importance.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()

def create_comprehensive_report(self, figsize_small=(12, 8), figsize_large=(16, 10)):
    """Create all visualization plots for comprehensive analysis"""
    print("Creating comprehensive visualization report...")
    
    # 1. Performance heatmap
    print("1. Creating performance heatmap...")
    self.plot_heatmap_matrix(figsize=figsize_large)
    
    # 2. Ranking chart
    print("2. Creating ranking chart...")
    self.plot_ranking_chart(figsize=figsize_large)
    
    # 3. Performance vs consistency
    print("3. Creating performance vs consistency plot...")
    self.plot_consistency_vs_performance(figsize=figsize_small)
    
    # 4. Dataset winners
    print("4. Creating dataset winners plot...")
    self.plot_winner_summary(figsize=figsize_large)
    
    # 5. Component importance
    print("5. Creating component importance plot...")
    self.plot_component_importance(figsize=figsize_small)
    
    # 6. Original detailed comparison (modified)
    print("6. Creating detailed comparison by fingerprint...")
    self.plot_detailed_comparison(figsize=figsize_large)
    
    print("Comprehensive visualization report complete!")