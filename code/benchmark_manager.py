import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class BenchmarkManager:
    def __init__(self, save_dir="results"):
        self.results = {}
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Track metadata
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.total_experiments = 0
        self.failed_experiments = 0
        
    def add_result(self, fingerprint, model, dataset, best_score, best_params):
        """Add a single experimental result"""
        if fingerprint not in self.results:
            self.results[fingerprint] = {}
        if model not in self.results[fingerprint]:
            self.results[fingerprint][model] = {}
            
        # Extract metric name and value
        if isinstance(best_score, dict):
            metric_name = list(best_score.keys())[0]
            score_value = best_score[metric_name]
        else:
            # Fallback if score is just a number
            metric_name = "Unknown"
            score_value = best_score
            
        self.results[fingerprint][model][dataset] = {
            'score': score_value,
            'metric': metric_name,
            'params': best_params,
            'timestamp': datetime.now().isoformat()
        }
        
        self.total_experiments += 1
        
    def add_failed_result(self, fingerprint, model, dataset, error_msg):
        """Track failed experiments for analysis"""
        self.failed_experiments += 1
        # Could extend this to store failure reasons for debugging
        
    def to_dataframe(self):
        """Convert results to pandas DataFrame for easier analysis"""
        rows = []
        for fp in self.results:
            for model in self.results[fp]:
                for dataset in self.results[fp][model]:
                    result = self.results[fp][model][dataset]
                    rows.append({
                        'fingerprint': fp,
                        'model': model,
                        'dataset': dataset,
                        'score': result['score'],
                        'metric': result['metric'],
                        'timestamp': result['timestamp']
                    })
        return pd.DataFrame(rows)
    
    def get_summary_stats(self):
        """Generate summary statistics"""
        if not self.results:
            return "No results available yet."
            
        df = self.to_dataframe()
        
        stats = {
            'total_experiments': len(df),
            'failed_experiments': self.failed_experiments,
            'success_rate': len(df) / (len(df) + self.failed_experiments) * 100 if (len(df) + self.failed_experiments) > 0 else 0,
            'unique_fingerprints': df['fingerprint'].nunique(),
            'unique_models': df['model'].nunique(),
            'unique_datasets': df['dataset'].nunique(),
            'metrics_used': df['metric'].unique().tolist()
        }
        
        return stats
    
    def get_best_combinations(self, top_n=5):
        """Find best performing combinations"""
        if not self.results:
            return "No results available."
            
        df = self.to_dataframe()
        
        best_combinations = {}
        
        for metric in df['metric'].unique():
            metric_df = df[df['metric'] == metric]
            
            # Sort based on metric type (AUROC higher is better, RRMSE lower is better)
            ascending = True if metric == 'RRMSE' else False
            sorted_df = metric_df.sort_values('score', ascending=ascending)
            
            best_combinations[metric] = sorted_df.head(top_n)[
                ['fingerprint', 'model', 'dataset', 'score']
            ].to_dict('records')
            
        return best_combinations
    
    def generate_report(self):
        """Generate comprehensive text report"""
        if not self.results:
            return "No results available for report generation."
            
        report_lines = []
        report_lines.append("="*60)
        report_lines.append("MOLECULAR FINGERPRINT BENCHMARKING REPORT")
        report_lines.append("="*60)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Run ID: {self.run_timestamp}")
        report_lines.append("")
        
        # Summary statistics
        stats = self.get_summary_stats()
        report_lines.append("SUMMARY STATISTICS:")
        report_lines.append("-" * 30)
        for key, value in stats.items():
            if key == 'success_rate':
                report_lines.append(f"{key.replace('_', ' ').title()}: {value:.1f}%")
            else:
                report_lines.append(f"{key.replace('_', ' ').title()}: {value}")
        report_lines.append("")
        
        # Best combinations
        best_combos = self.get_best_combinations()
        for metric, combos in best_combos.items():
            report_lines.append(f"TOP 5 COMBINATIONS FOR {metric}:")
            report_lines.append("-" * 30)
            for i, combo in enumerate(combos, 1):
                report_lines.append(
                    f"{i}. {combo['fingerprint']} + {combo['model']} + {combo['dataset']}: "
                    f"{combo['score']:.4f}"
                )
            report_lines.append("")
        
        # Performance analysis by component
        df = self.to_dataframe()
        
        report_lines.append("COMPONENT ANALYSIS:")
        report_lines.append("-" * 30)
        
        # Fingerprint rankings
        fp_performance = df.groupby('fingerprint')['score'].agg(['mean', 'std', 'count'])
        report_lines.append("Fingerprint Performance (average ± std):")
        for fp in fp_performance.index:
            mean_score = fp_performance.loc[fp, 'mean']
            std_score = fp_performance.loc[fp, 'std']
            count = fp_performance.loc[fp, 'count']
            report_lines.append(f"  {fp}: {mean_score:.4f} ± {std_score:.4f} (n={count})")
        report_lines.append("")
        
        # Model rankings
        model_performance = df.groupby('model')['score'].agg(['mean', 'std', 'count'])
        report_lines.append("Model Performance (average ± std):")
        for model in model_performance.index:
            mean_score = model_performance.loc[model, 'mean']
            std_score = model_performance.loc[model, 'std']
            count = model_performance.loc[model, 'count']
            report_lines.append(f"  {model}: {mean_score:.4f} ± {std_score:.4f} (n={count})")
        report_lines.append("")
        
        # Dataset difficulty analysis
        dataset_performance = df.groupby('dataset')['score'].agg(['mean', 'std', 'count'])
        report_lines.append("Dataset Difficulty (average ± std):")
        for dataset in dataset_performance.index:
            mean_score = dataset_performance.loc[dataset, 'mean']
            std_score = dataset_performance.loc[dataset, 'std']
            count = dataset_performance.loc[dataset, 'count']
            report_lines.append(f"  {dataset}: {mean_score:.4f} ± {std_score:.4f} (n={count})")
        
        report_text = "\n".join(report_lines)
        
        # Save report
        report_path = os.path.join(self.save_dir, f'benchmark_report_{self.run_timestamp}.txt')
        with open(report_path, 'w') as f:
            f.write(report_text)
            
        print(report_text)
        print(f"\nFull report saved to: {report_path}")
        
        return report_text
    
    def save_results(self):
        """Save results to JSON for future analysis"""
        results_path = os.path.join(self.save_dir, f'benchmark_results_{self.run_timestamp}.json')
        
        # Prepare serializable data
        serializable_results = {}
        for fp in self.results:
            serializable_results[fp] = {}
            for model in self.results[fp]:
                serializable_results[fp][model] = {}
                for dataset in self.results[fp][model]:
                    result = self.results[fp][model][dataset]
                    serializable_results[fp][model][dataset] = {
                        'score': result['score'],
                        'metric': result['metric'],
                        'params': result['params'],
                        'timestamp': result['timestamp']
                    }
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
    def load_results(self, filepath):
        """Load previously saved results"""
        with open(filepath, 'r') as f:
            self.results = json.load(f)
        print(f"Results loaded from: {filepath}")
    
    def plot_detailed_comparison(self, figsize=(16, 10)):
        """Create detailed comparison plots grouped by model with fingerprint performance"""
        if not self.results:
            print("No results to plot yet.")
            return
        
        df = self.to_dataframe()
        
        # Determine number of subplots needed
        n_datasets = df['dataset'].nunique()
        n_cols = min(3, n_datasets)
        n_rows = (n_datasets + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        fig.suptitle(f'Performance by Dataset and Model\nRun: {self.run_timestamp}',
                    fontsize=16, fontweight='bold')
        
        datasets = sorted(df['dataset'].unique())
        
        # Define consistent colors for fingerprints across all datasets
        all_fingerprints = sorted(df['fingerprint'].unique())
        fingerprint_colors = plt.cm.Set2(np.linspace(0, 1, len(all_fingerprints)))
        fingerprint_color_map = {fp: fingerprint_colors[i] for i, fp in enumerate(all_fingerprints)}
        
        for idx, dataset in enumerate(datasets):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            dataset_df = df[df['dataset'] == dataset]
            metric_name = dataset_df['metric'].iloc[0]
            
            # Get models and fingerprints for this dataset
            models = sorted(dataset_df['model'].unique())
            fingerprints = sorted(dataset_df['fingerprint'].unique())
            
            # Create positions for grouped bars
            n_fingerprints = len(fingerprints)
            n_models = len(models)
            
            # Width calculations
            group_width = 0.8
            bar_width = group_width / n_fingerprints
            
            # Calculate model averages
            model_averages = {}
            for model in models:
                model_data = dataset_df[dataset_df['model'] == model]
                model_averages[model] = model_data['score'].mean()
            
            # Position models on x-axis
            model_positions = np.arange(n_models)
            
            # Create bars for each fingerprint within each model group
            for fp_idx, fingerprint in enumerate(fingerprints):
                fp_scores = []
                fp_positions = []
                
                for model_idx, model in enumerate(models):
                    subset = dataset_df[(dataset_df['fingerprint'] == fingerprint) &
                                    (dataset_df['model'] == model)]
                    
                    if len(subset) > 0:
                        score = subset['score'].iloc[0]
                        fp_scores.append(score)
                        # Calculate position within model group
                        pos = model_positions[model_idx] + (fp_idx - (n_fingerprints-1)/2) * bar_width
                        fp_positions.append(pos)
                    else:
                        fp_scores.append(0)  # or np.nan if you prefer
                        pos = model_positions[model_idx] + (fp_idx - (n_fingerprints-1)/2) * bar_width
                        fp_positions.append(pos)
                
                # Plot bars for this fingerprint across all models
                ax.bar(fp_positions, fp_scores, bar_width * 0.9, 
                    label=fingerprint, color=fingerprint_color_map[fingerprint], 
                    alpha=0.8, edgecolor='white', linewidth=0.5)
            
            # Add model average lines/markers
            for model_idx, model in enumerate(models):
                avg_score = model_averages[model]
                # Draw a horizontal line across the model group showing average
                left_edge = model_positions[model_idx] - group_width/2
                right_edge = model_positions[model_idx] + group_width/2
                ax.hlines(avg_score, left_edge, right_edge, 
                        colors='red', linestyles='--', linewidth=2, alpha=0.7)
                
                # Add average value as text
                ax.text(model_positions[model_idx], avg_score + 0.02 * ax.get_ylim()[1], 
                    f'{avg_score:.3f}', ha='center', va='bottom', 
                    fontweight='bold', fontsize=8, color='red')
            
            # Customize the plot
            ax.set_xlabel('Model')
            ax.set_ylabel(metric_name)
            ax.set_title(f'{dataset}')
            ax.set_xticks(model_positions)
            ax.set_xticklabels(models, rotation=45, ha='right')
            
            # Add vertical lines to separate model groups
            for i in range(1, len(models)):
                ax.axvline(x=model_positions[i] - 0.5, color='gray', 
                        linestyle=':', alpha=0.5, linewidth=1)
            
            # Only show legend on first subplot to avoid redundancy
            if idx == 0:
                # Create custom legend with fingerprints and model average
                legend_elements = [plt.Rectangle((0,0),1,1, facecolor=fingerprint_color_map[fp], 
                                            alpha=0.8, label=fp) for fp in fingerprints]
                legend_elements.append(plt.Line2D([0], [0], color='red', linestyle='--', 
                                                linewidth=2, label='Model Average'))
                ax.legend(handles=legend_elements, fontsize=8, loc='upper right')
            
            ax.grid(axis='y', alpha=0.3)
            
            # Set y-axis to start from 0 for better comparison
            ax.set_ylim(bottom=0)
        
        # Hide unused subplots
        for idx in range(len(datasets), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.save_dir, f'detailed_comparison_{self.run_timestamp}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()


    def plot_model_summary(self, figsize=(14, 8)):
        """Create a summary plot showing overall model performance across all datasets"""
        if not self.results:
            print("No results to plot yet.")
            return
        
        df = self.to_dataframe()
        
        # Calculate overall model averages across all datasets and fingerprints
        model_summary = df.groupby('model').agg({
            'score': ['mean', 'std', 'count']
        }).round(4)
        model_summary.columns = ['mean_score', 'std_score', 'count']
        model_summary = model_summary.sort_values('mean_score', ascending=False)
        
        # Create the summary plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(f'Model Performance Summary\nRun: {self.run_timestamp}', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Overall model averages with error bars
        models = model_summary.index
        means = model_summary['mean_score']
        stds = model_summary['std_score']
        
        bars = ax1.bar(range(len(models)), means, yerr=stds, capsize=5, 
                    alpha=0.7, color=plt.cm.Set1(np.linspace(0, 1, len(models))))
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Average Score')
        ax1.set_title('Overall Model Performance\n(with standard deviation)')
        ax1.set_xticks(range(len(models)))
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax1.text(i, mean + std + 0.01 * ax1.get_ylim()[1], f'{mean:.3f}', 
                    ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Detailed breakdown by model and fingerprint
        model_fp_pivot = df.pivot_table(values='score', index='model', 
                                        columns='fingerprint', aggfunc='mean')
        
        # Create heatmap
        im = ax2.imshow(model_fp_pivot.values, cmap='RdYlGn', aspect='auto')
        
        # Set ticks and labels
        ax2.set_xticks(range(len(model_fp_pivot.columns)))
        ax2.set_yticks(range(len(model_fp_pivot.index)))
        ax2.set_xticklabels(model_fp_pivot.columns, rotation=45, ha='right')
        ax2.set_yticklabels(model_fp_pivot.index)
        ax2.set_xlabel('Fingerprint')
        ax2.set_ylabel('Model')
        ax2.set_title('Performance Heatmap\n(Average across datasets)')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Average Score')
        
        # Add text annotations
        for i in range(len(model_fp_pivot.index)):
            for j in range(len(model_fp_pivot.columns)):
                value = model_fp_pivot.iloc[i, j]
                if not np.isnan(value):
                    ax2.text(j, i, f'{value:.3f}', ha='center', va='center', 
                            color='white' if value < model_fp_pivot.values.mean() else 'black',
                            fontweight='bold', fontsize=8)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.save_dir, f'model_summary_{self.run_timestamp}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return model_summary
    
    def analyze_and_visualize(self):
        """Run complete analysis and visualization suite"""
        
        # Generate summary
        self.generate_report()
        
        # Create visualizations
        self.plot_detailed_comparison()
        
        # Save results
        self.save_results()
        
        return self.get_summary_stats()