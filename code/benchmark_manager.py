import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from collections import defaultdict
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
        """Create detailed comparison plots"""
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

        
        for idx, dataset in enumerate(datasets):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            dataset_df = df[df['dataset'] == dataset]

            metric_name = dataset_df['metric'].iloc[0]
            
            # Create grouped bar plot
            fingerprints = sorted(dataset_df['fingerprint'].unique())
            models = sorted(dataset_df['model'].unique())
            
            x = np.arange(len(fingerprints))
            width = 0.8 / len(models)
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
            
            for i, model in enumerate(models):
                model_data = []
                for fp in fingerprints:
                    subset = dataset_df[(dataset_df['fingerprint'] == fp) & 
                                      (dataset_df['model'] == model)]
                    if len(subset) > 0:
                        model_data.append(subset['score'].iloc[0])
                    else:
                        model_data.append(np.nan)
                
                # Filter out NaN values for plotting
                valid_indices = ~np.isnan(model_data)
                if np.any(valid_indices):
                    ax.bar(x[valid_indices] + i * width, np.array(model_data)[valid_indices], 
                          width, label=model, color=colors[i], alpha=0.8)
            
            
            ax.set_xlabel('Fingerprint')
            ax.set_ylabel(metric_name)
            ax.set_title(f'{dataset}')
            ax.set_xticks(x + width * (len(models) - 1) / 2)
            ax.set_xticklabels(fingerprints, rotation=45, ha='right', fontsize=8)
            ax.legend(fontsize=8)
            ax.grid(axis='y', alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(datasets), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.save_dir, f'detailed_comparison_{self.run_timestamp}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Detailed comparison saved to: {plot_path}")
    
    def analyze_and_visualize(self):
        """Run complete analysis and visualization suite"""
        print("Generating comprehensive benchmark analysis...")
        print("="*50)
        
        # Generate summary
        self.generate_report()
        
        print("\n" + "="*50)
        print("GENERATING VISUALIZATIONS...")
        print("="*50)
        
        # Create visualizations
        self.plot_detailed_comparison()
        
        # Save results
        self.save_results()
        
        print("\n" + "="*50)
        print("ANALYSIS COMPLETE!")
        print("="*50)
        print(f"All outputs saved to: {self.save_dir}")
        
        return self.get_summary_stats()