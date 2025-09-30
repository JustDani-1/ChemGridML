import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, mean_squared_error
from typing import Dict, List, Tuple, Optional
import os, sys
from pathlib import Path
from scipy.stats import friedmanchisquare, ttest_rel
from scipy.stats import rankdata
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import warnings
from database_manager import DatabaseManager
warnings.filterwarnings('ignore', category=RuntimeWarning)

class BenchmarkManager:
    """Enhanced analyzer for molecular property prediction with combined GRAPH and fingerprint results"""
    
    def __init__(self, fingerprint_db_manager, graph_db_manager, save_dir: str = "analysis_results"):
        """
        Initialize with separate database managers for fingerprint and graph experiments
        
        Args:
            fingerprint_db_manager: DatabaseManager for fingerprint experiments
            graph_db_manager: DatabaseManager for graph experiments
            save_dir: Directory to save results
        """
        self.fingerprint_db = fingerprint_db_manager
        self.graph_db = graph_db_manager
        self.save_dir = save_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        self.results = {}
        self.statistical_results = {}
    
    def is_classification_dataset(self, dataset_name: str, db_manager) -> bool:
        """Determine if dataset is classification by checking if all targets are 0 or 1"""
        with db_manager._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT DISTINCT target_value FROM dataset_targets 
                WHERE dataset_name = ?
            ''', (dataset_name,))
            
            unique_targets = [row[0] for row in cursor.fetchall()]
            return all(target in [0.0, 1.0] for target in unique_targets)
    
    def compute_auroc(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute AUROC score for binary classification"""
        try:
            return roc_auc_score(y_true, y_pred)
        except ValueError as e:
            print(f"Warning: Could not compute AUROC - {e}")
            return np.nan
    
    def compute_rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Root Mean Square Error for regression"""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    def get_test_predictions(self, dataset_name: str, db_manager) -> pd.DataFrame:
        """Get all test predictions for a dataset"""
        df = db_manager.get_predictions_dataframe(dataset_name)
        return df[df['split_type'] == 'random']
    
    def compute_metrics_for_dataset(self, dataset_name: str, db_manager, method_type: str) -> Dict:
        """Compute metrics for a dataset from either fingerprint or graph experiments"""
        print(f"Processing {method_type} - {dataset_name}")
        
        test_df = self.get_test_predictions(dataset_name, db_manager)
        
        if test_df.empty:
            print(f"No test predictions found for {method_type} - {dataset_name}")
            return {}
        
        is_classification = self.is_classification_dataset(dataset_name, db_manager)
        metric_name = "AUROC" if is_classification else "RMSE"
        
        results = {
            'dataset': dataset_name,
            'metric': metric_name,
            'is_classification': is_classification,
            'method_type': method_type,
            'scores': []
        }
        
        # Group by fingerprint/model/seed
        groups = test_df.groupby(['fingerprint', 'model_name', 'seed'])
        
        for (fingerprint, model, seed), group in groups:
            y_true = group['target_value'].values
            y_pred = group['prediction'].values
            
            score = self.compute_auroc(y_true, y_pred) if is_classification else self.compute_rmse(y_true, y_pred)
            
            results['scores'].append({
                'fingerprint': fingerprint,
                'model': model,
                'seed': seed,
                'score': score,
                'n_samples': len(y_true),
                'method_type': method_type
            })
        
        return results
    
    def analyze_all_datasets(self) -> Dict:
        """Analyze all datasets from both fingerprint and graph experiments"""
        # Get datasets from fingerprint experiments
        with self.fingerprint_db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT DISTINCT dataset_name FROM dataset_targets')
            fingerprint_datasets = [row[0] for row in cursor.fetchall()]
        
        # Get datasets from graph experiments
        with self.graph_db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT DISTINCT dataset_name FROM dataset_targets')
            graph_datasets = [row[0] for row in cursor.fetchall()]
        
        # Find common datasets
        common_datasets = set(fingerprint_datasets).intersection(set(graph_datasets))
        print(f"Found {len(common_datasets)} common datasets: {sorted(common_datasets)}")
        
        # Analyze each dataset from both sources
        for dataset in sorted(common_datasets):
            fp_results = self.compute_metrics_for_dataset(dataset, self.fingerprint_db, 'FINGERPRINT')
            graph_results = self.compute_metrics_for_dataset(dataset, self.graph_db, 'GRAPH')
            
            # Combine results
            if fp_results and graph_results:
                combined_scores = fp_results['scores'] + graph_results['scores']
                self.results[dataset] = {
                    'dataset': dataset,
                    'metric': fp_results['metric'],
                    'is_classification': fp_results['is_classification'],
                    'scores': combined_scores
                }
        
        return self.results
    
    def get_detailed_scores(self) -> pd.DataFrame:
        """Get individual seed scores for statistical analysis"""
        detailed_data = []
        
        for dataset_name, dataset_results in self.results.items():
            if not dataset_results or not dataset_results['scores']:
                continue
                
            scores_df = pd.DataFrame(dataset_results['scores'])
            scores_df['dataset'] = dataset_name
            scores_df['metric'] = dataset_results['metric']
            scores_df['is_classification'] = dataset_results['is_classification']
            scores_df['method'] = scores_df['fingerprint'] + '_' + scores_df['model']
            
            detailed_data.append(scores_df)
        
        if detailed_data:
            return pd.concat(detailed_data, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def friedman_ranking_analysis(self) -> Dict:
        """Perform Friedman test including both fingerprint and graph methods"""
        print("\n=== Friedman Ranking Analysis (Combined FINGERPRINT + GRAPH) ===")
        
        detailed_df = self.get_detailed_scores()
        if detailed_df.empty:
            return {}
        
        classification_df = detailed_df[detailed_df['is_classification'] == True]
        regression_df = detailed_df[detailed_df['is_classification'] == False]
        
        results = {}
        
        for task_type, df in [('Classification', classification_df), ('Regression', regression_df)]:
            if df.empty:
                continue
                
            print(f"\n{task_type} Datasets:")
            
            datasets = df['dataset'].unique()
            common_methods = set(df['method'].unique())
            
            for dataset in datasets:
                dataset_methods = set(df[df['dataset'] == dataset]['method'].unique())
                common_methods = common_methods.intersection(dataset_methods)
            
            common_methods = sorted(list(common_methods))
            
            if len(common_methods) < 2:
                print(f"Insufficient common methods ({len(common_methods)})")
                results[task_type] = {'error': 'Insufficient common methods'}
                continue
            
            # Separate fingerprint and graph methods
            fp_methods = [m for m in common_methods if df[df['method']==m]['method_type'].iloc[0] == 'FINGERPRINT']
            graph_methods = [m for m in common_methods if df[df['method']==m]['method_type'].iloc[0] == 'GRAPH']
            
            print(f"Fingerprint methods: {len(fp_methods)}, Graph methods: {len(graph_methods)}")
            
            rank_matrix = []
            for dataset in datasets:
                dataset_data = df[df['dataset'] == dataset]
                method_means = dataset_data.groupby('method')['score'].mean()
                method_means = method_means[common_methods]
                
                if len(method_means) != len(common_methods):
                    continue
                
                ranks = rankdata(-method_means.values if task_type == 'Classification' else method_means.values, method='average')
                rank_matrix.append(ranks)
            
            if len(rank_matrix) < 2:
                results[task_type] = {'error': 'Insufficient datasets'}
                continue
            
            rank_matrix = np.array(rank_matrix)
            
            try:
                statistic, p_value = friedmanchisquare(*rank_matrix.T)
                mean_ranks = np.mean(rank_matrix, axis=0)
                method_rankings = list(zip(common_methods, mean_ranks))
                method_rankings.sort(key=lambda x: x[1])
                
                results[task_type] = {
                    'friedman_stat': statistic,
                    'friedman_p': p_value,
                    'rankings': method_rankings,
                    'significant': p_value < 0.05,
                    'n_datasets': len(rank_matrix),
                    'n_methods': len(common_methods),
                    'fingerprint_methods': fp_methods,
                    'graph_methods': graph_methods
                }
                
                print(f"Friedman test: χ² = {statistic:.3f}, p = {p_value:.4f}")
                print(f"Top 5 methods (FINGERPRINT and GRAPH combined):")
                for i, (method, rank) in enumerate(method_rankings[:5]):
                    fp, model = method.split('_')
                    print(f"  {i+1}. {fp} + {model} (avg rank: {rank:.2f})")
                    
            except Exception as e:
                print(f"Error in Friedman test: {e}")
                results[task_type] = {'error': str(e)}
        
        return results
    
    def plot_combined_comparison(self, figsize=(20, 12)):
        """Create combined comparison plot with fingerprint and GRAPH models grouped by model"""
        if not self.results:
            print("No results to plot. Run analyze_all_datasets() first.")
            return
        
        detailed_df = self.get_detailed_scores()
        if detailed_df.empty:
            print("No valid results to plot.")
            return
        
        # Compute summary statistics
        summary_df = detailed_df.groupby(['dataset', 'fingerprint', 'model', 'method_type', 'is_classification', 'metric']).agg({
            'score': ['mean', 'std', 'count']
        }).reset_index()
        summary_df.columns = ['dataset', 'fingerprint', 'model', 'method_type', 'is_classification', 'metric', 'mean', 'std', 'count']
        
        # Sort datasets
        classification_datasets = sorted([d for d in summary_df['dataset'].unique() if summary_df[summary_df['dataset']==d]['is_classification'].iloc[0]])
        regression_datasets = sorted([d for d in summary_df['dataset'].unique() if not summary_df[summary_df['dataset']==d]['is_classification'].iloc[0]])
        datasets = classification_datasets + regression_datasets
        
        n_datasets = len(datasets)
        n_cols = min(3, n_datasets)
        n_rows = (n_datasets + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = np.array(axes).flatten() if n_datasets > 1 else [axes]
        
        fig.suptitle('Performance by Dataset and Model (FINGERPRINT vs GRAPH) - Mean ± Std across seeds',
                    fontsize=16, fontweight='bold')
        
        for idx, dataset in enumerate(datasets):
            if idx >= len(axes):
                break
            
            ax = axes[idx]
            dataset_df = summary_df[summary_df['dataset'] == dataset]
            metric_name = dataset_df['metric'].iloc[0]
            is_classification = dataset_df['is_classification'].iloc[0]
            
            # Separate fingerprint and graph data
            fp_df = dataset_df[dataset_df['method_type'] == 'FINGERPRINT']
            graph_df = dataset_df[dataset_df['method_type'] == 'GRAPH']
            
            # Get unique models and fingerprints
            models = sorted(set(list(fp_df['model'].unique()) + list(graph_df['model'].unique())))
            fingerprints = sorted(fp_df['fingerprint'].unique()) if not fp_df.empty else []
            graph_types = sorted(graph_df['fingerprint'].unique()) if not graph_df.empty else []
            
            # Color map for fingerprints
            fp_colors = plt.cm.Set2(np.linspace(0, 1, len(fingerprints)))
            fp_color_map = {fp: fp_colors[i] for i, fp in enumerate(fingerprints)}
            
            # Color map for graph methods (using different colormap)
            graph_colors = plt.cm.Set1(np.linspace(0, 1, len(graph_types)))
            graph_color_map = {g: graph_colors[i] for i, g in enumerate(graph_types)}
            
            n_fingerprints = len(fingerprints)
            n_graphs = len(graph_types)
            n_models = len(models)
            
            # Calculate bar widths and positions
            group_width = 0.8
            n_total_bars = n_fingerprints + n_graphs
            bar_width = group_width / n_total_bars if n_total_bars > 0 else 0.1
            
            model_positions = np.arange(n_models)
            
            # Plot fingerprint methods (left side of each model group)
            for fp_idx, fingerprint in enumerate(fingerprints):
                fp_means = []
                fp_stds = []
                fp_positions = []
                
                for model_idx, model in enumerate(models):
                    subset = fp_df[(fp_df['fingerprint'] == fingerprint) & (fp_df['model'] == model)]
                    if len(subset) > 0:
                        fp_means.append(subset['mean'].iloc[0])
                        fp_stds.append(subset['std'].iloc[0] if pd.notna(subset['std'].iloc[0]) else 0)
                    else:
                        fp_means.append(0)
                        fp_stds.append(0)
                    
                    pos = model_positions[model_idx] + (fp_idx - (n_total_bars-1)/2) * bar_width
                    fp_positions.append(pos)
                
                ax.bar(fp_positions, fp_means, bar_width * 0.9,
                      label=fingerprint, color=fp_color_map[fingerprint],
                      alpha=0.8, edgecolor='white', linewidth=0.5,
                      yerr=fp_stds, capsize=3, error_kw={'alpha': 0.6})
            
            # Plot graph methods (right side of each model group)
            for graph_idx, graph_type in enumerate(graph_types):
                graph_means = []
                graph_stds = []
                graph_positions = []
                
                for model_idx, model in enumerate(models):
                    subset = graph_df[(graph_df['fingerprint'] == graph_type) & (graph_df['model'] == model)]
                    if len(subset) > 0:
                        graph_means.append(subset['mean'].iloc[0])
                        graph_stds.append(subset['std'].iloc[0] if pd.notna(subset['std'].iloc[0]) else 0)
                    else:
                        graph_means.append(0)
                        graph_stds.append(0)
                    
                    # Position after all fingerprints
                    pos = model_positions[model_idx] + ((n_fingerprints + graph_idx) - (n_total_bars-1)/2) * bar_width
                    graph_positions.append(pos)
                
                ax.bar(graph_positions, graph_means, bar_width * 0.9,
                      label=f'{graph_type}', color=graph_color_map[graph_type],
                      alpha=0.9, edgecolor='black', linewidth=1.5,
                      yerr=graph_stds, capsize=3, error_kw={'alpha': 0.6})
            
            ax.set_ylabel(metric_name, fontsize=10)
            ax.set_title(dataset, fontsize=11, fontweight='bold')
            ax.set_xticks(model_positions)
            ax.set_xticklabels(models, rotation=45, ha='right')
            
            # Add vertical lines to separate model groups
            for i in range(1, len(models)):
                ax.axvline(x=model_positions[i] - 0.5, color='gray',
                          linestyle=':', alpha=0.5, linewidth=1)
            
            if idx == 0:
                ax.legend(fontsize=8, loc='upper right')
            
            ax.grid(axis='y', alpha=0.3)
            
            if is_classification:
                ax.set_ylim(0, 1)
            else:
                ax.set_ylim(bottom=0)
        
        for idx in range(len(datasets), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        plot_path = os.path.join(self.save_dir, 'combined_fingerprint_graph_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved plot to: {plot_path}")
        plt.show()
    
    def print_summary(self):
        """Print summary including both fingerprint and graph methods"""
        if not self.results:
            print("No results available.")
            return
        
        print("\n" + "="*70)
        print("COMBINED ANALYSIS SUMMARY (FINGERPRINT + GRAPH)")
        print("="*70)
        
        detailed_df = self.get_detailed_scores()
        
        for dataset in sorted(self.results.keys()):
            dataset_results = self.results[dataset]
            if not dataset_results or not dataset_results['scores']:
                continue
            
            dataset_df = detailed_df[detailed_df['dataset'] == dataset]
            metric_name = dataset_results['metric']
            is_classification = dataset_results['is_classification']
            
            print(f"\nDataset: {dataset} ({metric_name})")
            print("-" * 70)
            
            # Best overall
            grouped = dataset_df.groupby(['fingerprint', 'model', 'method_type'])['score'].mean().reset_index()
            grouped['mean_score'] = grouped['score']
            
            if is_classification:
                best = grouped.loc[grouped['mean_score'].idxmax()]
            else:
                best = grouped.loc[grouped['mean_score'].idxmin()]
            
            print(f"Best Overall: {best['fingerprint']} + {best['model']} ({best['method_type']}) = {best['mean_score']:.4f}")
            
            # Best fingerprint method
            fp_df = dataset_df[dataset_df['method_type'] == 'FINGERPRINT']
            if not fp_df.empty:
                fp_grouped = fp_df.groupby(['fingerprint', 'model'])['score'].mean()
                best_fp = fp_grouped.idxmax() if is_classification else fp_grouped.idxmin()
                print(f"Best Fingerprint: {best_fp[0]} + {best_fp[1]} = {fp_grouped[best_fp]:.4f}")
            
            # Best graph method
            graph_df = dataset_df[dataset_df['method_type'] == 'GRAPH']
            if not graph_df.empty:
                graph_grouped = graph_df.groupby(['fingerprint', 'model'])['score'].mean()
                best_graph = graph_grouped.idxmax() if is_classification else graph_grouped.idxmin()
                print(f"Best Graph: {best_graph[0]} + {best_graph[1]} = {graph_grouped[best_graph]:.4f}")


def run_combined_analysis(fingerprint_db_manager, graph_db_manager, save_dir="./combined_analysis"):
    """Run complete combined analysis pipeline"""
    analyzer = BenchmarkManager(fingerprint_db_manager, graph_db_manager, save_dir)
    
    # Analyze all datasets
    analyzer.analyze_all_datasets()
    
    # Run statistical analysis
    friedman_results = analyzer.friedman_ranking_analysis()
    
    # Create combined visualization
    analyzer.plot_combined_comparison()
    
    # Print summary
    analyzer.print_summary()
    
    return analyzer


if __name__ == '__main__':
    master_job_id = sys.argv[1]
    base = Path('./studies')
    path = base / master_job_id
    path_fp = path / "FINGERPRINT" / "predictions.db"
    path_lrn = path / "LEARNABLE" / "predictions.db"
    db_fp = DatabaseManager(path_fp)
    db_lrn = DatabaseManager(path_lrn)
    analyzer = run_combined_analysis(db_fp, db_lrn, path)