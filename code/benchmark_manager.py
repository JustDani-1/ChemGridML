# benchmark_manager.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, mean_squared_error
from typing import Dict, List, Tuple, Optional
import os, sys
from pathlib import Path
from datetime import datetime
from database_manager import DatabaseManager
from scipy.stats import friedmanchisquare, ttest_rel
from scipy.stats import rankdata
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

class BenchmarkManager:
    """Analyzer for molecular property prediction results with multiple train-test splits"""
    
    def __init__(self, db_manager, save_dir: str = "analysis_results"):
        self.db_manager = db_manager
        self.save_dir = save_dir
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Store computed results
        self.results = {}
        self.statistical_results = {}
    
    def is_classification_dataset(self, dataset_name: str) -> bool:
        """Determine if dataset is classification by checking if all targets are 0 or 1"""
        with self.db_manager._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT DISTINCT target_value FROM dataset_targets 
                WHERE dataset_name = ?
            ''', (dataset_name,))
            
            unique_targets = [row[0] for row in cursor.fetchall()]
            
            # Check if all values are either 0 or 1
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
    
    def get_test_predictions(self, dataset_name: str) -> pd.DataFrame:
        """Get all test predictions for a dataset"""
        df = self.db_manager.get_predictions_dataframe(dataset_name)
        return df[df['split_type'] == 'random']
    
    def compute_metrics_for_dataset(self, dataset_name: str) -> Dict:
        """Compute metrics for all fingerprint/model/seed combinations in a dataset"""
        print(f"Processing dataset: {dataset_name}")
        
        # Get test predictions
        test_df = self.get_test_predictions(dataset_name)
        
        if test_df.empty:
            print(f"No test predictions found for dataset {dataset_name}")
            return {}
        
        # Determine if classification or regression
        is_classification = self.is_classification_dataset(dataset_name)
        metric_name = "AUROC" if is_classification else "RMSE"
        
        print(f"Dataset {dataset_name} identified as {'classification' if is_classification else 'regression'}")
        
        results = {
            'dataset': dataset_name,
            'metric': metric_name,
            'is_classification': is_classification,
            'scores': []
        }
        
        # Group by fingerprint, model, and seed
        groups = test_df.groupby(['fingerprint', 'model_name', 'seed'])
        
        for (fingerprint, model, seed), group in groups:
            y_true = group['target_value'].values
            y_pred = group['prediction'].values
            
            if is_classification:
                score = self.compute_auroc(y_true, y_pred)
            else:
                score = self.compute_rmse(y_true, y_pred)
            
            results['scores'].append({
                'fingerprint': fingerprint,
                'model': model,
                'seed': seed,
                'score': score,
                'n_samples': len(y_true)
            })
        
        print(f"Computed {len(results['scores'])} metric scores for {dataset_name}")
        return results
    
    def analyze_all_datasets(self) -> Dict:
        """Analyze all datasets in the database"""
        # Get all unique datasets
        with self.db_manager._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT DISTINCT dataset_name FROM dataset_targets')
            datasets = [row[0] for row in cursor.fetchall()]
        
        print(f"Found {len(datasets)} datasets: {datasets}")
        
        # Analyze each dataset
        for dataset in datasets:
            self.results[dataset] = self.compute_metrics_for_dataset(dataset)
        
        return self.results
    
    def get_summary_statistics(self) -> pd.DataFrame:
        """Get summary statistics across all seeds for each fingerprint/model/dataset combination"""
        summary_data = []
        
        for dataset_name, dataset_results in self.results.items():
            if not dataset_results or not dataset_results['scores']:
                continue
                
            # Convert to DataFrame for easier manipulation
            scores_df = pd.DataFrame(dataset_results['scores'])
            
            # Group by fingerprint and model, compute statistics across seeds
            group_stats = scores_df.groupby(['fingerprint', 'model'])['score'].agg([
                'mean', 'std', 'min', 'max', 'count'
            ]).reset_index()
            
            # Add dataset and metric information
            group_stats['dataset'] = dataset_name
            group_stats['metric'] = dataset_results['metric']
            group_stats['is_classification'] = dataset_results['is_classification']
            
            summary_data.append(group_stats)
        
        if summary_data:
            return pd.concat(summary_data, ignore_index=True)
        else:
            return pd.DataFrame()
    
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
        """Perform Friedman test and ranking analysis"""
        print("\n=== Friedman Ranking Analysis ===")
        
        detailed_df = self.get_detailed_scores()
        if detailed_df.empty:
            return {}
        
        # Separate classification and regression datasets
        classification_df = detailed_df[detailed_df['is_classification'] == True]
        regression_df = detailed_df[detailed_df['is_classification'] == False]
        
        results = {}
        
        for task_type, df in [('Classification', classification_df), ('Regression', regression_df)]:
            if df.empty:
                continue
                
            print(f"\n{task_type} Datasets:")
            
            # Create ranking matrix: rows=datasets, columns=methods
            datasets = df['dataset'].unique()
            methods = df['method'].unique()
            
            rank_matrix = []
            for dataset in datasets:
                dataset_data = df[df['dataset'] == dataset]
                method_means = dataset_data.groupby('method')['score'].mean()
                
                # Rank methods (1=best)
                if task_type == 'Classification':  # Higher AUROC is better
                    ranks = rankdata(-method_means.values, method='average')
                else:  # Lower RMSE is better
                    ranks = rankdata(method_means.values, method='average')
                
                rank_matrix.append(ranks)
            
            rank_matrix = np.array(rank_matrix)
            
            # Friedman test
            try:
                statistic, p_value = friedmanchisquare(*rank_matrix.T)
                
                # Calculate mean ranks
                mean_ranks = np.mean(rank_matrix, axis=0)
                method_rankings = list(zip(methods, mean_ranks))
                method_rankings.sort(key=lambda x: x[1])  # Sort by rank (lower is better)
                
                results[task_type] = {
                    'friedman_stat': statistic,
                    'friedman_p': p_value,
                    'rankings': method_rankings,
                    'significant': p_value < 0.05
                }
                
                print(f"Friedman test: χ² = {statistic:.3f}, p = {p_value:.4f}")
                if p_value < 0.05:
                    print("Significant differences between methods detected!")
                    print("Top 5 methods:")
                    for i, (method, rank) in enumerate(method_rankings[:5]):
                        fp, model = method.split('_')
                        print(f"  {i+1}. {fp} + {model} (avg rank: {rank:.2f})")
                else:
                    print("No significant differences between methods")
                    
            except Exception as e:
                print(f"Error in Friedman test: {e}")
                results[task_type] = {'error': str(e)}
        
        return results
    
    def component_anova_analysis(self) -> Dict:
        """Perform ANOVA analysis on fingerprints and models"""
        print("\n=== Component ANOVA Analysis ===")
        
        detailed_df = self.get_detailed_scores()
        if detailed_df.empty:
            return {}
        
        # Separate classification and regression
        classification_df = detailed_df[detailed_df['is_classification'] == True]
        regression_df = detailed_df[detailed_df['is_classification'] == False]
        
        results = {}
        
        for task_type, df in [('Classification', classification_df), ('Regression', regression_df)]:
            if df.empty:
                continue
                
            print(f"\n{task_type} Datasets:")
            
            try:
                # Main effects model
                model_main = ols('score ~ C(fingerprint) + C(model) + C(dataset)', data=df).fit()
                anova_main = anova_lm(model_main)
                
                # Interaction model  
                model_int = ols('score ~ C(fingerprint) * C(model) + C(dataset)', data=df).fit()
                anova_int = anova_lm(model_int)
                
                # Component means
                fp_means = df.groupby('fingerprint')['score'].mean().sort_values(ascending=(task_type=='Regression'))
                model_means = df.groupby('model')['score'].mean().sort_values(ascending=(task_type=='Regression'))
                
                results[task_type] = {
                    'anova_main': anova_main,
                    'anova_interaction': anova_int,
                    'fingerprint_means': fp_means,
                    'model_means': model_means
                }
                
                # Print results
                fp_p = anova_main.loc['C(fingerprint)', 'PR(>F)']
                model_p = anova_main.loc['C(model)', 'PR(>F)']
                int_p = anova_int.loc['C(fingerprint):C(model)', 'PR(>F)']
                
                print(f"Fingerprint effect: p = {fp_p:.4f} {'***' if fp_p < 0.001 else '**' if fp_p < 0.01 else '*' if fp_p < 0.05 else ''}")
                print(f"Model effect: p = {model_p:.4f} {'***' if model_p < 0.001 else '**' if model_p < 0.01 else '*' if model_p < 0.05 else ''}")
                print(f"Interaction effect: p = {int_p:.4f} {'***' if int_p < 0.001 else '**' if int_p < 0.01 else '*' if int_p < 0.05 else ''}")
                
                if fp_p < 0.05:
                    print("Best fingerprints:")
                    for i, (fp, score) in enumerate(fp_means.items()):
                        if i < 3:
                            print(f"  {i+1}. {fp}: {score:.4f}")
                
                if model_p < 0.05:
                    print("Best models:")
                    for i, (model, score) in enumerate(model_means.items()):
                        if i < 3:
                            print(f"  {i+1}. {model}: {score:.4f}")
                            
            except Exception as e:
                print(f"Error in ANOVA: {e}")
                results[task_type] = {'error': str(e)}
        
        return results
    
    def dataset_winners_analysis(self) -> Dict:
        """Find statistical winners for each dataset"""
        print("\n=== Dataset Winners Analysis ===")
        
        summary_df = self.get_summary_statistics()
        detailed_df = self.get_detailed_scores()
        
        winners = {}
        
        for dataset in summary_df['dataset'].unique():
            dataset_summary = summary_df[summary_df['dataset'] == dataset]
            dataset_detailed = detailed_df[detailed_df['dataset'] == dataset]
            
            is_classification = dataset_summary['is_classification'].iloc[0]
            
            # Find best method
            if is_classification:
                best_idx = dataset_summary['mean'].idxmax()
            else:
                best_idx = dataset_summary['mean'].idxmin()
            
            best_method = dataset_summary.loc[best_idx]
            best_fp = best_method['fingerprint']
            best_model = best_method['model']
            best_score = best_method['mean']
            
            # Get scores for statistical testing
            best_scores = dataset_detailed[
                (dataset_detailed['fingerprint'] == best_fp) & 
                (dataset_detailed['model'] == best_model)
            ]['score'].values
            
            # Test against second best
            dataset_summary_sorted = dataset_summary.sort_values('mean', ascending=not is_classification)
            if len(dataset_summary_sorted) > 1:
                second_best = dataset_summary_sorted.iloc[1]
                second_fp = second_best['fingerprint']
                second_model = second_best['model']
                
                second_scores = dataset_detailed[
                    (dataset_detailed['fingerprint'] == second_fp) & 
                    (dataset_detailed['model'] == second_model)
                ]['score'].values
                
                if len(best_scores) == len(second_scores) and len(best_scores) > 1:
                    _, p_value = ttest_rel(best_scores, second_scores)
                    significant = p_value < 0.05
                else:
                    p_value = np.nan
                    significant = False
            else:
                p_value = np.nan
                significant = False
            
            winners[dataset] = {
                'winner': f"{best_fp}+{best_model}",
                'score': best_score,
                'p_value': p_value,
                'significant': significant
            }
            
            sig_text = " (significant)" if significant else ""
            print(f"{dataset}: {best_fp}+{best_model} = {best_score:.4f}{sig_text}")
        
        return winners
    
    def run_statistical_analysis(self) -> Dict:
        """Run complete statistical analysis"""
        if not self.results:
            print("No results available. Run analyze_all_datasets() first.")
            return {}
        
        print("Running Statistical Analysis...")
        
        # Run all analyses
        self.statistical_results = {
            'dataset_winners': self.dataset_winners_analysis(),
            'friedman_ranking': self.friedman_ranking_analysis(), 
            'component_anova': self.component_anova_analysis()
        }
        
        return self.statistical_results
    
    def print_winners_summary(self):
        """Print concise summary of winners"""
        if not self.statistical_results:
            print("Run statistical analysis first!")
            return
            
        print("\n" + "="*50)
        print("WINNERS SUMMARY")
        print("="*50)
        
        # Overall winners from Friedman test
        friedman_results = self.statistical_results.get('friedman_ranking', {})
        for task_type in ['Classification', 'Regression']:
            if task_type in friedman_results and 'rankings' in friedman_results[task_type]:
                rankings = friedman_results[task_type]['rankings']
                if rankings:
                    winner = rankings[0][0]
                    fp, model = winner.split('_')
                    rank = rankings[0][1]
                    print(f"\nOverall {task_type} Winner: {fp} + {model} (avg rank: {rank:.2f})")
        
        # Best components from ANOVA
        anova_results = self.statistical_results.get('component_anova', {})
        for task_type in ['Classification', 'Regression']:
            if task_type in anova_results and 'fingerprint_means' in anova_results[task_type]:
                fp_means = anova_results[task_type]['fingerprint_means']
                model_means = anova_results[task_type]['model_means']
                
                if not fp_means.empty and not model_means.empty:
                    best_fp = fp_means.index[0]
                    best_model = model_means.index[0]
                    print(f"{task_type} - Best Fingerprint: {best_fp} ({fp_means.iloc[0]:.4f})")
                    print(f"{task_type} - Best Model: {best_model} ({model_means.iloc[0]:.4f})")
    
    def plot_detailed_comparison(self, figsize=(16, 10)):
        """Create detailed comparison plots grouped by model with fingerprint performance"""
        if not self.results:
            print("No results to plot yet. Please run analyze_all_datasets() first.")
            return
        
        # Get summary statistics
        df = self.get_summary_statistics()
        
        if df.empty:
            print("No valid results to plot.")
            return
        
        # Group datasets by type and sort alphabetically within each group
        classification_datasets = []
        regression_datasets = []
        
        for dataset in df['dataset'].unique():
            dataset_df = df[df['dataset'] == dataset]
            is_classification = dataset_df['is_classification'].iloc[0]
            
            if is_classification:
                classification_datasets.append(dataset)
            else:
                regression_datasets.append(dataset)
        
        # Sort alphabetically within each group
        classification_datasets = sorted(classification_datasets)
        regression_datasets = sorted(regression_datasets)
        
        # Combine: classification first, then regression
        datasets = classification_datasets + regression_datasets
        
        # Determine number of subplots needed
        n_datasets = len(datasets)
        n_cols = min(3, n_datasets)
        n_rows = (n_datasets + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        fig.suptitle(f'Performance by Dataset and Model (Mean ± Std across seeds)\nRun: {self.run_timestamp}',
                    fontsize=16, fontweight='bold')
        
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
            is_classification = dataset_df['is_classification'].iloc[0]
            
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
            model_stds = {}
            for model in models:
                model_data = dataset_df[dataset_df['model'] == model]
                model_averages[model] = model_data['mean'].mean()
                model_stds[model] = model_data['mean'].std()
            
            # Position models on x-axis
            model_positions = np.arange(n_models)
            
            # Create bars for each fingerprint within each model group
            for fp_idx, fingerprint in enumerate(fingerprints):
                fp_means = []
                fp_stds = []
                fp_positions = []
                
                for model_idx, model in enumerate(models):
                    subset = dataset_df[(dataset_df['fingerprint'] == fingerprint) & 
                                    (dataset_df['model'] == model)]
                    if len(subset) > 0:
                        mean_score = subset['mean'].iloc[0]
                        std_score = subset['std'].iloc[0]
                        fp_means.append(mean_score)
                        fp_stds.append(std_score if pd.notna(std_score) else 0)
                    else:
                        fp_means.append(0)
                        fp_stds.append(0)
                    
                    # Calculate position within model group
                    pos = model_positions[model_idx] + (fp_idx - (n_fingerprints-1)/2) * bar_width
                    fp_positions.append(pos)
                
                # Plot bars for this fingerprint across all models with error bars
                ax.bar(fp_positions, fp_means, bar_width * 0.9,
                    label=fingerprint, color=fingerprint_color_map[fingerprint],
                    alpha=0.8, edgecolor='white', linewidth=0.5,
                    yerr=fp_stds, capsize=3, error_kw={'alpha': 0.6})
            
            # Add model average lines/markers
            for model_idx, model in enumerate(models):
                avg_score = model_averages[model]
                # Draw a horizontal line across the model group showing average
                left_edge = model_positions[model_idx] - group_width/2
                right_edge = model_positions[model_idx] + group_width/2
                ax.hlines(avg_score, left_edge, right_edge,
                        colors='red', linestyles='--', linewidth=2, alpha=0.7)
                
                # Add average value as text
                y_offset = 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0])
                ax.text(model_positions[model_idx], avg_score + y_offset,
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
            
            # Set y-axis limits appropriately
            if is_classification:
                ax.set_ylim(0, 1)  # AUROC is bounded between 0 and 1
            else:
                ax.set_ylim(bottom=0)  # RMSE starts from 0
        
        # Hide unused subplots
        for idx in range(len(datasets), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.save_dir, f'detailed_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, filename: Optional[str] = None):
        """Save analysis results to CSV"""
        if filename is None:
            filename = f"analysis_results_{self.run_timestamp}.csv"
        
        df = self.get_summary_statistics()
        filepath = os.path.join(self.save_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Results saved to: {filepath}")
        return filepath
    
    def print_summary(self):
        """Print a summary of the analysis results"""
        if not self.results:
            print("No results available. Please run analyze_all_datasets() first.")
            return
        
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        
        df = self.get_summary_statistics()
        
        for dataset in sorted(self.results.keys()):
            dataset_results = self.results[dataset]
            if not dataset_results or not dataset_results['scores']:
                continue
                
            dataset_df = df[df['dataset'] == dataset]
            metric_name = dataset_results['metric']
            
            print(f"\nDataset: {dataset} ({metric_name})")
            print("-" * 40)
            
            # Best performing combinations
            if metric_name == "AUROC":
                best_row = dataset_df.loc[dataset_df['mean'].idxmax()]
                print(f"Best: {best_row['fingerprint']} + {best_row['model']} = {best_row['mean']:.4f} ± {best_row['std']:.4f}")
            else:  # RMSE - lower is better
                best_row = dataset_df.loc[dataset_df['mean'].idxmin()]
                print(f"Best: {best_row['fingerprint']} + {best_row['model']} = {best_row['mean']:.4f} ± {best_row['std']:.4f}")
            
            print(f"Number of seeds: {best_row['count']}")
            print(f"Total combinations tested: {len(dataset_df)}")

# Example usage:
def run_analysis(db_manager, save_dir="./analysis_results"):
    """Run complete analysis pipeline"""
    analyzer = BenchmarkManager(db_manager, save_dir)
    
    # Analyze all datasets
    analyzer.analyze_all_datasets()
    
    # Run statistical analysis
    analyzer.run_statistical_analysis()
    
    # Print winners summary
    analyzer.print_winners_summary()
    
    # Create visualization
    analyzer.plot_detailed_comparison()
    
    # Save results
    #analyzer.save_results()
    
    # Print summary
    analyzer.print_summary()
    
    return analyzer

if __name__ == '__main__':
    master_job_id = sys.argv[1]
    base = Path('./studies')
    path = base / master_job_id
    directories = [item.name for item in path.iterdir() if item.is_dir()]
    for dic in directories:
        subdir_path = path / dic
        string = f"{subdir_path}/predictions.db"
        db_manager = DatabaseManager(string)
        analyzer = run_analysis(db_manager, str(subdir_path))