# experiments.py
from dataclasses import dataclass
from typing import List, Dict
from itertools import product

@dataclass
class Method:
    feature: str
    model: int
    dataset: int

    def __str__(self):
        return f"{self.feature}_{self.model}_{self.dataset}"

@dataclass
class Resources:
    """Cluster resource requirements"""
    wall_time: str  # e.g., "10:00:0" for 10 hours
    memory: int     # e.g., "8" for 8GB
    cores: int      # number of CPU cores
    gpu: bool       # wether to request a GPU


@dataclass
class Experiment:
    """
    Encapsulates a complete experiment including methods and cluster resources
    """
    name: str
    methods: List[Method]
    resources: Resources
    

class ExperimentRegistry:
    """Registry for all experiment configurations"""
    
    def __init__(self):
        self.experiments: Dict[str, Experiment] = {}
        self._setup_experiments()

    def _create_by_product(self, features: List[str], models: List[str], datasets: List[str]) -> List[Method]:
        """Create all combinations of features, models, and data as Method objects"""
        return [Method(feature=f, model=m, dataset=d) 
                for f, m, d in product(features, models, datasets)]
    
    def _setup_experiments(self):
        """Define all experiment configurations"""
        
        # FINGERPRINT experiment - Traditional ML methods with fingerprints
        features = ['ECFP', 'AtomPair', 'MACCS', 'RDKitFP', 'TOPOTOR', 'MOL2VEC']
        models = ['FNN', 'RF', 'XGBoost', 'SVM', 'ElasticNet', 'KNN']
        datasets = ['Caco2_Wang', 'PPBR_AZ', 'Lipophilicity_AstraZeneca', 'BBB_Martins', 'PAMPA_NCATS', 'Pgp_Broccatelli']

        self.experiments["FINGERPRINT"] = Experiment(
            name="FINGERPRINT",
            methods=self._create_by_product(features, models, datasets),
            resources=Resources(
                wall_time="10:00:0",    
                memory=4,              
                cores=5,                
                gpu=False
            )
        )

        features = ['GRAPH']
        models = ['GCN', 'GAT']
        datasets = ['Caco2_Wang', 'PPBR_AZ', 'Lipophilicity_AstraZeneca', 'BBB_Martins', 'PAMPA_NCATS', 'Pgp_Broccatelli']
        
        # LEARNABLE experiment - end-to-end ML 
        self.experiments["LEARNABLE"] = Experiment(
            name="LEARNABLE",
            methods=self._create_by_product(features, models, datasets),
            resources=Resources(
                wall_time="12:00:0",    
                memory=16,           
                cores=10,   
                gpu=False            
            )
        )
        
        # FAST experiment - Quick methods for testing
        # fast_methods = [m for m in self.method_registry.methods 
        #                if m.model in ['RF', 'FNN'] and m.input_representation in ['ECFP', 'GRAPH']]
        # self.experiments["FAST"] = Experiment(
        #     name="FAST",
        #     methods=fast_methods,
        #     resources=Resources(
        #         wall_time="2:00:0",   # 2 hours
        #         memory="4G",          # 4GB RAM
        #         cores=2,              # 2 CPU cores
        #         job_name="FAST"
        #     )
        # )
        
        # ENSEMBLE experiment - Methods suitable for ensembling
        # ensemble_models = ['RF', 'XGBoost', 'FNN', 'GCN']
        # ensemble_methods = [m for m in self.method_registry.methods if m.model in ensemble_models]
        # self.experiments["ENSEMBLE"] = Experiment(
        #     name="ENSEMBLE",
        #     methods=ensemble_methods,
        #     resources=Resources(
        #         wall_time="20:00:0",  # 20 hours (long for ensemble training)
        #         memory="24G",         # 24GB RAM
        #         cores=12,             # 12 CPU cores
        #         job_name="ENS"
        #     )
        # )
    
    def get_experiment(self, name: str) -> Experiment:
        """Get experiment by name"""
        if name not in self.experiments:
            available = list(self.experiments.keys())
            raise ValueError(f"Experiment '{name}' not found. Available: {available}")
        return self.experiments[name]
    
    def list_experiments(self) -> List[str]:
        """List all available experiment names"""
        return list(self.experiments.keys())
    
    def add_custom_experiment(self, experiment: Experiment):
        """Add a custom experiment configuration"""
        self.experiments[experiment.name] = experiment
