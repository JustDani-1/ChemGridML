# experiments.py
from dataclasses import dataclass
from typing import List, Dict
from itertools import product

@dataclass
class Method:
    feature: str
    model: str
    dataset: str

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
class Group:
    """A named group of methods with shared resources"""
    name: str
    methods: List[Method]
    resources: Resources
    lower: int          # lower bound for task id
    upper: int          # upper bound for task id
    
    def __len__(self):
        return len(self.methods)


@dataclass
class Experiment:
    """
    Encapsulates a complete experiment including groups of methods and cluster resources
    """
    name: str
    groups: List[Group]

    def __post_init__(self):
        """Automatically calculate array job bounds after initialization"""
        self._calculate_array_bounds()

    def _calculate_array_bounds(self):
        """Calculate array job bounds for each group"""
        current_start = 1
        
        for group in self.groups:
            group_size = len(group.methods)
            current_end = current_start + group_size - 1
            
            group.lower = current_start
            group.upper = current_end
            
            current_start = current_end + 1

    def get_method(self, task_id: int) -> Method:
        """Get method for a given task_id"""
        for group in self.groups:
            if group.lower <= task_id <= group.upper:
                # Convert global task_id to local index within the group
                local = task_id - group.lower
                return group.methods[local]
    
    def total_methods(self) -> int:
        """Get total amount of methods over all groups"""
        return self.groups[-1].upper
    

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

        methods = self._create_by_product(features, models, datasets)

        self.experiments["FINGERPRINT"] = Experiment(
            name="FINGERPRINT",
            groups=[
                Group(
                    name="FINGERPRINT",
                    methods=methods,
                    resources=Resources(wall_time="10:00:0", memory=4, cores=5, gpu=False)
                )
            ]
        )

        # LEARNABLE experiment - end-to-end ML 
        features = ['GRAPH']
        models = ['GCN', 'GAT']
        datasets = ['Caco2_Wang', 'PPBR_AZ', 'Lipophilicity_AstraZeneca', 'BBB_Martins', 'PAMPA_NCATS', 'Pgp_Broccatelli']

        methods = self._create_by_product(features, models, datasets)
        
        self.experiments["LEARNABLE"] = Experiment(
            name="LEARNABLE",
            groups=[
                Group(
                    name="LEARNABLE",
                    methods=methods,
                    resources=Resources(wall_time="12:00:0", memory=16, cores=10, gpu=False)
                )
            ]
        )
    
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