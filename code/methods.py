# method_registry.py
from dataclasses import dataclass
from typing import List, Dict, Set
from enum import Enum

class InputType(Enum):
    FINGERPRINT = "fingerprint"
    GRAPH = "graph"
    # Add more as needed

@dataclass
class Method:
    """Represents a complete method (input representation + model)"""
    name: str
    input_type: InputType
    input_representation: str  # e.g., 'ECFP', 'GRAPH', etc.
    model: str  # e.g., 'FNN', 'GCN', etc.

class MethodRegistry:
    """Registry for all valid method combinations"""
    
    def __init__(self):
        self.methods: List[Method] = []
        self._setup_methods()
    
    def _setup_methods(self):
        """Define all valid method combinations"""
        
        # Fingerprint-based methods
        fingerprints = ['ECFP', 'AtomPair', 'MACCS', 'RDKitFP', 'TOPOTOR', 'MOL2VEC']
        traditional_models = ['FNN', 'RF', 'XGBoost', 'SVM', 'ElasticNet', 'KNN']
        
        for fp in fingerprints:
            for model in traditional_models:
                self.methods.append(Method(
                    name=f"{fp}_{model}",
                    input_type=InputType.FINGERPRINT,
                    input_representation=fp,
                    model=model,
                ))
        
        # Graph-based methods
        graph_models = ['GCN', 'GAT']
        for model in graph_models:
            self.methods.append(Method(
                name=f"GRAPH_{model}",
                input_type=InputType.GRAPH,
                input_representation='GRAPH',
                model=model,
            ))
        
        # Future: Add pre-trained methods
        # pretrained_models = ['MOLBERT', 'GROVER']
        # for model in pretrained_models:
        #     self.methods.append(Method(
        #         name=f"PRETRAINED_{model}",
        #         input_type=InputType.SEQUENCE,
        #         input_representation='SMILES',
        #         model=model,
        #     ))
    
    def get_method(self, index: int) -> Method:
        """Get method by index for array job indexing"""
        return self.methods[index]
    
    def get_methods_by_input_type(self, input_type: InputType) -> List[Method]:
        """Get methods filtered by input type"""
        return [m for m in self.methods if m.input_type == input_type]
    
    def get_methods_by_model(self, model: str) -> List[Method]:
        """Get methods filtered by model type"""
        return [m for m in self.methods if m.model == model]
    
    def total_methods(self) -> int:
        """Get total number of methods"""
        return len(self.methods)