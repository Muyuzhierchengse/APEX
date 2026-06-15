
import torch
import torch.nn as nn
import numpy as np
import copy
from typing import Dict, List, Tuple, Optional


class FSX:
    """
    FSX (Message Flow Sensitivity Enhanced Structural Explainer)
    
    A hybrid framework combining internal message flow analysis with 
    cooperative game theory for GNN explainability.
    """
    
    def __init__(self, model: nn.Module, explain_graph: bool = True):
        """
        Initialize FSX explainer.
        
        Args:
            model: The GNN model to explain (assumed to be trained)
            explain_graph: Whether explaining graph-level (True) or node-level (False) predictions
        """
        self.model = model
        self.model.eval()
        self.explain_graph = explain_graph
        
    def __call__(self, 
                 x: torch.Tensor, 
                 edge_index: torch.Tensor,
                 sparsity: float = 0.5,
                 num_classes: int = 2,
                 node_idx: int = 0,
                 max_nodes: Optional[int] = None,
                 **kwargs) -> Dict:
        """
        Main explanation interface compatible with existing explainers.
        
        Args:
            x: Node feature matrix [num_nodes, num_features]
            edge_index: Edge index [2, num_edges]
            sparsity: Target sparsity level (not used directly in FSX)
            num_classes: Number of prediction classes
            node_idx: Target node index (for node-level explanation)
            max_nodes: Maximum nodes to keep (derived from sparsity in main.py)
            **kwargs: Additional arguments
            
        Returns:
            masks: Dictionary with explanation masks for each class
                   Format: {class_idx: edge_mask or node_mask}
        """
        # Step 1: Get original prediction
        y_original, prediction = self._get_original_prediction(x, edge_index, num_classes)
        
        # Step 2: Compute message flow sensitivity via perturbation
        message_flow_sensitivity = self._compute_message_flow_sensitivity(
            x, edge_index, y_original, prediction, num_classes
        )
        
        # Step 3: Identify key subgraph based on flow sensitivity
        key_subgraph = self._identify_key_subgraph(
            message_flow_sensitivity, x, edge_index, max_nodes
        )
        
        # Step 4: Compute flow-aware weighted Shapley values
        node_contributions = self._compute_weighted_shapley(
            x, edge_index, key_subgraph, y_original, prediction, num_classes
        )
        
        # Step 5: Convert node contributions to masks compatible with existing evaluation
        masks = self._convert_to_masks(
            node_contributions, edge_index, num_classes
        )
        
        return masks
    
    def _get_original_prediction(self, 
                                  x: torch.Tensor, 
                                  edge_index: torch.Tensor,
                                  num_classes: int) -> Tuple[torch.Tensor, int]:
        """
        Compute original prediction for the input graph.
        
        Args:
            x: Node features
            edge_index: Edge index
            num_classes: Number of classes
            
        Returns:
            y_original: Original prediction logits/probabilities [num_classes]
            prediction: Predicted class index
        """
        with torch.no_grad():
            output = self.model(x, edge_index)
            if self.explain_graph:
                y_original = output[0]  # Graph-level: take first (only) output
            else:
                y_original = output[0]  # Node-level: take target node          
            prediction = y_original.argmax(-1).item()
        
        return y_original, prediction
    
    def _compute_message_flow_sensitivity(self,
                                          x: torch.Tensor,
                                          edge_index: torch.Tensor,
                                          y_original: torch.Tensor,
                                          target_class: int,
                                          num_classes: int) -> Dict[Tuple[int, int, int], float]:
        """
        Core perturbation-based message flow sensitivity analysis.
        
        For each edge (u,v) and each layer l, compute sensitivity S(u,v,l)
        by perturbing the message flow from u to v at layer l.
        
        Args:
            x: Node features
            edge_index: Edge index [2, num_edges]
            y_original: Original prediction output
            target_class: Target class to explain
            num_classes: Number of classes
            
        Returns:
            sensitivity_dict: Dictionary mapping (u, v, l) -> sensitivity_score
                            where sensitivity = |Y_original - Y_perturbed|
        """
        sensitivity_dict = {}
        num_layers = self._get_num_layers()
        num_edges = edge_index.size(1)
        
        # Get original prediction value for target class
        y_orig_value = y_original[target_class].item()
        
        # For each layer
        for layer_idx in range(num_layers):
            # For each edge
            for edge_idx in range(num_edges):
                u = edge_index[0, edge_idx].item()
                v = edge_index[1, edge_idx].item()
                
                # Perturb this specific edge at this specific layer
                y_perturbed = self._perturb_edge_at_layer(
                    x, edge_index, u, v, layer_idx, target_class
                )
                
                # Compute sensitivity as absolute difference
                sensitivity = abs(y_orig_value - y_perturbed)
                sensitivity_dict[(u, v, layer_idx)] = sensitivity
        
        return sensitivity_dict
    
    def _perturb_edge_at_layer(self,
                            x: torch.Tensor,
                            edge_index: torch.Tensor,
                            u: int,
                            v: int,
                            layer_idx: int,
                            target_class: int) -> float:
        """
        精确拦截并扰动第layer_idx层中从节点u传递到节点v的消息h_u^(layer_idx-1)
        
        实现策略：
        在目标层的消息传递中，临时移除边(u,v)，从而精确阻断h_u^(l-1)到节点v的传递
        
        Args:
            x: Node features
            edge_index: Edge index [2, num_edges]
            u, v: Edge endpoints (message from u to v)
            layer_idx: Target layer index (0-indexed, 0表示第一层)
            target_class: Target class to explain
            
        Returns:
            y_perturbed: Prediction value after perturbation
        """
        # 找到要移除的边(u,v)的索引
        edge_to_remove = None
        for i in range(edge_index.size(1)):
            if edge_index[0, i].item() == u and edge_index[1, i].item() == v:
                edge_to_remove = i
                break
        
        # 如果边不存在，返回原始预测（无扰动）
        if edge_to_remove is None:
            with torch.no_grad():
                output = self.model(x, edge_index)
                if self.explain_graph:
                    return output[0][target_class].item()
                else:
                    return output[0][target_class].item()
        
        # 创建移除该边后的edge_index
        edge_mask = torch.ones(edge_index.size(1), dtype=torch.bool, device=edge_index.device)
        edge_mask[edge_to_remove] = False
        edge_index_perturbed = edge_index[:, edge_mask]
        
        # 使用hook机制在特定层使用扰动后的edge_index
        original_edge_index = edge_index.clone()
        
        def hook_fn(module, input):
            """
            前向hook：在目标层的输入阶段替换edge_index
            GINConv.forward的输入是(x, edge_index, ...)
            """
            # 修改输入中的edge_index
            if isinstance(input, tuple) and len(input) >= 2:
                modified_input = list(input)
                # input[1]是edge_index
                modified_input[1] = edge_index_perturbed
                return tuple(modified_input)
            return input
        
        # 注册hook到目标层
        hooks = []
        layer_count = 0
        
        # 遍历模型的GIN层，找到目标层并注册hook
        for name, module in self.model.named_modules():
            if isinstance(module, type(self.model.conv1)):  # GINConv类型
                if layer_count == layer_idx:
                    # 在目标层注册hook
                    hook = module.register_forward_pre_hook(hook_fn)
                    hooks.append(hook)
                    break  # 找到目标层后立即退出，只注册一次
                layer_count += 1
        
        # 执行扰动后的前向传播
        with torch.no_grad():
            output = self.model(x, original_edge_index)
            if self.explain_graph:
                y_perturbed = output[0][target_class].item()
            else:
                y_perturbed = output[0][target_class].item()
        
        # 移除所有hooks
        for hook in hooks:
            hook.remove()
        
        return y_perturbed
    
    def _identify_key_subgraph(self,
                               sensitivity_dict: Dict[Tuple[int, int, int], float],
                               x: torch.Tensor,
                               edge_index: torch.Tensor,
                               max_nodes: Optional[int] = None) -> Dict:
        """
        Identify key subgraph based on message flow sensitivity.
        
        Args:
            sensitivity_dict: Message flow sensitivity scores {(u,v,l): score}
            x: Node features
            edge_index: Edge index
            max_nodes: Maximum number of nodes/edges to keep
            
        Returns:
            key_subgraph: Dictionary containing key subgraph information
        """
        # Step 1: Aggregate sensitivity across layers: W(u,v) = sum_l S(u,v,l)
        edge_weights = {}
        for (u, v, l), sensitivity in sensitivity_dict.items():
            edge_key = (u, v)
            if edge_key not in edge_weights:
                edge_weights[edge_key] = 0.0
            edge_weights[edge_key] += sensitivity
        
        # Step 2: Sort edges by aggregated importance
        sorted_edges = sorted(edge_weights.items(), key=lambda x: x[1], reverse=True)
        
        # Step 3: Select top edges based on max_nodes
        if max_nodes is not None:
            # Select top edges
            num_edges_to_keep = min(max_nodes, len(sorted_edges))
        else:
            # Keep top 20% of edges by default
            num_edges_to_keep = max(1, int(0.2* len(sorted_edges))) 
        
        top_edges = sorted_edges[:num_edges_to_keep]
        
        # Step 4: Extract nodes involved in selected edges
        key_nodes = set()
        key_edges = []
        key_edge_weights = {}
        
        for (u, v), weight in top_edges:
            key_nodes.add(u)
            key_nodes.add(v)
            key_edges.append((u, v))
            key_edge_weights[(u, v)] = weight
        
        return {
            'nodes': list(key_nodes),
            'edges': key_edges,
            'edge_weights': key_edge_weights,
            'sensitivity_dict': sensitivity_dict
        }
    
    def _compute_weighted_shapley(self,
                                  x: torch.Tensor,
                                  edge_index: torch.Tensor,
                                  key_subgraph: Dict,
                                  y_original: torch.Tensor,
                                  target_class: int,
                                  num_classes: int,
                                  num_samples: int = 100) -> Dict[int, float]:
        """
        Compute flow-aware weighted Shapley values for nodes in key subgraph.
        
        Args:
            x: Node features
            edge_index: Edge index
            key_subgraph: Key subgraph from _identify_key_subgraph
            y_original: Original prediction
            target_class: Target class
            num_classes: Number of classes
            num_samples: Number of Monte Carlo samples
            
        Returns:
            node_contributions: Dictionary mapping node_idx -> weighted_shapley_value
        """
        key_nodes = key_subgraph['nodes']
        edge_weights = key_subgraph['edge_weights']
        
        # Initialize
        shapley_values_weighted = {node: 0.0 for node in key_nodes}
        sum_of_weights = {node: 0.0 for node in key_nodes}
        
        # Compute x0 for sigmoid (mean of all edge weights)
        all_weights = list(edge_weights.values())
        x0 = np.mean(all_weights) if all_weights else 0.0   #mean
        temperature = 1.0
        
        # Monte Carlo sampling
        for _ in range(num_samples):
            # Generate random permutation
            permutation = np.random.permutation(key_nodes).tolist()
            
            # For each node in permutation
            for idx, node in enumerate(permutation):
                # Construct coalitions
                coalition_before = permutation[:idx]
                coalition_current = permutation[:idx+1]
                
                # Compute coalition values
                v_before = self._compute_coalition_value(
                    x, edge_index, coalition_before, target_class
                )
                v_current = self._compute_coalition_value(
                    x, edge_index, coalition_current, target_class
                )
                
                # Marginal contribution
                mc = v_current - v_before
                
                # Compute coalition importance
                importance = self._compute_coalition_importance(
                    coalition_current, edge_index, edge_weights
                )
                
                # Compute weight using sigmoid
                weight = self._sigmoid_weighting(importance, x0, temperature)
                
                # Update weighted contributions
                shapley_values_weighted[node] += weight * mc
                sum_of_weights[node] += weight
        
        # Normalize by sum of weights
        node_contributions = {}
        for node in key_nodes:
            if sum_of_weights[node] > 0:
                node_contributions[node] = shapley_values_weighted[node] / sum_of_weights[node]
            else:
                node_contributions[node] = 0.0
        
        return node_contributions
    
    def _compute_coalition_value(self,
                                 x: torch.Tensor,
                                 edge_index: torch.Tensor,
                                 coalition_nodes: List[int],
                                 target_class: int) -> float:
        """
        Compute the value function v(S) for a coalition of nodes.
        
        Args:
            x: Node features
            edge_index: Edge index
            coalition_nodes: List of node indices in the coalition
            target_class: Target class to evaluate
            
        Returns:
            coalition_value: Model prediction for the coalition
        """
        if len(coalition_nodes) == 0:
            return 0.0
        
        # Create masked graph
        x_masked = x.clone()
        coalition_set = set(coalition_nodes)
        
        # Zero out nodes not in coalition
        num_nodes = x.size(0)
        for node in range(num_nodes):
            if node not in coalition_set:
                x_masked[node] = 0.0
        
        # Filter edges to only include those within coalition
        edge_mask = torch.zeros(edge_index.size(1), dtype=torch.bool)
        for i in range(edge_index.size(1)):
            u = edge_index[0, i].item()
            v = edge_index[1, i].item()
            if u in coalition_set and v in coalition_set:
                edge_mask[i] = True
        
        edge_index_masked = edge_index[:, edge_mask]
        
        # Handle empty edge case
        if edge_index_masked.size(1) == 0:
            return 0.0
        
        # Run model
        with torch.no_grad():
            output = self.model(x_masked, edge_index_masked)
            if self.explain_graph:
                value = output[0][target_class].item()
            else:
                value = output[0][target_class].item()
        
        return value
    
    def _compute_coalition_importance(self,
                                      coalition_nodes: List[int],
                                      edge_index: torch.Tensor,
                                      edge_weights: Dict[Tuple[int, int], float]) -> float:
        """
        Compute information flow importance I(G_S) for a coalition.
        
        Args:
            coalition_nodes: Nodes in the coalition
            edge_index: Edge index
            edge_weights: Edge weights W(u,v) from aggregated sensitivity
            
        Returns:
            importance: Sum of edge weights for edges within the coalition
        """
        coalition_set = set(coalition_nodes)
        importance = 0.0
        
        for (u, v), weight in edge_weights.items():
            if u in coalition_set and v in coalition_set:
                importance += weight
        
        return importance
    
    def _sigmoid_weighting(self,
                          importance_score: float,
                          x0: float = 0.0,
                          temperature: float = 1.0) -> float:
        """
        Apply sigmoid weighting function α(x).
        
        Args:
            importance_score: Coalition importance I(G_S)
            x0: Center point of sigmoid
            temperature: Temperature parameter T
            
        Returns:
            weight: Sigmoid weight in [0, 1]
        """
        return 1.0 / (1.0 + np.exp(-(importance_score - x0) / temperature))
    
    def _convert_to_masks(self,
                         node_contributions: Dict[int, float],
                         edge_index: torch.Tensor,
                         num_classes: int) -> Dict[int, torch.Tensor]:
        """
        Convert node contribution scores to edge/node masks for evaluation.
        
        Args:
            node_contributions: Node-level Shapley values
            edge_index: Edge index [2, num_edges]
            num_classes: Number of classes
            
        Returns:
            masks: Dictionary mapping class_idx -> mask
        """
        num_edges = edge_index.size(1)
        edge_mask = torch.zeros(num_edges)
        
        # Convert node contributions to edge scores
        # Edge score = average of endpoint node contributions
        for i in range(num_edges):
            u = edge_index[0, i].item()
            v = edge_index[1, i].item()
            
            u_contrib = node_contributions.get(u, 0.0)
            v_contrib = node_contributions.get(v, 0.0)
            edge_mask[i] = (u_contrib + v_contrib) / 2.0
        
        # Normalize to [0, 1]
        if edge_mask.max() > 0:
            edge_mask = edge_mask / edge_mask.max()
        
        # Return masks for all classes (same mask for all)
        masks = {i: edge_mask for i in range(num_classes)}
        
        return masks
    
    def _get_num_layers(self) -> int:
        """
        Get number of GNN layers in the model.
        
        Returns:
            num_layers: Number of message passing layers
        """
        num_layers = 0
        for name, module in self.model.named_modules():
            if 'conv' in name.lower() or 'gnn' in name.lower():
                num_layers += 1
        
        # Default to 3 if cannot detect
        return num_layers if num_layers > 0 else 3
