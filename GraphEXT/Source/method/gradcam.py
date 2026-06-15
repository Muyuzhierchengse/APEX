from typing import Any, Callable, List, Tuple, Union, Dict

import captum.attr as ca
import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr._utils.common import (
    _format_additional_forward_args,
    _format_attributions,
    _format_input,
)
from captum.attr._utils.gradient import (
    apply_gradient_requirements,
    compute_layer_gradients_and_eval,
    undo_gradient_requirements,
)
from captum.attr._utils.typing import (
    TargetType,
)
from torch import Tensor
from torch.nn import Module
from torch_geometric.utils.loop import add_remaining_self_loops

from dig.xgraph.method.base_explainer import WalkBase
from dig.xgraph.models.utils import subgraph, normalize

EPS = 1e-15


class GradCAM(WalkBase):
    def __init__(self, model: nn.Module, explain_graph: bool = False):
        super().__init__(model, explain_graph=explain_graph)
        self.last_node_scores = None

    def forward(self, x: Tensor, edge_index: Tensor, **kwargs):
        self.model.eval()
        super().forward(x, edge_index)

        num_classes = kwargs.get('num_classes')
        labels = tuple(i for i in range(num_classes))
        ex_labels = tuple(torch.tensor([label]).to(self.device) for label in labels)

        self_loop_edge_index, _ = add_remaining_self_loops(
            edge_index, num_nodes=self.num_nodes
        )

        # ── 节点分类分支 ──────────────────────────────────────────────────────
        if not self.explain_graph:
            node_idx = kwargs.get('node_idx')
            if isinstance(node_idx, int):
                node_idx = torch.tensor([node_idx], device=self.device, dtype=torch.long)
            if not node_idx.dim():
                node_idx = node_idx.reshape(-1)
            node_idx = node_idx.to(self.device)
            assert node_idx is not None

            self.subset, _, _, self.hard_edge_mask = subgraph(
                node_idx, self.__num_hops__, self_loop_edge_index,
                relabel_nodes=True, num_nodes=None, flow=self.__flow__()
            )
            self.new_node_idx = torch.where(self.subset == node_idx)[0]

            # model_node 在全图上前向，只取目标节点的输出
            # additional_forward_args 传全图 edge_index
            class model_node(nn.Module):
                def __init__(self, cls):
                    super().__init__()
                    self.cls = cls
                    self.convs = cls.model.convs

                def forward(self, *args, **kwargs):
                    return self.cls.model(*args, **kwargs)[node_idx]

            model = model_node(self)
            self.explain_method = GraphLayerGradCam(model, model.convs[-1])

            num_nodes   = x.size(0)
            edge_masks      = []
            node_masks_list = []
            raw_node_scores = []

            for ex_label in ex_labels:
                # attr shape: [num_nodes, 1] 或 [num_nodes]
                # 因为 model_node 在全图 x 上运行，GradCAM 对全图所有节点求梯度
                attr = self.explain_method.attribute(
                    x, ex_label, additional_forward_args=edge_index
                ).detach()

                # ── 归一化：得到全图节点分数 [num_nodes] ────────────────────
                node_scores = normalize(attr.relu()).squeeze()  # [num_nodes]

                if node_scores.dim() == 0:
                    # 极端情况：标量，扩展为全图大小
                    node_scores = node_scores.unsqueeze(0).expand(num_nodes)
                elif node_scores.shape[0] != num_nodes:
                    # attr 输出与全图节点数不符时，用零向量兜底
                    # （正常情况不应进入此分支）
                    scores_raw = node_scores
                    node_scores = torch.zeros(num_nodes, device=x.device)
                    copy_len = min(scores_raw.shape[0], num_nodes)
                    node_scores[:copy_len] = scores_raw[:copy_len]

                raw_node_scores.append(node_scores)

                # ── edge_mask：在全图 self_loop_edge_index 上聚合 ────────────
                edge_score = (
                    node_scores[self_loop_edge_index[0]] +
                    node_scores[self_loop_edge_index[1]]
                ) / 2
                n_orig = edge_index.shape[1]
                edge_masks.append(edge_score[:n_orig].detach())

                # ── node_mask：全图 top-k ────────────────────────────────────
                max_nodes = kwargs.get('max_nodes', max(1, int(num_nodes * 0.5)))
                k = min(max_nodes, num_nodes)
                topk_indices = node_scores.topk(k).indices
                node_mask = torch.zeros(num_nodes, device=x.device)
                node_mask[topk_indices] = 1.0
                node_masks_list.append(node_mask)

            self.last_node_scores = raw_node_scores
            return edge_masks, node_masks_list

        # ── 图分类分支（完全不动）────────────────────────────────────────────
        else:
            model = self.model
            self.explain_method = GraphLayerGradCam(model, model.convs[-1])

            num_nodes   = x.size(0)
            edge_masks      = []
            node_masks_list = []
            raw_node_scores = []

            for ex_label in ex_labels:
                attr = self.explain_method.attribute(
                    x, ex_label, additional_forward_args=edge_index
                ).detach()

                node_scores = normalize(attr.relu()).squeeze()

                if node_scores.dim() == 0:
                    node_scores = node_scores.unsqueeze(0).expand(num_nodes)
                if node_scores.shape[0] != num_nodes:
                    node_scores = node_scores[:num_nodes]

                raw_node_scores.append(node_scores)

                edge_score = (
                    node_scores[self_loop_edge_index[0]] +
                    node_scores[self_loop_edge_index[1]]
                ) / 2
                n_orig = edge_index.shape[1]
                edge_masks.append(edge_score[:n_orig].detach())

                max_nodes = kwargs.get('max_nodes', max(1, int(num_nodes * 0.5)))
                k = min(max_nodes, num_nodes)
                topk_indices = node_scores.topk(k).indices
                node_mask = torch.zeros(num_nodes, device=x.device)
                node_mask[topk_indices] = 1.0
                node_masks_list.append(node_mask)

            self.last_node_scores = raw_node_scores
            return edge_masks, node_masks_list


class GraphLayerGradCam(ca.LayerGradCam):

    def __init__(
            self,
            forward_func: Callable,
            layer: Module,
            device_ids: Union[None, List[int]] = None,
    ) -> None:
        super().__init__(forward_func, layer, device_ids)

    def attribute(
            self,
            inputs: Union[Tensor, Tuple[Tensor, ...]],
            target: TargetType = None,
            additional_forward_args: Any = None,
            attribute_to_layer_input: bool = False,
            relu_attributions: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        inputs = _format_input(inputs)
        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )
        gradient_mask = apply_gradient_requirements(inputs)
        layer_gradients, layer_evals, is_layer_tuple = compute_layer_gradients_and_eval(
            self.forward_func,
            self.layer,
            inputs,
            target,
            additional_forward_args,
            device_ids=self.device_ids,
            attribute_to_layer_input=attribute_to_layer_input,
        )
        undo_gradient_requirements(inputs, gradient_mask)

        layer_gradients = tuple(layer_grad.transpose(0, 1).unsqueeze(0)
                                for layer_grad in layer_gradients)
        layer_evals = tuple(layer_eval.transpose(0, 1).unsqueeze(0)
                            for layer_eval in layer_evals)

        summed_grads = tuple(
            torch.mean(
                layer_grad,
                dim=tuple(x for x in range(2, len(layer_grad.shape))),
                keepdim=True,
            )
            for layer_grad in layer_gradients
        )

        scaled_acts = tuple(
            torch.sum(summed_grad * layer_eval, dim=1, keepdim=True)
            for summed_grad, layer_eval in zip(summed_grads, layer_evals)
        )

        if relu_attributions:
            scaled_acts = tuple(F.relu(scaled_act) for scaled_act in scaled_acts)

        scaled_acts = tuple(scaled_act.squeeze(0).transpose(0, 1)
                            for scaled_act in scaled_acts)

        return _format_attributions(is_layer_tuple, scaled_acts)