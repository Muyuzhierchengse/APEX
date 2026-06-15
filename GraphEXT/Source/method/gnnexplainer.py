import torch
from torch import Tensor
from torch_geometric.utils.loop import add_remaining_self_loops
from dig.version import debug
from dig.xgraph.models.utils import subgraph
from dig.xgraph.method.utils import symmetric_edge_mask_indirect_graph
from torch.nn.functional import cross_entropy
from dig.xgraph.method.base_explainer import ExplainerBase
from typing import Union
EPS = 1e-15


def cross_entropy_with_logit(y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs):
    return cross_entropy(y_pred, y_true.long(), **kwargs)


class GNNExplainer(ExplainerBase):
    def __init__(self,
                 model: torch.nn.Module,
                 epochs: int = 100,
                 lr: float = 0.01,
                 coff_edge_size: float = 0.001,
                 coff_edge_ent: float = 0.001,
                 coff_node_feat_size: float = 1.0,
                 coff_node_feat_ent: float = 0.1,
                 explain_graph: bool = False,
                 indirect_graph_symmetric_weights: bool = False):
        super(GNNExplainer, self).__init__(model, epochs, lr, explain_graph)
        self.coff_node_feat_size = coff_node_feat_size
        self.coff_node_feat_ent = coff_node_feat_ent
        self.coff_edge_size = coff_edge_size
        self.coff_edge_ent = coff_edge_ent
        self._symmetric_edge_mask_indirect_graph: bool = indirect_graph_symmetric_weights
        self.last_node_scores = None   # 新增：节点分类时存储节点分数

    def __loss__(self, raw_preds: Tensor, x_label: Union[Tensor, int]):
        if self.explain_graph:
            loss = cross_entropy_with_logit(raw_preds, x_label)
        else:
            loss = cross_entropy_with_logit(raw_preds[self.node_idx].reshape(1, -1), x_label)

        m = self.edge_mask.sigmoid()
        loss = loss + self.coff_edge_size * m.sum()
        ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
        loss = loss + self.coff_edge_ent * ent.mean()

        if self.mask_features:
            m = self.node_feat_mask.sigmoid()
            loss = loss + self.coff_node_feat_size * m.sum()
            ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
            loss = loss + self.coff_node_feat_ent * ent.mean()

        return loss

    def gnn_explainer_alg(self,
                          x: Tensor,
                          edge_index: Tensor,
                          ex_label: Tensor,
                          mask_features: bool = False,
                          **kwargs) -> Tensor:
        self.to(x.device)
        self.mask_features = mask_features

        optimizer = torch.optim.Adam([self.node_feat_mask, self.edge_mask], lr=self.lr)

        for epoch in range(1, self.epochs + 1):
            if mask_features:
                h = x * self.node_feat_mask.view(1, -1).sigmoid()
            else:
                h = x
            raw_preds = self.model(x=h, edge_index=edge_index, **kwargs)
            loss = self.__loss__(raw_preds, ex_label)
            if epoch % 20 == 0 and debug:
                print(f'Loss:{loss.item()}')

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=2.0)
            optimizer.step()

        return self.edge_mask.data

    def forward(self, x, edge_index, mask_features=False, target_label=None, **kwargs):
        super().forward(x=x, edge_index=edge_index, **kwargs)
        self.model.eval()

        self_loop_edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=self.num_nodes)

        # ── 图分类分支（完全不动）────────────────────────────────────────────
        if self.explain_graph:
            labels    = tuple(i for i in range(kwargs.get('num_classes')))
            ex_labels = tuple(torch.tensor([label]).to(self.device) for label in labels)

            edge_masks = []
            for ex_label in ex_labels:
                if target_label is None or ex_label.item() == target_label.item():
                    self.__clear_masks__()
                    self.__set_masks__(x, self_loop_edge_index)
                    edge_mask = self.gnn_explainer_alg(
                        x, self_loop_edge_index, ex_label
                    ).sigmoid()

                    if self._symmetric_edge_mask_indirect_graph:
                        edge_mask = symmetric_edge_mask_indirect_graph(
                            self_loop_edge_index, edge_mask
                        )

                    edge_masks.append(edge_mask)

            self.__clear_masks__()
            return edge_masks

        # ── 节点分类分支 ──────────────────────────────────────────────────────
        else:
            self.node_idx = node_idx = kwargs.get('node_idx')
            assert node_idx is not None
            if isinstance(node_idx, torch.Tensor) and not node_idx.dim():
                node_idx = node_idx.to(self.device).flatten()
            elif isinstance(node_idx, (int, list, tuple)):
                node_idx = torch.tensor(
                    [node_idx], device=self.device, dtype=torch.int64
                ).flatten()
            else:
                raise TypeError(
                    f'node_idx should be in types of int, list, tuple, '
                    f'or torch.Tensor, but got {type(node_idx)}'
                )
            self.node_idx = node_idx

            self.subset, _, _, self.hard_edge_mask = subgraph(
                node_idx, self.__num_hops__, self_loop_edge_index,
                relabel_nodes=True, num_nodes=None, flow=self.__flow__()
            )
            self.new_node_idx = torch.where(self.subset == node_idx)[0]

            num_nodes   = x.size(0)
            num_classes = kwargs.get('num_classes')
            labels      = tuple(i for i in range(num_classes))
            ex_labels   = tuple(
                torch.tensor([label]).to(self.device) for label in labels
            )

            # max_nodes 控制 node_mask top-k
            max_nodes = kwargs.get('max_nodes', None)
            sparsity  = kwargs.get('sparsity', 0.5)
            if max_nodes is not None:
                num_keep = max(1, int(max_nodes))
            else:
                num_keep = max(1, int(num_nodes * (1 - sparsity)))

            # 目标节点的直接邻居（含自身），用于限定 node_mask 范围
            s_cpu = edge_index[0].cpu()
            d_cpu = edge_index[1].cpu()
            node_idx_scalar = node_idx[0].item() if node_idx.numel() > 1 else node_idx.item()
            neighbors = set()
            neighbors.add(node_idx_scalar)
            for i in range(edge_index.size(1)):
                u, v = s_cpu[i].item(), d_cpu[i].item()
                if u == node_idx_scalar:
                    neighbors.add(v)
                if v == node_idx_scalar:
                    neighbors.add(u)
            neighbors       = sorted(neighbors)
            neighbor_tensor = torch.tensor(
                neighbors, dtype=torch.long, device=x.device
            )

            edge_masks      = []
            node_masks_list = []
            raw_node_scores = []

            for ex_label in ex_labels:
                if target_label is None or ex_label.item() == target_label.item():
                    self.__clear_masks__()
                    self.__set_masks__(x, self_loop_edge_index)
                    edge_mask_full = self.gnn_explainer_alg(
                        x, self_loop_edge_index, ex_label
                    ).sigmoid()   # shape: [E_self_loop]

                    if self._symmetric_edge_mask_indirect_graph:
                        edge_mask_full = symmetric_edge_mask_indirect_graph(
                            self_loop_edge_index, edge_mask_full
                        )

                    # ── 从 self_loop edge_mask 聚合出全图节点分数 ────────────
                    # 每个节点的分数 = 所有关联边掩码的均值
                    ns  = torch.zeros(num_nodes, device=x.device)
                    deg = torch.zeros(num_nodes, device=x.device)
                    src_sl, dst_sl = self_loop_edge_index[0], self_loop_edge_index[1]
                    ones = torch.ones(self_loop_edge_index.size(1), device=x.device)
                    ns.scatter_add_(0, src_sl, edge_mask_full)
                    ns.scatter_add_(0, dst_sl, edge_mask_full)
                    deg.scatter_add_(0, src_sl, ones)
                    deg.scatter_add_(0, dst_sl, ones)
                    node_scores = ns / (deg + EPS)   # [N]，全图节点分数
                    raw_node_scores.append(node_scores)

                    # ── edge_mask：截取原始边（去掉自环补充部分）────────────
                    n_orig = edge_index.shape[1]
                    edge_masks.append(edge_mask_full[:n_orig].detach())

                    # ── node_mask：在直接邻居范围内选 top-k ─────────────────
                    scores_sub  = node_scores[neighbor_tensor]
                    k_sub       = min(num_keep, len(neighbors))
                    topk_local  = scores_sub.topk(k_sub).indices
                    topk_global = neighbor_tensor[topk_local]

                    nm = torch.zeros(num_nodes, dtype=torch.float32, device=x.device)
                    nm[topk_global] = 1.0
                    node_masks_list.append(nm)

            self.__clear_masks__()

            # last_node_scores 供 Efficiency Gap 使用
            self.last_node_scores = raw_node_scores

            return edge_masks, node_masks_list

    def __repr__(self):
        return f'{self.__class__.__name__}()'