"""
FileName: models.py
Description: GNN models' set
Time: 2020/7/30 9:01
Project: GNN_benchmark
Author: Shurui Gui
"""

import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.utils.loop import add_self_loops, remove_self_loops
from torch_geometric.data.batch import Batch

from typing import Callable, Union, Tuple
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
from torch import Tensor

from torch_sparse import SparseTensor


class GNNBasic(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def arguments_read(self, *args, **kwargs):

        data: Batch = kwargs.get('data') or None

        if not data:
            if not args:
                assert 'x' in kwargs
                assert 'edge_index' in kwargs
                x, edge_index = kwargs['x'], kwargs['edge_index'],
                batch = kwargs.get('batch')
                if batch is None:
                    batch = torch.zeros(kwargs['x'].shape[0], dtype=torch.int64, device=x.device)
            elif len(args) == 2:
                x, edge_index = args[0], args[1]
                batch = torch.zeros(args[0].shape[0], dtype=torch.int64, device=x.device)
            elif len(args) == 3:
                x, edge_index, batch = args[0], args[1], args[2]
            else:
                raise ValueError(f"forward's args should take 2 or 3 arguments but got {len(args)}")
        else:
            x, edge_index, batch = data.x, data.edge_index, data.batch

        return x, edge_index, batch


class GCN_3l(GNNBasic):

    def __init__(self, model_level, dim_node, dim_hidden, num_classes):
        super().__init__()
        num_layer = 3

        self.conv1 = GCNConv(dim_node, dim_hidden)
        self.convs = nn.ModuleList(
            [
                GCNConv(dim_hidden, dim_hidden)
                for _ in range(num_layer - 1)
             ]
        )
        self.relu1 = nn.ReLU()
        self.relus = nn.ModuleList(
            [
                nn.ReLU()
                for _ in range(num_layer - 1)
            ]
        )
        if model_level == 'node':
            self.readout = IdenticalPool()
        else:
            self.readout = GlobalMeanPool()

        self.ffn = nn.Sequential(*(
                [nn.Linear(dim_hidden, dim_hidden)] +
                [nn.ReLU(), nn.Dropout(), nn.Linear(dim_hidden, num_classes)]
        ))

        self.dropout = nn.Dropout()

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        :param Required[data]: Batch - input data
        :return:
        """
        x, edge_index, batch = self.arguments_read(*args, **kwargs)

        post_conv = self.relu1(self.conv1(x, edge_index))
        for conv, relu in zip(self.convs, self.relus):
            post_conv = relu(conv(post_conv, edge_index))

        out_readout = self.readout(post_conv, batch)

        out = self.ffn(out_readout)
        return out

    def get_emb(self, *args, **kwargs) -> torch.Tensor:
        x, edge_index, batch = self.arguments_read(*args, **kwargs)
        post_conv = self.relu1(self.conv1(x, edge_index))
        for conv, relu in zip(self.convs, self.relus):
            post_conv = relu(conv(post_conv, edge_index))
        return post_conv

class GCN_3l_BN(GCN_3l):
    def __init__(self, model_level, dim_node, dim_hidden, num_classes):
        super().__init__(model_level, dim_node, dim_hidden, num_classes)
        num_layer = 3

        self.relu1 = nn.Sequential(
            nn.BatchNorm1d(dim_hidden),
            nn.ReLU()
        )

        self.relus = nn.ModuleList(
            [
                nn.Sequential(
                    nn.BatchNorm1d(dim_hidden),
                    nn.ReLU(),
                )
                for _ in range(num_layer - 1)
            ]
        )

class GCN_2l(GNNBasic):

    def __init__(self, model_level, dim_node, dim_hidden, num_classes):
        super().__init__()
        num_layer = 2

        self.conv1 = GCNConv(dim_node, dim_hidden)
        self.convs = nn.ModuleList(
            [
                GCNConv(dim_hidden, dim_hidden)
                for _ in range(num_layer - 1)
            ]
        )
        self.relu1 = nn.ReLU()
        self.relus = nn.ModuleList(
            [
                nn.ReLU()
                for _ in range(num_layer - 1)
            ]
        )
        if model_level == 'node':
            self.readout = IdenticalPool()
        else:
            self.readout = GlobalMeanPool()

        self.ffn = nn.Sequential(*(
                [nn.Linear(dim_hidden, num_classes)]
        ))

        self.dropout = nn.Dropout()

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        :param Required[data]: Batch - input data
        :return:
        """
        x, edge_index, batch = self.arguments_read(*args, **kwargs)

        post_conv = self.relu1(self.conv1(x, edge_index))
        for conv, relu in zip(self.convs, self.relus):
            post_conv = relu(conv(post_conv, edge_index))

        out_readout = self.readout(post_conv, batch)

        out = self.ffn(out_readout)

        return out

    def get_emb(self, *args, **kwargs) -> torch.Tensor:
        x, edge_index, batch = self.arguments_read(*args, **kwargs)
        
        post_conv = self.relu1(self.conv1(x, edge_index))
        for conv, relu in zip(self.convs, self.relus):
            post_conv = relu(conv(post_conv, edge_index))
            
        return post_conv


class GIN_3l(GNNBasic):

    def __init__(self, model_level, dim_node, dim_hidden, num_classes):
        super().__init__()
        num_layer = 3

        self.conv1 = GINConv(nn.Sequential(nn.Linear(dim_node, dim_hidden), nn.ReLU(),
                                           nn.Linear(dim_hidden, dim_hidden), nn.ReLU()))#,
                                           # nn.BatchNorm1d(dim_hidden)))
        self.convs = nn.ModuleList(
            [
                GINConv(nn.Sequential(nn.Linear(dim_hidden, dim_hidden), nn.ReLU(),
                                      nn.Linear(dim_hidden, dim_hidden), nn.ReLU()))#,
                                      # nn.BatchNorm1d(dim_hidden)))
                for _ in range(num_layer - 1)
             ]
        )
        self.relu1 = nn.ReLU()
        self.relus = nn.ModuleList(
            [
                nn.ReLU()
                for _ in range(num_layer - 1)
            ]
        )
        if model_level == 'node':
            self.readout = IdenticalPool()
        else:
            self.readout = GlobalMeanPool()

        self.ffn = nn.Sequential(*(
                [nn.Linear(dim_hidden, dim_hidden)] +
                [nn.ReLU(), nn.Dropout(), nn.Linear(dim_hidden, num_classes)]
        ))

        self.dropout = nn.Dropout()

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        :param Required[data]: Batch - input data
        :return:
        """
        x, edge_index, batch = self.arguments_read(*args, **kwargs)


        post_conv = self.conv1(x, edge_index)
        for conv in self.convs:
            post_conv = conv(post_conv, edge_index)


        out_readout = self.readout(post_conv, batch)

        out = self.ffn(out_readout)
        return out

    def get_emb(self, *args, **kwargs) -> torch.Tensor:
        x, edge_index, batch = self.arguments_read(*args, **kwargs)
        post_conv = self.conv1(x, edge_index)
        for conv in self.convs:
            post_conv = conv(post_conv, edge_index)
        return post_conv


class GIN_2l(GNNBasic):

    def __init__(self, model_level, dim_node, dim_hidden, num_classes):
        super().__init__()
        num_layer = 2

        self.conv1 = GINConv(nn.Sequential(nn.Linear(dim_node, dim_hidden), nn.ReLU(),
                                           nn.Linear(dim_hidden, dim_hidden), nn.ReLU()))#,
                                           # nn.BatchNorm1d(dim_hidden)))
        self.convs = nn.ModuleList(
            [
                GINConv(nn.Sequential(nn.Linear(dim_hidden, dim_hidden), nn.ReLU(),
                                      nn.Linear(dim_hidden, dim_hidden), nn.ReLU()))#,
                                      # nn.BatchNorm1d(dim_hidden)))
                for _ in range(num_layer - 1)
             ]
        )
        self.relu1 = nn.ReLU()
        self.relus = nn.ModuleList(
            [
                nn.ReLU()
                for _ in range(num_layer - 1)
            ]
        )
        if model_level == 'node':
            self.readout = IdenticalPool()
        else:
            self.readout = GlobalMeanPool()

        self.ffn = nn.Sequential(*(
                [nn.Linear(dim_hidden, dim_hidden)] +
                [nn.ReLU(), nn.Dropout(), nn.Linear(dim_hidden, num_classes)]
        ))

        self.dropout = nn.Dropout()

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        :param Required[data]: Batch - input data
        :return:
        """
        x, edge_index, batch = self.arguments_read(*args, **kwargs)


        post_conv = self.conv1(x, edge_index)
        for conv in self.convs:
            post_conv = conv(post_conv, edge_index)


        out_readout = self.readout(post_conv, batch)

        out = self.ffn(out_readout)
        return out

    def get_emb(self, *args, **kwargs) -> torch.Tensor:
        x, edge_index, batch = self.arguments_read(*args, **kwargs)
        post_conv = self.conv1(x, edge_index)
        for conv in self.convs:
            post_conv = conv(post_conv, edge_index)
        return post_conv


class GCNConv(gnn.GCNConv):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__explain_flow__ = False
        self.edge_weight = None
        self.layer_edge_mask = None
        self.weight = nn.Parameter(self.lin.weight.data.T.clone().detach())

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""

        if self.normalize and edge_weight is None:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gnn.conv.gcn_conv.gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gnn.conv.gcn_conv.gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        # --- add require_grad ---
        edge_weight.requires_grad_(True)

        x = torch.matmul(x, self.weight)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        if self.bias is not None:
            out += self.bias

        # --- My: record edge_weight ---
        self.edge_weight = edge_weight

        return out

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        size = self._check_input(edge_index, size)

        # Run "fused" message and aggregation (if applicable).
        if (isinstance(edge_index, SparseTensor) and self.fuse
                and not self._explain):
            coll_dict = self._collect(self._fused_user_args, edge_index,
                                      size, kwargs)

            # 对于 SparseTensor，直接调用 message_and_aggregate
            out = self.message_and_aggregate(edge_index, **coll_dict)
            return self.update(out, **coll_dict)

        # Otherwise, run both functions in separation.
        elif isinstance(edge_index, Tensor) or not self.fuse:
            coll_dict = self._collect(self._user_args, edge_index, size,
                                      kwargs)

            # 过滤掉 message 方法不需要的参数
            # PyG 内置的 GCNConv.message 方法通常需要 x_j 和 edge_weight
            message_kwargs = {}
            if 'x_j' in coll_dict:
                message_kwargs['x_j'] = coll_dict['x_j']
            if 'edge_weight' in coll_dict:
                message_kwargs['edge_weight'] = coll_dict['edge_weight']
            
            out = self.message(**message_kwargs)
            
            # For `GNNExplainer`, we require a separate message and aggregate
            # procedure since this allows us to inject the `edge_mask` into the
            # message passing computation scheme.
            if self._explain:
                edge_mask = self.__edge_mask__
                # Some ops add self-loops to `edge_index`. We need to do the
                # same for `edge_mask` (but do not train those).
                if out.size(self.node_dim) != edge_mask.size(0):
                    loop = edge_mask.new_ones(size[0])
                    edge_mask = torch.cat([edge_mask, loop], dim=0)
                assert out.size(self.node_dim) == edge_mask.size(0)
                out = out * edge_mask.view([-1] + [1] * (out.dim() - 1))
            elif self.__explain_flow__:
                edge_mask = self.layer_edge_mask
                # Some ops add self-loops to `edge_index`. We need to do the
                # same for `edge_mask` (but do not train those).
                if out.size(self.node_dim) != edge_mask.size(0):
                    loop = edge_mask.new_ones(size[0])
                    edge_mask = torch.cat([edge_mask, loop], dim=0)
                assert out.size(self.node_dim) == edge_mask.size(0)
                out = out * edge_mask.view([-1] + [1] * (out.dim() - 1))

            # 对于 aggregate 和 update，同样过滤参数
            aggregate_kwargs = {k: v for k, v in coll_dict.items() 
                               if k in ['index', 'ptr', 'dim_size']}
            out = self.aggregate(out, **aggregate_kwargs)
            
            return self.update(out)


class GINConv(gnn.GINConv):

    def __init__(self, nn: Callable, eps: float = 0., train_eps: bool = False,
                 **kwargs):
        super().__init__(nn, eps, train_eps, **kwargs)
        self.__explain_flow__ = False
        self.edge_weight = None
        self.layer_edge_mask = None
        self.fc_steps = None
        self.reweight = None


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_weight: OptTensor = None, task='explain', **kwargs) -> Tensor:
        """"""
        self.num_nodes = x.shape[0]
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        if edge_weight is not None:
            self.edge_weight = edge_weight
            assert edge_weight.shape[0] == edge_index.shape[1]
            self.reweight = False
        else:
            edge_index, _ = remove_self_loops(edge_index)
            self_loop_edge_index, _ = add_self_loops(edge_index, num_nodes=self.num_nodes)
            if self_loop_edge_index.shape[1] != edge_index.shape[1]:
                edge_index = self_loop_edge_index
            self.reweight = True
        out = self.propagate(edge_index, x=x[0], size=None)

        if task == 'explain':
            layer_extractor = []
            hooks = []

            def register_hook(module: nn.Module):
                if not list(module.children()):
                    hooks.append(module.register_forward_hook(forward_hook))

            def forward_hook(module: nn.Module, input: Tuple[Tensor], output: Tensor):
                # input contains x and edge_index
                layer_extractor.append((module, input[0], output))

            # --- register hooks ---
            self.nn.apply(register_hook)

            nn_out = self.nn(out)

            for hook in hooks:
                hook.remove()

            fc_steps = []
            step = {'input': None, 'module': [], 'output': None}
            for layer in layer_extractor:
                if isinstance(layer[0], nn.Linear):
                    if step['module']:
                        fc_steps.append(step)
                    # step = {'input': layer[1], 'module': [], 'output': None}
                    step = {'input': None, 'module': [], 'output': None}
                step['module'].append(layer[0])
                if kwargs.get('probe'):
                    step['output'] = layer[2]
                else:
                    step['output'] = None

            if step['module']:
                fc_steps.append(step)
            self.fc_steps = fc_steps
        else:
            nn_out = self.nn(out)


        return nn_out

    def message(self, x_j: Tensor) -> Tensor:
        if self.reweight:
            edge_weight = torch.ones(x_j.shape[0], device=x_j.device)
            edge_weight.data[-self.num_nodes:] += self.eps
            edge_weight = edge_weight.detach().clone()
            edge_weight.requires_grad_(True)
            self.edge_weight = edge_weight
        return x_j * self.edge_weight.view(-1, 1)

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        size = self._check_input(edge_index, size)

        # Run "fused" message and aggregation (if applicable).
        if (isinstance(edge_index, SparseTensor) and self.fuse
                and not self._explain):
            coll_dict = self._collect(self._fused_user_args, edge_index,
                                      size, kwargs)

            out = self.message_and_aggregate(edge_index, **coll_dict)
            return self.update(out)  # 只传递out，不传递其他参数

        # Otherwise, run both functions in separation.
        elif isinstance(edge_index, Tensor) or not self.fuse:
            coll_dict = self._collect(self._user_args, edge_index, size,
                                      kwargs)

            # 只传递 message 方法需要的参数
            msg_kwargs = {}
            if 'x_j' in coll_dict:
                msg_kwargs['x_j'] = coll_dict['x_j']
            if 'edge_weight' in coll_dict:
                msg_kwargs['edge_weight'] = coll_dict['edge_weight']
            
            out = self.message(**msg_kwargs)

            # For `GNNExplainer`, we require a separate message and aggregate
            # procedure since this allows us to inject the `edge_mask` into the
            # message passing computation scheme.
            if self._explain:
                edge_mask = self.__edge_mask__
                # Some ops add self-loops to `edge_index`. We need to do the
                # same for `edge_mask` (but do not train those).
                if out.size(self.node_dim) != edge_mask.size(0):
                    loop = edge_mask.new_ones(size[0])
                    edge_mask = torch.cat([edge_mask, loop], dim=0)
                assert out.size(self.node_dim) == edge_mask.size(0)
                out = out * edge_mask.view([-1] + [1] * (out.dim() - 1))
            elif self.__explain_flow__:
                edge_mask = self.layer_edge_mask
                # Some ops add self-loops to `edge_index`. We need to do the
                # same for `edge_mask` (but do not train those).
                if out.size(self.node_dim) != edge_mask.size(0):
                    loop = edge_mask.new_ones(size[0])
                    edge_mask = torch.cat([edge_mask, loop], dim=0)
                assert out.size(self.node_dim) == edge_mask.size(0)
                out = out * edge_mask.view([-1] + [1] * (out.dim() - 1))

            # 对于 aggregate，只传递需要的参数
            aggr_kwargs = {k: v for k, v in coll_dict.items() 
                          if k in ['index', 'ptr', 'dim_size']}
            out = self.aggregate(out, **aggr_kwargs)

            # 关键修复：update 方法只接受 out 参数
            return self.update(out)


class GNNPool(nn.Module):
    def __init__(self):
        super().__init__()


class GlobalMeanPool(GNNPool):

    def __init__(self):
        super().__init__()

    def forward(self, x, batch):
        return gnn.global_mean_pool(x, batch)


class IdenticalPool(GNNPool):

    def __init__(self):
        super().__init__()

    def forward(self, x, batch):
        return x


class GraphSequential(nn.Sequential):

    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, *input) -> Tensor:
        for module in self:
            if isinstance(input, tuple):
                input = module(*input)
            else:
                input = module(input)
        return input


# explain_mask in propagation haven't pass sigmoid func
class GCN_2l_mask(GNNBasic):

    def __init__(self, model_level, dim_node, dim_hidden, num_classes):
        super().__init__()
        num_layer = 2

        self.conv1 = GCNConv_mask(dim_node, dim_hidden)
        self.convs = nn.ModuleList(
            [
                GCNConv_mask(dim_hidden, dim_hidden)
                for _ in range(num_layer - 1)
            ]
        )
        self.relu1 = nn.ReLU()
        self.relus = nn.ModuleList(
            [
                nn.ReLU()
                for _ in range(num_layer - 1)
            ]
        )
        if model_level == 'node':
            self.readout = IdenticalPool()
        else:
            self.readout = GlobalMeanPool()

        self.ffn = nn.Sequential(*(
                [nn.Linear(dim_hidden, num_classes)]
        ))

        self.dropout = nn.Dropout()

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        :param Required[data]: Batch - input data
        :return:
        """
        x, edge_index, batch = self.arguments_read(*args, **kwargs)

        post_conv = self.relu1(self.conv1(x, edge_index))
        for conv, relu in zip(self.convs, self.relus):
            post_conv = relu(conv(post_conv, edge_index))

        out_readout = self.readout(post_conv, batch)

        out = self.ffn(out_readout)

        return out

    def get_emb(self, *args, **kwargs) -> torch.Tensor:
        x, edge_index, batch = self.arguments_read(*args, **kwargs)
        post_conv = self.conv1(x, edge_index)
        for conv in self.convs:
            post_conv = conv(post_conv, edge_index)
        return post_conv


class GIN_2l_mask(GNNBasic):

    def __init__(self, model_level, dim_node, dim_hidden, num_classes):
        super().__init__()
        num_layer = 2

        self.conv1 = GINConv_mask(nn.Sequential(nn.Linear(dim_node, dim_hidden), nn.ReLU(),
                                           nn.Linear(dim_hidden, dim_hidden), nn.ReLU()))#,
                                           # nn.BatchNorm1d(dim_hidden)))
        self.convs = nn.ModuleList(
            [
                GINConv_mask(nn.Sequential(nn.Linear(dim_hidden, dim_hidden), nn.ReLU(),
                                      nn.Linear(dim_hidden, dim_hidden), nn.ReLU()))#,
                                      # nn.BatchNorm1d(dim_hidden)))
                for _ in range(num_layer - 1)
             ]
        )
        self.relu1 = nn.ReLU()
        self.relus = nn.ModuleList(
            [
                nn.ReLU()
                for _ in range(num_layer - 1)
            ]
        )
        if model_level == 'node':
            self.readout = IdenticalPool()
        else:
            self.readout = GlobalMeanPool()

        self.ffn = nn.Sequential(*(
                [nn.Linear(dim_hidden, dim_hidden)] +
                [nn.ReLU(), nn.Dropout(), nn.Linear(dim_hidden, num_classes)]
        ))

        self.dropout = nn.Dropout()

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        :param Required[data]: Batch - input data
        :return:
        """
        x, edge_index, batch = self.arguments_read(*args, **kwargs)


        post_conv = self.conv1(x, edge_index)
        for conv in self.convs:
            post_conv = conv(post_conv, edge_index)


        out_readout = self.readout(post_conv, batch)

        out = self.ffn(out_readout)
        return out

    def get_emb(self, *args, **kwargs) -> torch.Tensor:
        x, edge_index, batch = self.arguments_read(*args, **kwargs)
        post_conv = self.conv1(x, edge_index)
        for conv in self.convs:
            post_conv = conv(post_conv, edge_index)
        return post_conv


class GCNConv_mask(gnn.GCNConv):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__explain_flow__ = False
        self.edge_weight = None
        self.layer_edge_mask = None
        self.weight = nn.Parameter(self.lin.weight.data.T.clone().detach())

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""

        if self.normalize and edge_weight is None:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gnn.conv.gcn_conv.gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gnn.conv.gcn_conv.gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        # --- add require_grad ---
        edge_weight.requires_grad_(True)

        x = torch.matmul(x, self.weight)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        if self.bias is not None:
            out += self.bias

        # --- My: record edge_weight ---
        self.edge_weight = edge_weight

        return out

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        size = self._check_input(edge_index, size)

        # Run "fused" message and aggregation (if applicable).
        if (isinstance(edge_index, SparseTensor) and self.fuse
                and not self._explain):
            coll_dict = self._collect(self._fused_user_args, edge_index,
                                         size, kwargs)

            msg_aggr_kwargs = coll_dict
            out = self.message_and_aggregate(edge_index, **msg_aggr_kwargs)

            update_kwargs = coll_dict
            return self.update(out, **update_kwargs)

        # Otherwise, run both functions in separation.
        elif isinstance(edge_index, Tensor) or not self.fuse:
            coll_dict = self._collect(self._user_args, edge_index, size,
                                         kwargs)

            msg_kwargs = coll_dict
            out = self.message(**msg_kwargs)

            # For `GNNExplainer`, we require a separate message and aggregate
            # procedure since this allows us to inject the `edge_mask` into the
            # message passing computation scheme.
            if self._explain:
                edge_mask = self.__edge_mask__
                # Some ops add self-loops to `edge_index`. We need to do the
                # same for `edge_mask` (but do not train those).
                if out.size(self.node_dim) != edge_mask.size(0):
                    loop = edge_mask.new_ones(size[0])
                    edge_mask = torch.cat([edge_mask, loop], dim=0)
                assert out.size(self.node_dim) == edge_mask.size(0)
                out = out * edge_mask.view([-1] + [1] * (out.dim() - 1))
            elif self.__explain_flow__:
                edge_mask = self.layer_edge_mask
                # Some ops add self-loops to `edge_index`. We need to do the
                # same for `edge_mask` (but do not train those).
                if out.size(self.node_dim) != edge_mask.size(0):
                    loop = edge_mask.new_ones(size[0])
                    edge_mask = torch.cat([edge_mask, loop], dim=0)
                assert out.size(self.node_dim) == edge_mask.size(0)
                out = out * edge_mask.view([-1] + [1] * (out.dim() - 1))

            aggr_kwargs = coll_dict
            out = self.aggregate(out, **aggr_kwargs)

            update_kwargs = coll_dict
            return self.update(out, **update_kwargs)


class GINConv_mask(gnn.GINConv):

    def __init__(self, nn: Callable, eps: float = 0., train_eps: bool = False,
                 **kwargs):
        super().__init__(nn, eps, train_eps, **kwargs)
        self.__explain_flow__ = False
        self.edge_weight = None
        self.layer_edge_mask = None
        self.fc_steps = None
        self.reweight = None


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_weight: OptTensor = None, task='explain', **kwargs) -> Tensor:
        """"""
        self.num_nodes = x.shape[0]
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        if edge_weight is not None:
            self.edge_weight = edge_weight
            assert edge_weight.shape[0] == edge_index.shape[1]
            self.reweight = False
        else:
            edge_index, _ = remove_self_loops(edge_index)
            self_loop_edge_index, _ = add_self_loops(edge_index, num_nodes=self.num_nodes)
            if self_loop_edge_index.shape[1] != edge_index.shape[1]:
                edge_index = self_loop_edge_index
            self.reweight = True
        out = self.propagate(edge_index, x=x[0], size=None)

        if task == 'explain':
            layer_extractor = []
            hooks = []

            def register_hook(module: nn.Module):
                if not list(module.children()):
                    hooks.append(module.register_forward_hook(forward_hook))

            def forward_hook(module: nn.Module, input: Tuple[Tensor], output: Tensor):
                # input contains x and edge_index
                layer_extractor.append((module, input[0], output))

            # --- register hooks ---
            self.nn.apply(register_hook)

            nn_out = self.nn(out)

            for hook in hooks:
                hook.remove()

            fc_steps = []
            step = {'input': None, 'module': [], 'output': None}
            for layer in layer_extractor:
                if isinstance(layer[0], nn.Linear):
                    if step['module']:
                        fc_steps.append(step)
                    # step = {'input': layer[1], 'module': [], 'output': None}
                    step = {'input': None, 'module': [], 'output': None}
                step['module'].append(layer[0])
                if kwargs.get('probe'):
                    step['output'] = layer[2]
                else:
                    step['output'] = None

            if step['module']:
                fc_steps.append(step)
            self.fc_steps = fc_steps
        else:
            nn_out = self.nn(out)


        return nn_out

    def message(self, x_j: Tensor) -> Tensor:
        if self.reweight:
            edge_weight = torch.ones(x_j.shape[0], device=x_j.device)
            edge_weight.data[-self.num_nodes:] += self.eps
            edge_weight = edge_weight.detach().clone()
            edge_weight.requires_grad_(True)
            self.edge_weight = edge_weight
        return x_j * self.edge_weight.view(-1, 1)

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        size = self._check_input(edge_index, size)

        # Run "fused" message and aggregation (if applicable).
        if (isinstance(edge_index, SparseTensor) and self.fuse
                and not self._explain):
            coll_dict = self._collect(self._fused_user_args, edge_index,
                                      size, kwargs)

            out = self.message_and_aggregate(edge_index, **coll_dict)
            return self.update(out)  # 只传递out，不传递其他参数

        # Otherwise, run both functions in separation.
        elif isinstance(edge_index, Tensor) or not self.fuse:
            coll_dict = self._collect(self._user_args, edge_index, size,
                                      kwargs)

            # 只传递 message 方法需要的参数
            msg_kwargs = {}
            if 'x_j' in coll_dict:
                msg_kwargs['x_j'] = coll_dict['x_j']
            if 'edge_weight' in coll_dict:
                msg_kwargs['edge_weight'] = coll_dict['edge_weight']
            
            out = self.message(**msg_kwargs)

            # For `GNNExplainer`, we require a separate message and aggregate
            # procedure since this allows us to inject the `edge_mask` into the
            # message passing computation scheme.
            if self._explain:
                edge_mask = self.__edge_mask__
                # Some ops add self-loops to `edge_index`. We need to do the
                # same for `edge_mask` (but do not train those).
                if out.size(self.node_dim) != edge_mask.size(0):
                    loop = edge_mask.new_ones(size[0])
                    edge_mask = torch.cat([edge_mask, loop], dim=0)
                assert out.size(self.node_dim) == edge_mask.size(0)
                out = out * edge_mask.view([-1] + [1] * (out.dim() - 1))
            elif self.__explain_flow__:
                edge_mask = self.layer_edge_mask
                # Some ops add self-loops to `edge_index`. We need to do the
                # same for `edge_mask` (but do not train those).
                if out.size(self.node_dim) != edge_mask.size(0):
                    loop = edge_mask.new_ones(size[0])
                    edge_mask = torch.cat([edge_mask, loop], dim=0)
                assert out.size(self.node_dim) == edge_mask.size(0)
                out = out * edge_mask.view([-1] + [1] * (out.dim() - 1))

            # 对于 aggregate，只传递需要的参数
            aggr_kwargs = {k: v for k, v in coll_dict.items() 
                          if k in ['index', 'ptr', 'dim_size']}
            out = self.aggregate(out, **aggr_kwargs)

            # 关键修复：update 方法只接受 out 参数
            return self.update(out)
        

class PolyGNN_3l(GNNBasic):
    """
    Polynomial Graph Neural Network with 3 layers.
    Replaces non-linear activations with learnable polynomial transformations.
    """
    
    def __init__(self, model_level, dim_node, dim_hidden, num_classes, poly_degree=3):
        super().__init__()
        num_layer = 3
        
        # GCN convolution layers
        self.conv1 = GCNConv(dim_node, dim_hidden)
        self.convs = nn.ModuleList(
            [
                GCNConv(dim_hidden, dim_hidden)
                for _ in range(num_layer - 1)
            ]
        )
        
        # Layer normalization before polynomial activation
        self.norm1 = nn.LayerNorm(dim_hidden)
        self.norms = nn.ModuleList(
            [
                nn.LayerNorm(dim_hidden)
                for _ in range(num_layer - 1)
            ]
        )
        
        # Polynomial activation layers
        self.poly1 = PolyActivation(dim_hidden, degree=poly_degree)
        self.polys = nn.ModuleList(
            [
                PolyActivation(dim_hidden, degree=poly_degree)
                for _ in range(num_layer - 1)
            ]
        )
        
        # Dropout
        self.dropout = nn.Dropout()
        
        # Readout layer
        if model_level == 'node':
            self.readout = IdenticalPool()
        else:
            self.readout = GlobalMeanPool()
        
        # Final feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.LayerNorm(dim_hidden),
            PolyActivation(dim_hidden, degree=poly_degree),
            self.dropout,
            nn.Linear(dim_hidden, num_classes)
        )
        
        # Store polynomial degree for attribution
        self.poly_degree = poly_degree
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass of PolyGNN.
        """
        x, edge_index, batch = self.arguments_read(*args, **kwargs)
        
        # Layer 1: Conv -> Norm -> Poly Activation
        z1 = self.conv1(x, edge_index)
        z1_norm = self.norm1(z1)
        h1 = self.poly1(z1_norm)
        h1 = self.dropout(h1)
        
        # Subsequent layers
        h = h1
        for conv, norm, poly in zip(self.convs, self.norms, self.polys):
            z = conv(h, edge_index)
            z_norm = norm(z)
            h = poly(z_norm)
            h = self.dropout(h)
        
        # Readout and final classification
        out_readout = self.readout(h, batch)
        out = self.ffn(out_readout)
        
        return out
    
    def get_emb(self, *args, **kwargs) -> torch.Tensor:
        """
        Get node embeddings (before the final classifier).
        """
        x, edge_index, batch = self.arguments_read(*args, **kwargs)
        
        # Layer 1
        z1 = self.conv1(x, edge_index)
        z1_norm = self.norm1(z1)
        h1 = self.poly1(z1_norm)
        
        # Subsequent layers
        h = h1
        for conv, norm, poly in zip(self.convs, self.norms, self.polys):
            z = conv(h, edge_index)
            z_norm = norm(z)
            h = poly(z_norm)
        
        return h
    
    def get_polynomial_coefficients(self):
        """
        Get polynomial coefficients for interpretability.
        Returns a list of coefficient tensors for each layer.
        """
        coeffs = []
        coeffs.append(self.poly1.get_coefficients())
        for poly in self.polys:
            coeffs.append(poly.get_coefficients())
        return coeffs


class PolyGNN_2l(GNNBasic):
    """
    Polynomial Graph Neural Network with 2 layers.
    """
    
    def __init__(self, model_level, dim_node, dim_hidden, num_classes, poly_degree=3):
        super().__init__()
        num_layer = 2
        
        # GCN convolution layers
        self.conv1 = GCNConv(dim_node, dim_hidden)
        self.convs = nn.ModuleList(
            [
                GCNConv(dim_hidden, dim_hidden)
                for _ in range(num_layer - 1)
            ]
        )
        
        # Layer normalization before polynomial activation
        self.norm1 = nn.LayerNorm(dim_hidden)
        self.norms = nn.ModuleList(
            [
                nn.LayerNorm(dim_hidden)
                for _ in range(num_layer - 1)
            ]
        )
        
        # Polynomial activation layers
        self.poly1 = PolyActivation(dim_hidden, degree=poly_degree)
        self.polys = nn.ModuleList(
            [
                PolyActivation(dim_hidden, degree=poly_degree)
                for _ in range(num_layer - 1)
            ]
        )
        
        # Dropout
        self.dropout = nn.Dropout()
        
        # Readout layer
        if model_level == 'node':
            self.readout = IdenticalPool()
        else:
            self.readout = GlobalMeanPool()
        
        # Final feed-forward network (simpler for 2-layer version)
        self.ffn = nn.Sequential(*(
            [nn.Linear(dim_hidden, num_classes)]
        ))
        
        # Store polynomial degree for attribution
        self.poly_degree = poly_degree
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass of PolyGNN.
        """
        x, edge_index, batch = self.arguments_read(*args, **kwargs)
        
        # Layer 1: Conv -> Norm -> Poly Activation
        z1 = self.conv1(x, edge_index)
        z1_norm = self.norm1(z1)
        h1 = self.poly1(z1_norm)
        h1 = self.dropout(h1)
        
        # Subsequent layers
        h = h1
        for conv, norm, poly in zip(self.convs, self.norms, self.polys):
            z = conv(h, edge_index)
            z_norm = norm(z)
            h = poly(z_norm)
            h = self.dropout(h)
        
        # Readout and final classification
        out_readout = self.readout(h, batch)
        out = self.ffn(out_readout)
        
        return out
    
    def get_emb(self, *args, **kwargs) -> torch.Tensor:
        """
        Get node embeddings (before the final classifier).
        """
        x, edge_index, batch = self.arguments_read(*args, **kwargs)
        
        # Layer 1
        z1 = self.conv1(x, edge_index)
        z1_norm = self.norm1(z1)
        h1 = self.poly1(z1_norm)
        
        # Subsequent layers
        h = h1
        for conv, norm, poly in zip(self.convs, self.norms, self.polys):
            z = conv(h, edge_index)
            z_norm = norm(z)
            h = poly(z_norm)
        
        return h
    
    def get_polynomial_coefficients(self):
        """
        Get polynomial coefficients for interpretability.
        """
        coeffs = []
        coeffs.append(self.poly1.get_coefficients())
        for poly in self.polys:
            coeffs.append(poly.get_coefficients())
        return coeffs


class PolyActivation(nn.Module):
    """
    Learnable polynomial activation function.
    P(x) = a_1 * x + a_2 * x^2 + ... + a_k * x^k
    """
    
    def __init__(self, dim, degree=3):
        super().__init__()
        self.dim = dim
        self.degree = degree
        
        # Learnable polynomial coefficients (scalars shared across all features)
        self.coeffs = nn.Parameter(torch.zeros(degree))
        
        # Initialize coefficients to approximate ReLU near zero
        # For ReLU: f(x)=max(0,x) ≈ x for x>0, 0 for x<=0
        # We initialize with a_1=1, others small random values
        with torch.no_grad():
            self.coeffs[0] = 1.0  # Linear term approximates identity
            # Initialize higher-order terms with small random values
            self.coeffs[1:] = torch.randn(degree - 1) * 0.01
        
        # Epsilon for numerical stability
        self.eps = 1e-8
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply polynomial transformation element-wise.
        x: tensor of shape [batch_size, dim] or [num_nodes, dim]
        """
        # Apply polynomial: sum_{k=1}^{degree} a_k * x^k
        result = torch.zeros_like(x)
        for k in range(1, self.degree + 1):
            # Element-wise power
            x_pow = x.pow(k)
            # Multiply by coefficient (scalar)
            result = result + self.coeffs[k-1] * x_pow
        
        return result
    
    def get_coefficients(self) -> torch.Tensor:
        """
        Get the polynomial coefficients.
        """
        return self.coeffs.clone()
    
    def extra_repr(self) -> str:
        return f'dim={self.dim}, degree={self.degree}'


# ============================================================================
# PolyGNN with GIN convolution
# ============================================================================

'''
class PolyGIN_3l(GNNBasic):

    def __init__(self, model_level, dim_node, dim_hidden, num_classes, poly_degree=3):
        super().__init__()
        num_layer = 3
        
        # GIN convolution layers with simple MLPs
        self.conv1 = GINConv(
            nn.Sequential(
                nn.Linear(dim_node, dim_hidden),
                nn.LayerNorm(dim_hidden),
                PolyActivation(dim_hidden, degree=poly_degree),
                nn.Linear(dim_hidden, dim_hidden),
            )
        )
        
        self.convs = nn.ModuleList(
            [
                GINConv(
                    nn.Sequential(
                        nn.Linear(dim_hidden, dim_hidden),
                        nn.LayerNorm(dim_hidden),
                        PolyActivation(dim_hidden, degree=poly_degree),
                        nn.Linear(dim_hidden, dim_hidden),
                    )
                )
                for _ in range(num_layer - 1)
            ]
        )

        self.dropout = nn.Dropout()

        if model_level == 'node':
            self.readout = IdenticalPool()
        else:
            self.readout = GlobalMeanPool()
        
        self.ffn = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.LayerNorm(dim_hidden),
            PolyActivation(dim_hidden, degree=poly_degree),
            self.dropout,
            nn.Linear(dim_hidden, num_classes)
        )
        
        self.poly_degree = poly_degree
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        x, edge_index, batch = self.arguments_read(*args, **kwargs)
        
        # First layer
        h1 = self.conv1(x, edge_index)
        h1 = self.dropout(h1)
        
        h = h1
        for conv in self.convs:
            h = conv(h, edge_index)
            h = self.dropout(h)
        
        out_readout = self.readout(h, batch)
        out = self.ffn(out_readout)
        
        return out
    
    def get_emb(self, *args, **kwargs) -> torch.Tensor:
        x, edge_index, batch = self.arguments_read(*args, **kwargs)
        
        h = self.conv1(x, edge_index)
        for conv in self.convs:
            h = conv(h, edge_index)
        
        return h
'''

class PolyGIN_3l(GNNBasic):
    
    def __init__(self, model_level, dim_node, dim_hidden, num_classes, poly_degree=3):
        super().__init__()
        num_layer = 3
        
       
        self.conv1 = GINConv(
            nn.Sequential(
                nn.Linear(dim_node, dim_hidden),
                PolyActivation(dim_hidden, degree=poly_degree),
                nn.Linear(dim_hidden, dim_hidden),
                PolyActivation(dim_hidden, degree=poly_degree)
            )
        )
        
        self.convs = nn.ModuleList(
            [
                GINConv(
                    nn.Sequential(
                        nn.Linear(dim_hidden, dim_hidden),
                        PolyActivation(dim_hidden, degree=poly_degree),
                        nn.Linear(dim_hidden, dim_hidden),
                        PolyActivation(dim_hidden, degree=poly_degree)
                    )
                )
                for _ in range(num_layer - 1)
            ]
        )
        
        self.norms = nn.ModuleList(
            [nn.LayerNorm(dim_hidden) for _ in range(num_layer)]
        )
        
        self.dropout = nn.Dropout(0.5)  
        
        if model_level == 'node':
            self.readout = IdenticalPool()
        else:
            self.readout = GlobalMeanPool()
        
        # 简化FFN
        self.ffn = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            PolyActivation(dim_hidden, degree=poly_degree),
            nn.Dropout(0.5),
            nn.Linear(dim_hidden, num_classes)
        )
        
        self.poly_degree = poly_degree
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        x, edge_index, batch = self.arguments_read(*args, **kwargs)
        
        h = self.conv1(x, edge_index)
        h = self.norms[0](h) 
        h = self.dropout(h)
        
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            h = self.norms[i+1](h)  
            h = self.dropout(h)
        
        out_readout = self.readout(h, batch)
        out = self.ffn(out_readout)
        
        return out
    
    def get_emb(self, *args, **kwargs) -> torch.Tensor:
        x, edge_index, batch = self.arguments_read(*args, **kwargs)
        
        h = self.conv1(x, edge_index)
        h = self.norms[0](h)
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            h = self.norms[i+1](h)
        
        return h


class PolyGIN_2l(GNNBasic):
    """
    PolyGNN with GIN convolution layers (2-layer version).
    """
    
    def __init__(self, model_level, dim_node, dim_hidden, num_classes, poly_degree=3):
        super().__init__()
        num_layer = 2
        
        # GIN convolution layers
        self.conv1 = GINConv(
            nn.Sequential(
                nn.Linear(dim_node, dim_hidden),
                nn.LayerNorm(dim_hidden),
                PolyActivation(dim_hidden, degree=poly_degree),
                nn.Linear(dim_hidden, dim_hidden),
            )
        )
        
        self.convs = nn.ModuleList(
            [
                GINConv(
                    nn.Sequential(
                        nn.Linear(dim_hidden, dim_hidden),
                        nn.LayerNorm(dim_hidden),
                        PolyActivation(dim_hidden, degree=poly_degree),
                        nn.Linear(dim_hidden, dim_hidden),
                    )
                )
                for _ in range(num_layer - 1)
            ]
        )
        
        # Dropout
        self.dropout = nn.Dropout()
        
        # Readout layer
        if model_level == 'node':
            self.readout = IdenticalPool()
        else:
            self.readout = GlobalMeanPool()
        
        # Final feed-forward network
        self.ffn = nn.Sequential(*(
            [nn.Linear(dim_hidden, num_classes)]
        ))
        
        # Store polynomial degree
        self.poly_degree = poly_degree
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        x, edge_index, batch = self.arguments_read(*args, **kwargs)
        
        # First layer
        h1 = self.conv1(x, edge_index)
        h1 = self.dropout(h1)
        
        # Subsequent layers
        h = h1
        for conv in self.convs:
            h = conv(h, edge_index)
            h = self.dropout(h)
        
        # Readout and classification
        out_readout = self.readout(h, batch)
        out = self.ffn(out_readout)
        
        return out
    
    def get_emb(self, *args, **kwargs) -> torch.Tensor:
        x, edge_index, batch = self.arguments_read(*args, **kwargs)
        
        h = self.conv1(x, edge_index)
        for conv in self.convs:
            h = conv(h, edge_index)
        
        return h