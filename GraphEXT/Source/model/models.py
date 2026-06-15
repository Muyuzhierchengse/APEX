
import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.utils.loop import add_self_loops, remove_self_loops
from torch_geometric.data.batch import Batch

from typing import Callable, Union, Tuple
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
from torch import Tensor

from torch_sparse import SparseTensor

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
            return self.update(out)  

        # Otherwise, run both functions in separation.
        elif isinstance(edge_index, Tensor) or not self.fuse:
            coll_dict = self._collect(self._user_args, edge_index, size,
                                      kwargs)

            msg_kwargs = {}
            if 'x_j' in coll_dict:
                msg_kwargs['x_j'] = coll_dict['x_j']
            if 'edge_weight' in coll_dict:
                msg_kwargs['edge_weight'] = coll_dict['edge_weight']
            
            out = self.message(**msg_kwargs)

            if self._explain:
                edge_mask = self.__edge_mask__
          
                if out.size(self.node_dim) != edge_mask.size(0):
                    loop = edge_mask.new_ones(size[0])
                    edge_mask = torch.cat([edge_mask, loop], dim=0)
                assert out.size(self.node_dim) == edge_mask.size(0)
                out = out * edge_mask.view([-1] + [1] * (out.dim() - 1))
            elif self.__explain_flow__:
                edge_mask = self.layer_edge_mask
               
                if out.size(self.node_dim) != edge_mask.size(0):
                    loop = edge_mask.new_ones(size[0])
                    edge_mask = torch.cat([edge_mask, loop], dim=0)
                assert out.size(self.node_dim) == edge_mask.size(0)
                out = out * edge_mask.view([-1] + [1] * (out.dim() - 1))

            aggr_kwargs = {k: v for k, v in coll_dict.items() 
                          if k in ['index', 'ptr', 'dim_size']}
            out = self.aggregate(out, **aggr_kwargs)

            return self.update(out)
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

# ──────────────────────────────────────────────────────────────────────────────
# 二阶多项式激活  f(x) = x + alpha * x^2
# ──────────────────────────────────────────────────────────────────────────────
class PolyActivation(nn.Module):
    """
    f(x) = x + alpha * x^2
    保留线性残差项 x，保证梯度通路畅通（类似 ResNet skip）。
    alpha 可学习，初始化为小值，训练中自动调节非线性强度。
    """
    def __init__(self, alpha: float = 0.05):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))

    def forward(self, x: Tensor) -> Tensor:
        # clamp alpha 防止其绝对值过大导致 x^2 爆炸
        alpha = self.alpha.clamp(-0.5, 0.5)
        return x + alpha * x * x


# ──────────────────────────────────────────────────────────────────────────────
# PolyScaleNorm：纯代数幅度控制，不破坏多项式性质
# 只在特征幅度超过阈值时才缩放，避免过度压制梯度
# ──────────────────────────────────────────────────────────────────────────────
class PolyScaleNorm(nn.Module):
    """
    纯多项式幅度控制：x * scale（逐维可学习标量缩放）
    等价于在线性层权重上乘一个对角矩阵，完全是代数操作。
    scale 初始化为小值（0.3），防止初期 x^2 爆炸；
    训练中自动调节，不引入任何分段或非多项式操作。
    """
    def __init__(self, dim: int, init_scale: float = 0.3):
        super().__init__()
        self.scale = nn.Parameter(torch.full((dim,), init_scale))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.scale


# ──────────────────────────────────────────────────────────────────────────────
# PolyMLP：严格单激活（每层阶数恰好 ×2）
# 结构：Linear -> PolyScaleNorm -> PolyAct -> Linear
# 线性层使用普通 init（不用谱归一化），靠 PolyScaleNorm 控制幅度
# ──────────────────────────────────────────────────────────────────────────────
def _poly_mlp(in_f: int, out_f: int) -> nn.Sequential:
    """
    单 PolyAct：每次调用阶数 ×2。
    L=3 层后整体阶数 = 2^3 = 8，高斯节点数 m = 4。
    """
    lin1 = nn.Linear(in_f, out_f)
    lin2 = nn.Linear(out_f, out_f)
    nn.init.xavier_uniform_(lin1.weight, gain=1.0)
    nn.init.zeros_(lin1.bias)
    nn.init.xavier_uniform_(lin2.weight, gain=1.0)
    nn.init.zeros_(lin2.bias)
    return nn.Sequential(
        lin1,
        PolyScaleNorm(out_f),   # 幅度软限制，放在激活之前
        PolyActivation(),       # 唯一的非线性，阶数 ×2
        lin2,                   # 第二线性，不加激活，保持阶数不再升
    )


# ──────────────────────────────────────────────────────────────────────────────
# PolyGINConv
# ──────────────────────────────────────────────────────────────────────────────
class PolyGINConv(GINConv):
    def __init__(self, in_f: int, out_f: int,
                 eps: float = 0., train_eps: bool = False, **kwargs):
        super().__init__(
            nn=_poly_mlp(in_f, out_f),
            eps=eps,
            train_eps=train_eps,
            **kwargs
        )


# ──────────────────────────────────────────────────────────────────────────────
# PolyGIN_3l
# ──────────────────────────────────────────────────────────────────────────────
class PolyGIN_3l(GNNBasic):
    """
    Pure-polynomial GIN, 3 layers (L=3).
    ┌─────────────────────────────────────────────────────┐
    │  激活：PolyActivation  f(x)=x+α·x²，无 ReLU        │
    │  归一化：PolyScaleNorm（软幅度限制），无 BN/LN      │
    │  整体阶数：2^4 = 16；归因高斯节点数：m = 5          │
    └─────────────────────────────────────────────────────┘
    """

    def __init__(self, model_level: str, dim_node: int,
                 dim_hidden: int, num_classes: int):
        super().__init__()
        num_layer = 3  # L = 3

        self.conv1 = PolyGINConv(dim_node, dim_hidden)
        self.convs = nn.ModuleList([
            PolyGINConv(dim_hidden, dim_hidden)
            for _ in range(num_layer - 1)
        ])

        if model_level == 'node':
            self.readout = IdenticalPool()
        else:
            self.readout = GlobalMeanPool()

        # FFN 分类头：Linear -> PolyAct -> Dropout -> Linear
        # 不加额外 PolyScaleNorm，让分类头梯度更自由
        ffn_lin1 = nn.Linear(dim_hidden, dim_hidden)
        ffn_lin2 = nn.Linear(dim_hidden, num_classes)
        nn.init.xavier_uniform_(ffn_lin1.weight, gain=1.0)
        nn.init.zeros_(ffn_lin1.bias)
        nn.init.xavier_uniform_(ffn_lin2.weight, gain=0.5)
        nn.init.zeros_(ffn_lin2.bias)

        self.ffn = nn.Sequential(
            ffn_lin1,
            PolyActivation(),
            nn.Dropout(p=0.5),
            ffn_lin2,
        )

    def forward(self, *args, **kwargs) -> Tensor:
        x, edge_index, batch = self.arguments_read(*args, **kwargs)
        post_conv = self.conv1(x, edge_index)
        for conv in self.convs:
            post_conv = conv(post_conv, edge_index)
        out_readout = self.readout(post_conv, batch)
        return self.ffn(out_readout)

    def get_emb(self, *args, **kwargs) -> Tensor:
        x, edge_index, batch = self.arguments_read(*args, **kwargs)
        post_conv = self.conv1(x, edge_index)
        for conv in self.convs:
            post_conv = conv(post_conv, edge_index)
        return post_conv
    
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

        edge_weight.requires_grad_(True)

        x = torch.matmul(x, self.weight)

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        if self.bias is not None:
            out += self.bias

        self.edge_weight = edge_weight

        return out

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        size = self._check_input(edge_index, size)

        if (isinstance(edge_index, SparseTensor) and self.fuse
                and not self._explain):
            coll_dict = self._collect(self._fused_user_args, edge_index,
                                      size, kwargs)

            out = self.message_and_aggregate(edge_index, **coll_dict)
            return self.update(out, **coll_dict)

        elif isinstance(edge_index, Tensor) or not self.fuse:
            coll_dict = self._collect(self._user_args, edge_index, size,
                                      kwargs)

            message_kwargs = {}
            if 'x_j' in coll_dict:
                message_kwargs['x_j'] = coll_dict['x_j']
            if 'edge_weight' in coll_dict:
                message_kwargs['edge_weight'] = coll_dict['edge_weight']
            
            out = self.message(**message_kwargs)
            
           
            if self._explain:
                edge_mask = self.__edge_mask__
     
                if out.size(self.node_dim) != edge_mask.size(0):
                    loop = edge_mask.new_ones(size[0])
                    edge_mask = torch.cat([edge_mask, loop], dim=0)
                assert out.size(self.node_dim) == edge_mask.size(0)
                out = out * edge_mask.view([-1] + [1] * (out.dim() - 1))
            elif self.__explain_flow__:
                edge_mask = self.layer_edge_mask
    
                if out.size(self.node_dim) != edge_mask.size(0):
                    loop = edge_mask.new_ones(size[0])
                    edge_mask = torch.cat([edge_mask, loop], dim=0)
                assert out.size(self.node_dim) == edge_mask.size(0)
                out = out * edge_mask.view([-1] + [1] * (out.dim() - 1))

            aggregate_kwargs = {k: v for k, v in coll_dict.items() 
                               if k in ['index', 'ptr', 'dim_size']}
            out = self.aggregate(out, **aggregate_kwargs)
            
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

        self.edge_weight = edge_weight

        return out

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        size = self._check_input(edge_index, size)

        if (isinstance(edge_index, SparseTensor) and self.fuse
                and not self._explain):
            coll_dict = self._collect(self._fused_user_args, edge_index,
                                         size, kwargs)

            msg_aggr_kwargs = coll_dict
            out = self.message_and_aggregate(edge_index, **msg_aggr_kwargs)

            update_kwargs = coll_dict
            return self.update(out, **update_kwargs)

        elif isinstance(edge_index, Tensor) or not self.fuse:
            coll_dict = self._collect(self._user_args, edge_index, size,
                                         kwargs)

            msg_kwargs = coll_dict
            out = self.message(**msg_kwargs)

            if self._explain:
                edge_mask = self.__edge_mask__
             
                if out.size(self.node_dim) != edge_mask.size(0):
                    loop = edge_mask.new_ones(size[0])
                    edge_mask = torch.cat([edge_mask, loop], dim=0)
                assert out.size(self.node_dim) == edge_mask.size(0)
                out = out * edge_mask.view([-1] + [1] * (out.dim() - 1))
            elif self.__explain_flow__:
                edge_mask = self.layer_edge_mask
       
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

        if (isinstance(edge_index, SparseTensor) and self.fuse
                and not self._explain):
            coll_dict = self._collect(self._fused_user_args, edge_index,
                                      size, kwargs)

            out = self.message_and_aggregate(edge_index, **coll_dict)
            return self.update(out)  

        elif isinstance(edge_index, Tensor) or not self.fuse:
            coll_dict = self._collect(self._user_args, edge_index, size,
                                      kwargs)

            msg_kwargs = {}
            if 'x_j' in coll_dict:
                msg_kwargs['x_j'] = coll_dict['x_j']
            if 'edge_weight' in coll_dict:
                msg_kwargs['edge_weight'] = coll_dict['edge_weight']
            
            out = self.message(**msg_kwargs)

            if self._explain:
                edge_mask = self.__edge_mask__
              
                if out.size(self.node_dim) != edge_mask.size(0):
                    loop = edge_mask.new_ones(size[0])
                    edge_mask = torch.cat([edge_mask, loop], dim=0)
                assert out.size(self.node_dim) == edge_mask.size(0)
                out = out * edge_mask.view([-1] + [1] * (out.dim() - 1))
            elif self.__explain_flow__:
                edge_mask = self.layer_edge_mask
         
                if out.size(self.node_dim) != edge_mask.size(0):
                    loop = edge_mask.new_ones(size[0])
                    edge_mask = torch.cat([edge_mask, loop], dim=0)
                assert out.size(self.node_dim) == edge_mask.size(0)
                out = out * edge_mask.view([-1] + [1] * (out.dim() - 1))

            aggr_kwargs = {k: v for k, v in coll_dict.items() 
                          if k in ['index', 'ptr', 'dim_size']}
            out = self.aggregate(out, **aggr_kwargs)

            return self.update(out)
        