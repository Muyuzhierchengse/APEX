import os
import os.path as osp
import torch
import numpy as np
import random
from apex.models.gnn import *
from apex.data.loaders import *
from apex.utils.paths import CHECKPOINT_DIR, DATA_DIR, OUTPUT_DIR
from apex.utils.reproducibility import set_seed
import argparse
import torch.nn as nn
from torch.nn.functional import cross_entropy
from torch_geometric.loader import DataLoader
import functools
torch.load = functools.partial(torch.load, weights_only=False)
import time
from datetime import datetime
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import roc_auc_score

# ───────────────────────────────────────────────
#  Focal Loss
# ───────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    alpha : per-class weight tensor  (same shape as class_weight in CrossEntropy)
    gamma : focusing parameter, 0 => 退化为普通 CE
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha      # Tensor[num_classes] or None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce = cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)                         # p_t
        loss = (1 - pt) ** self.gamma * ce
        return loss.mean() if self.reduction == 'mean' else loss.sum()


# ───────────────────────────────────────────────
#  工具函数
# ───────────────────────────────────────────────
def accuracy(pred, y):
    return (pred == y).sum() / y.shape[0]


def accuracy_dataloader(device, model, dataloader):
    pred, y = [], []
    for data in dataloader:
        data = data.to(device)
        pred.append(model(data.x, data.edge_index, data.batch).argmax(dim=1))
        y.append(data.y.view(-1))
    pred = torch.cat(pred, dim=0)
    y = torch.cat(y, dim=0)
    return (pred == y).sum() / y.shape[0]

def auc_dataloader(device, model, dataloader):
    """返回 ROC-AUC，若只有一个类别则返回 None。"""
    probs, y = [], []
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            logits = model(data.x, data.edge_index, data.batch)
            prob = torch.softmax(logits, dim=1)[:, 1]   # 正类概率
            probs.append(prob.cpu())
            y.append(data.y.view(-1).cpu())
    probs = torch.cat(probs).numpy()
    y = torch.cat(y).numpy()
    if len(set(y)) < 2:          # batch 内只有一个类别时 AUC 无意义
        return None
    return roc_auc_score(y, probs)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ───────────────────────────────────────────────
#  计算类别权重  (inverse frequency)
# ───────────────────────────────────────────────
def compute_class_weights(dataset, num_classes, device):
    """
    返回 Tensor[num_classes]，权重 = total / (num_classes * count_i)
    适用于 WeightedCE 和 FocalLoss 的 alpha。
    """
    counts = torch.zeros(num_classes)
    for data in dataset:
        counts[data.y.item()] += 1
    counts = counts.clamp(min=1)                    # 防止除零
    weights = counts.sum() / (num_classes * counts)
    return weights.to(device)


# ───────────────────────────────────────────────
#  构建 WeightedRandomSampler（每个样本的采样概率）
# ───────────────────────────────────────────────
def build_weighted_sampler(dataset, num_classes):
    """
    少数类被更频繁采样，使每个 batch 中各类数量趋于均衡。
    """
    counts = torch.zeros(num_classes)
    for data in dataset:
        counts[data.y.item()] += 1
    counts = counts.clamp(min=1)
    class_weight = 1.0 / counts                     # 频率越低权重越高

    sample_weights = torch.tensor(
        [class_weight[data.y.item()].item() for data in dataset]
    )
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler


# ───────────────────────────────────────────────
#  过采样：复制少数类样本直到各类数量与多数类持平
# ───────────────────────────────────────────────
def oversample_dataset(dataset, num_classes):
    """
    返回新的列表，少数类样本被重复复制（整数倍上取整），
    最终每类样本数 ≈ 多数类数量。
    """
    # 按类别分桶
    buckets = [[] for _ in range(num_classes)]
    for data in dataset:
        buckets[data.y.item()].append(data)

    max_count = max(len(b) for b in buckets)
    balanced = []
    for bucket in buckets:
        if len(bucket) == 0:
            continue
        repeat = (max_count + len(bucket) - 1) // len(bucket)  # ceil
        balanced.extend((bucket * repeat)[:max_count])

    random.shuffle(balanced)
    return balanced
def undersample_dataset(dataset, num_classes):
    """
    返回新的列表，多数类样本被随机丢弃，
    最终每类样本数 ≈ 少数类数量。
    """
    buckets = [[] for _ in range(num_classes)]
    for data in dataset:
        buckets[data.y.item()].append(data)

    min_count = min(len(b) for b in buckets if len(b) > 0)
    balanced = []
    for bucket in buckets:
        if len(bucket) == 0:
            continue
        balanced.extend(random.sample(bucket, min_count))

    random.shuffle(balanced)
    return balanced

# ───────────────────────────────────────────────
#  主函数
# ───────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Graph-Twitter',
                        choices=['BBBP', 'ClinTox', 'Graph-SST2', 'Graph-Twitter',
                                 'BA_2Motifs', 'BACE', 'Tox21', 'ToxCast','MUTAG', 
                                 'Mutagenicity', 'Spurious-Motif', 'NCI1','ogbg-molhiv','ogbg-molpcba','ogbn-proteins'])
    parser.add_argument('--model_used', type=str, default='GIN_3l',
                        choices=['GCN_2l', 'GCN_3l', 'GIN_2l', 'GIN_3l', 'PolyGIN_3l'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--dim_hidden', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-3)

    # ── 类别不平衡处理策略（四选一，默认 weighted_ce）──
    parser.add_argument('--imbalance_strategy', type=str, default='weighted_ce',
                        choices=['none', 'weighted_ce', 'focal', 'oversample', 'weighted_sampler', 'undersample'],
                        help=(
                            'none            : 不做任何处理\n'
                            'weighted_ce     : 加权交叉熵\n'
                            'focal           : Focal Loss\n'
                            'oversample      : 过采样少数类\n'
                            'weighted_sampler: WeightedRandomSampler\n'
                        ))
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal Loss 的 gamma 参数（仅 focal 策略有效）')

    args = parser.parse_args()

    # ── 路径 ──
    data_path = str(DATA_DIR)
    checkpoint_path = osp.join(str(CHECKPOINT_DIR), args.dataset)
    os.makedirs(checkpoint_path, exist_ok=True)
    output_path = str(OUTPUT_DIR)
    os.makedirs(output_path, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'{args.dataset}_{args.model_used}_{timestamp}.txt'
    log_filepath = osp.join(output_path, log_filename)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data, num_nodes, dim_node, num_classes = load_dataset(data_path, args.dataset)

    with open(log_filepath, 'w', encoding='utf-8') as log_file:

        def log(msg, end='\n'):
            print(msg, end=end)
            log_file.write(msg + end)

        log(f'实验配置\n{"="*80}')
        log(f'数据集: {args.dataset}')
        log(f'模型: {args.model_used}')
        log(f'训练轮数: {args.epochs}')
        log(f'隐藏层维度: {args.dim_hidden}')
        log(f'学习率: {args.lr}')
        log(f'设备: {device}')
        log(f'不平衡策略: {args.imbalance_strategy}')
        log(f'{"="*80}\n')

        for seed in range(1):
            log(f'\n{"="*80}\n随机种子 {seed} 的训练结果\n{"="*80}')
            set_seed(seed)
            model_save_path = osp.join(checkpoint_path,
                                       args.model_used + f'_seed{seed}.pkl')

            # ── 节点级任务（原逻辑不变）──
            if args.dataset in ['BA_shapes','ogbn-proteins']:
                model_level = 'node'
                model = eval(args.model_used)(
                    model_level=model_level, dim_node=dim_node,
                    dim_hidden=args.dim_hidden, num_classes=num_classes
                ).to(device)
                log(f'模型参数量: {count_parameters(model):,}')

                optimizer = torch.optim.Adam(
                    model.parameters(), lr=args.lr, weight_decay=5e-4)
                data = data.to(device)
                epoch_times = []

                for epoch in range(1, args.epochs + 1):
                    t0 = time.time()
                    model.train()
                    optimizer.zero_grad()
                    pred = model(data.x, data.edge_index)
                    loss = cross_entropy(pred[data.train_mask],
                                        data.y[data.train_mask])
                    loss.backward()
                    optimizer.step()

                    if epoch % 10 == 0:
                        model.eval()
                        pred = model(data.x, data.edge_index).argmax(dim=1)
                        log(f'epoch #{epoch:3d}, loss = {loss:.4f}, '
                            f'train_acc = {accuracy(pred[data.train_mask], data.y[data.train_mask]):.4f}, '
                            f'valid_acc = {accuracy(pred[data.val_mask], data.y[data.val_mask]):.4f}, '
                            f'test_acc = {accuracy(pred[data.test_mask], data.y[data.test_mask]):.4f}')
                    epoch_times.append(time.time() - t0)

                log(f'平均每轮训练时间: {sum(epoch_times)/len(epoch_times):.4f}秒')
                torch.save(model.state_dict(), model_save_path)

            # ── 图级任务 ──
            else:
                model_level = 'graph'
                model = eval(args.model_used)(
                    model_level=model_level, dim_node=dim_node,
                    dim_hidden=args.dim_hidden, num_classes=num_classes
                ).to(device)
                log(f'模型参数量: {count_parameters(model):,}')

                optimizer = torch.optim.Adam(
                    model.parameters(), lr=args.lr, weight_decay=5e-6)

                # ── 训练集预处理（过采样 / 正常）──
                train_data = data['train']
                strategy = args.imbalance_strategy

                if strategy == 'oversample':
                    train_data = oversample_dataset(train_data, num_classes)
                    log(f'过采样后训练集大小: {len(train_data)}')
                if strategy == 'undersample':                             
                    train_data = undersample_dataset(train_data, num_classes)
                    log(f'欠采样后训练集大小: {len(train_data)}')
                # ── 构建 DataLoader ──
                if strategy == 'weighted_sampler':
                    sampler = build_weighted_sampler(train_data, num_classes)
                    train_loader = DataLoader(train_data, batch_size=32,
                                             sampler=sampler)
                    log('使用 WeightedRandomSampler')
                else:
                    train_loader = DataLoader(train_data, batch_size=32,
                                             shuffle=True)

                valid_loader = DataLoader(data['val'], batch_size=32, shuffle=False)
                test_loader  = DataLoader(data['test'], batch_size=32, shuffle=False)

                # ── 计算类别权重（weighted_ce / focal 需要）──
                class_weights = None
                if strategy in ('weighted_ce', 'focal'):
                    class_weights = compute_class_weights(
                        data['train'], num_classes, device)
                    log(f'类别权重: {class_weights.cpu().tolist()}')

                # ── 构建损失函数 ──
                if strategy == 'weighted_ce':
                    criterion = nn.CrossEntropyLoss(weight=class_weights)
                elif strategy == 'focal':
                    criterion = FocalLoss(alpha=class_weights,
                                         gamma=args.focal_gamma)
                else:
                    criterion = nn.CrossEntropyLoss()   # none / oversample / weighted_sampler

                # ── 统计数据集信息 ──
                total_nodes = sum(d.num_nodes for loader in
                                  [train_loader, valid_loader, test_loader]
                                  for d in loader)
                log(f'总图数量: {len(train_loader)+len(valid_loader)+len(test_loader)}, '
                    f'总节点数: {total_nodes}')

                # ── 打印训练集类别分布 ──
                counts = torch.zeros(num_classes, dtype=torch.long)
                for d in data['train']:
                    counts[d.y.item()] += 1
                log(f'训练集类别分布: { {i: counts[i].item() for i in range(num_classes)} }')

                # ── 训练循环 ──
                best = 0
                epoch_times = []
                for epoch in range(1, args.epochs + 1):
                    t0 = time.time()
                    model.train()
                    total_loss = 0

                    for batch in train_loader:
                        optimizer.zero_grad()
                        batch = batch.to(device)
                        logits = model(batch.x, batch.edge_index, batch.batch)
                        loss = criterion(logits, batch.y.view(-1))
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item() * batch.num_graphs

                    model.eval()
                    acc_train = accuracy_dataloader(device, model, train_loader)
                    acc_val   = accuracy_dataloader(device, model, valid_loader)
                    acc_test  = accuracy_dataloader(device, model, test_loader)

                    auc_train = auc_dataloader(device, model, train_loader)
                    auc_val   = auc_dataloader(device, model, valid_loader)
                    auc_test  = auc_dataloader(device, model, test_loader)

                    auc_str = (
                        f'train_auc = {auc_train:.4f}, '
                        f'valid_auc = {auc_val:.4f}, '
                        f'test_auc = {auc_test:.4f}'
                    ) if auc_test is not None else 'AUC = N/A (单类别batch)'

                    log(f'epoch #{epoch:3d}, '
                        f'loss = {total_loss / len(train_loader):.4f}, '
                        f'train_acc = {acc_train:.4f}, '
                        f'valid_acc = {acc_val:.4f}, '
                        f'test_acc = {acc_test:.4f}, '
                        + auc_str)

                    # 用 AUC 而非 acc 来保存最优模型（AUC 更可靠）
                    score = auc_test if auc_test is not None else acc_test.item()
                    if epoch > args.epochs // 2 and score >= best:
                        best = score
                        torch.save(model.state_dict(), model_save_path)

                    epoch_times.append(time.time() - t0)

                log(f'平均每轮训练时间: {sum(epoch_times)/len(epoch_times):.4f}秒')
                log(f'最佳测试准确率: {best:.4f}')

            log_file.flush()

    print(f'\n训练日志已保存到: {log_filepath}')


if __name__ == '__main__':
    main()
