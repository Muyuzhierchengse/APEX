import os
import os.path as osp
import torch
import numpy as np
import random
from model.models import *
from dataset.data import *
import argparse
from torch.nn.functional import cross_entropy
from torch_geometric.loader import DataLoader
import functools
torch.load = functools.partial(torch.load, weights_only=False)
import time
from datetime import datetime

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def accuracy(pred, y):
    return (pred == y).sum() / y.shape[0]

def accuracy_dataloader(device, model, dataloader):
    pred, y = [], []
    for data in dataloader:
        data = data.to(device)
        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        pred.append(model(x, edge_index, batch).argmax(dim=1))
        y.append(data.y.view(-1))

    pred = torch.cat(pred, dim=0)
    y = torch.cat(y, dim=0)
    return (pred == y).sum() / y.shape[0]

def eval_dataloader(device, model, dataloader):
    logits_list, y = [], []
    for data in dataloader:
        data = data.to(device)
        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        logits_list.append(model(x, edge_index, batch))
        y.append(data.y.view(-1))

    logits = torch.cat(logits_list, dim=0)
    y = torch.cat(y, dim=0)
    acc = (logits.argmax(dim=1) == y).sum() / y.shape[0]
    loss = cross_entropy(logits, y).item()
    return acc, loss

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Trains the chosen model across multiple random seeds and keeps the checkpoint
# with the best validation accuracy (ties broken by lower validation loss).
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='BBBP', 
                        choices=[ 'BBBP', 'Graph-SST2','BACE', 'Mutagenicity', 'BA_shapes'])
    parser.add_argument('--model_used', type=str, default='GIN',
                        choices=['GCN_2l', 'GCN', 'GIN_2l', 'GIN','PolyGIN'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--dim_hidden', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-3)#5e-5
    args = parser.parse_args()

    data_path = './dataset'
    checkpoint_path = osp.join('model', 'checkpoint', args.dataset)
    if not osp.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    
    output_path = './output'
    if not osp.exists(output_path):
        os.makedirs(output_path)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'{args.dataset}_{args.model_used}_{timestamp}.txt'
    log_filepath = osp.join(output_path, log_filename)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data, num_nodes, dim_node, num_classes = load_dataset(data_path, args.dataset)

    with open(log_filepath, 'w', encoding='utf-8') as log_file:

        log_file.write(f'Experiment Config\n')
        log_file.write(f'{"="*80}\n')
        log_file.write(f'Dataset: {args.dataset}\n')
        log_file.write(f'Model: {args.model_used}\n')
        log_file.write(f'Epochs: {args.epochs}\n')
        log_file.write(f'Hidden dim: {args.dim_hidden}\n')
        log_file.write(f'Learning rate: {args.lr}\n')
        log_file.write(f'Device: {device}\n')
        log_file.write(f'{"="*80}\n\n')
        
        best_accs = []
        num_seeds = 10
        for seed in range(num_seeds):
            separator = f'\n{"="*80}\n'
            seed_info = f'Training results for random seed {seed}\n'
            print(separator + seed_info + "="*80)
            log_file.write(separator + seed_info + "="*80 + '\n')
            
            set_seed(seed)
            model_save_path = osp.join(checkpoint_path, args.model_used + f'_seed{seed}.pkl')

            if args.dataset in ['BA_shapes']:
                model_level = 'node'
                model = eval(args.model_used)(model_level=model_level, dim_node=dim_node,
                                        dim_hidden=args.dim_hidden, num_classes=num_classes).to(device)
                param_info = f'Model parameters: {count_parameters(model):,}\n'
                print(param_info)
                log_file.write(param_info)

                optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=5e-6)
                data = data.to(device)
                epoch_times = []
                best = 0
                best_val_acc = -1.0
                best_val_loss = float('inf')

                for epoch in range(1, args.epochs + 1):
                    start_time = time.time()
                    model.train()
                    optimizer.zero_grad()
                    logits = model(data.x, data.edge_index)
                    loss = cross_entropy(logits[data.train_mask], data.y[data.train_mask])
                    loss.backward()
                    optimizer.step()

                    model.eval()
                    with torch.no_grad():
                        logits = model(data.x, data.edge_index)
                        pred = logits.argmax(dim=1)
                        acc_test = accuracy(pred[data.test_mask], data.y[data.test_mask])
                        acc_val = accuracy(pred[data.val_mask], data.y[data.val_mask])
                        loss_val = cross_entropy(logits[data.val_mask], data.y[data.val_mask]).item()

                    if epoch % 10 == 0:
                        output = (f'epoch #{epoch:3d}, loss = {loss:.4f}, '
                                f'train_acc = {accuracy(pred[data.train_mask], data.y[data.train_mask]):.4f}, '
                                f'valid_acc = {acc_val:.4f}, '
                                f'test_acc = {acc_test:.4f}')
                        print(output)
                        log_file.write(output + '\n')

                    if epoch > args.epochs // 2:
                        acc_val_f = float(acc_val)
                        is_better = (acc_val_f > best_val_acc) or (
                            acc_val_f == best_val_acc and loss_val < best_val_loss)
                        if is_better:
                            best_val_acc = acc_val_f
                            best_val_loss = loss_val
                            best = acc_test
                            torch.save(model.state_dict(), model_save_path)

                    epoch_times.append(time.time() - start_time)

                avg_time = f'Avg time per epoch: {sum(epoch_times) / len(epoch_times):.4f}s\n'
                best_acc = (f'Best val accuracy: {best_val_acc:.4f}, '
                            f'corresponding test accuracy: {best:.4f}\n')
                print(avg_time + best_acc)
                log_file.write(avg_time + best_acc)
                best_accs.append(float(best))

            # Graph classification datasets are split into mini-batches via DataLoader.
            else:
                model_level = 'graph'
                model = eval(args.model_used)(model_level=model_level, dim_node=dim_node,
                                    dim_hidden=args.dim_hidden, num_classes=num_classes).to(device)
                param_info = f'Model parameters: {count_parameters(model):,}\n'
                print(param_info)
                log_file.write(param_info)

                optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=5e-6)#5e-4 6

                train_loader = DataLoader(data['train'], batch_size=32, shuffle=True)#batch16、32，epoch1000.当梯度有点爆炸的时候换成1
                valid_loader = DataLoader(data['val'], batch_size=32, shuffle=True)
                test_loader = DataLoader(data['test'], batch_size=32, shuffle=True)

                total_nodes = 0
                for data_item in train_loader:
                    total_nodes += data_item.num_nodes
                for data_item in valid_loader:
                    total_nodes += data_item.num_nodes
                for data_item in test_loader:
                    total_nodes += data_item.num_nodes
                
                dataset_info = f'Total graphs: {len(train_loader) + len(valid_loader) + len(test_loader)}, total nodes: {total_nodes}\n'
                print(dataset_info.strip())
                log_file.write(dataset_info)

                best = 0
                best_val_acc = -1.0
                best_val_loss = float('inf')
                epoch_times = []
                for epoch in range(1, args.epochs + 1):
                    start_time = time.time()
                    model.train()
                    total_loss = 0
                    for data_item in train_loader:
                        optimizer.zero_grad()
                        data_item = data_item.to(device)
                        x = data_item.x
                        edge_index = data_item.edge_index
                        batch = data_item.batch

                        logits = model(x, edge_index, batch)
                        loss = cross_entropy(logits, data_item.y.view(-1)).to(device)
                        loss.backward()
                        optimizer.step()

                        total_loss += loss * data_item.num_graphs

                    model.eval()
                    with torch.no_grad():
                        acc_val, loss_val = eval_dataloader(device, model, valid_loader)
                        acc_test = accuracy_dataloader(device, model, test_loader)
                    output = (f'epoch #{epoch:3d}, loss = {total_loss / len(train_loader):.4f}, '
                            f'train_acc = {accuracy_dataloader(device, model, train_loader):.4f}, '
                            f'valid_acc = {acc_val:.4f}, '
                            f'test_acc = {acc_test:.4f}')
                    print(output)
                    log_file.write(output + '\n')

                    if epoch > args.epochs // 2:
                        acc_val_f = float(acc_val)
                        is_better = (acc_val_f > best_val_acc) or (
                            acc_val_f == best_val_acc and loss_val < best_val_loss)
                        if is_better:
                            best_val_acc = acc_val_f
                            best_val_loss = loss_val
                            best = acc_test
                            torch.save(model.state_dict(), model_save_path)
                    epoch_times.append(time.time() - start_time)

                avg_time = f'Avg time per epoch: {sum(epoch_times) / len(epoch_times):.4f}s\n'
                best_acc = (f'Best val accuracy: {best_val_acc:.4f}, '
                            f'corresponding test accuracy: {best:.4f}\n')
                print(avg_time + best_acc)
                log_file.write(avg_time + best_acc)
                best_accs.append(float(best))

            log_file.flush()

        best_accs_pct = [acc * 100 for acc in best_accs]
        mean_acc = np.mean(best_accs_pct)
        std_acc = np.std(best_accs_pct)
        max_acc = np.max(best_accs_pct)
        min_acc = np.min(best_accs_pct)

        summary = (
            f'\n{"="*80}\n'
            f'Summary of best test accuracy across {num_seeds} random seeds\n'
            f'{"="*80}\n'
            f'Best test accuracy per seed: {[f"{acc:.2f}" for acc in best_accs_pct]}\n'
            f'Max: {max_acc:.2f}\n'
            f'Min: {min_acc:.2f}\n'
            f'Mean±std: {mean_acc:.2f}±{std_acc:.2f}\n'
            f'{"="*80}\n'
        )
        print(summary)
        log_file.write(summary)

    print(f'\nTraining log saved to: {log_filepath}')

if __name__ == '__main__':
    main()