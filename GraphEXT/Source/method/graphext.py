import torch
import numpy as np
import random


class GraphEXT(torch.nn.Module):
    def __init__(self, model, explain_graph=False):
        super().__init__()
        self.model = model
        self.explain_graph = explain_graph
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def random_generate_partition(self, n):
        if self.explain_graph:
            permutation = [i for i in range(0, n)]
            for i in range(n):
                pos = random.randint(0, n - 1)
                permutation[i], permutation[pos] = permutation[pos], permutation[i]
        else:
            permutation = [i for i in range(1, n)]
            for i in range(n - 1):
                pos = random.randint(0, n - 2)
                permutation[i], permutation[pos] = permutation[pos], permutation[i]

        partition = []
        belong = [-1] * n
        cnt = -1
        if self.explain_graph:
            for i in range(0, n):
                if belong[permutation[i]] != -1:
                    continue
                cnt += 1
                nw = permutation[i]
                coal = [nw]
                belong[nw] = cnt
                nw = permutation[nw]
                while nw != permutation[i]:
                    coal.append(nw)
                    belong[nw] = cnt
                    nw = permutation[nw]
                partition.append(coal)
        else:
            for i in range(1, n):
                if belong[permutation[i - 1]] != -1:
                    continue
                cnt += 1
                nw = permutation[i - 1]
                coal = [nw]
                belong[nw] = cnt
                nw = permutation[nw - 1]
                while nw != permutation[i - 1]:
                    coal.append(nw)
                    belong[nw] = cnt
                    nw = permutation[nw - 1]
                partition.append(coal)

        return partition, belong

    def calc_partition_value(self, belong, S, model, num_nodes, edge_index, x, num_classes):
        def extract_edge_from_partition(R, ori_edge):
            if not self.explain_graph:
                bel = [1] + [0] * (num_nodes - 1)
                Flg, cnt = 0, 1
            else:
                bel = [0] * num_nodes
                Flg, cnt = 0, 0
            ys = [0] * num_nodes
            for node in R:
                bel[node] = 1
                ys[node] = cnt
                cnt = cnt + 1

            extracted_edge = [[], []]
            for i in range(len(ori_edge[0])):
                src, dst = ori_edge[0][i], ori_edge[1][i]
                if bel[src] != 1 or bel[dst] != 1:
                    continue
                extracted_edge[0].append(ys[src])
                extracted_edge[1].append(ys[dst])
                if src == 0 or dst == 0:
                    Flg = 1

            return Flg, torch.LongTensor(extracted_edge).to(self.device)

        def extract_component_from_coalition(num_nodes, S, ori_edge):
            component_list = []
            out_edges = [[] for _ in range(num_nodes)]
            for i in range(len(ori_edge[0])):
                src, dst = ori_edge[0][i], ori_edge[1][i]
                out_edges[src].append(dst)

            col, vis = [0] * num_nodes, [0] * num_nodes
            for u in S:
                col[u] = 1

            for i in range(num_nodes):
                if col[i] != 1 or vis[i] != 0:
                    continue
                vis[i] = 1
                component = [i]
                queue = [i]
                while len(queue) != 0:
                    u = queue.pop()
                    for v in out_edges[u]:
                        if col[v] == 1 and vis[v] == 0:
                            queue.append(v)
                            component.append(v)
                            vis[v] = 1

                component_list.append(component)

            return component_list

        components = extract_component_from_coalition(num_nodes, S, edge_index.tolist())
        value = torch.zeros(num_classes).to(self.device)
        if not self.explain_graph:
            for R in components:
                flag, edges = extract_edge_from_partition(R, edge_index.tolist())
                value = value + model(x[[0] + R], edges)[0]
        else:
            for R in components:
                flag, edges = extract_edge_from_partition(R, edge_index.tolist())
                value = value + model(x[R], edges)[0]

        return torch.softmax(value / len(components), dim=0)

    def sv_str(self, model, num_nodes, edge_index, x, num_classes, T=100):
        sv_node = torch.zeros(num_nodes, num_classes).to(self.device)
        model.eval()
        with torch.no_grad():
            for _ in range(T):
                P, belong = self.random_generate_partition(num_nodes)
                P.append([])
                if self.explain_graph:
                    permutation = [i for i in range(0, num_nodes)]
                    for i in range(num_nodes):
                        pos = random.randint(0, num_nodes - 1)
                        permutation[i], permutation[pos] = permutation[pos], permutation[i]
                else:
                    permutation = [i for i in range(1, num_nodes)]
                    for i in range(num_nodes - 1):
                        pos = random.randint(0, num_nodes - 2)
                        permutation[i], permutation[pos] = permutation[pos], permutation[i]

                las = torch.zeros(num_classes).to(self.device)
                for u in permutation:
                    P[len(P) - 1].append(u)
                    for i in range(len(P[belong[u]])):
                        if P[belong[u]][i] == u:
                            P[belong[u]].pop(i)
                            break
                    belong[u] = len(P) - 1
                    now = self.calc_partition_value(belong, P[len(P) - 1], model, num_nodes, edge_index, x, num_classes)
                    sv_node[u] += now - las
                    las = now

        return sv_node

    def forward(self, x, edge_index, **kwargs):
        num_classes = kwargs.get('num_classes')
        num_nodes = x.shape[0]
        num_edges = edge_index.shape[1]

        sv_node = self.sv_str(self.model, num_nodes, edge_index, x, num_classes)

        self.last_node_scores = [sv_node[:, cls].detach() for cls in range(num_classes)]

        edge_masks = []
        for cls in range(num_classes):
            scores = sv_node[:, cls].detach()
            src = edge_index[0]
            dst = edge_index[1]
            edge_score = scores[src] + scores[dst]
            edge_masks.append(edge_score)

        return edge_masks