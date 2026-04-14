import torch
import torch.nn as nn
import torch.nn.functional as F


class BehaviorAwareGCNLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W           = nn.Linear(dim, dim, bias=False)
        self.self_linear = nn.Linear(dim, dim, bias=False)
        nn.init.xavier_normal_(self.W.weight)
        nn.init.xavier_normal_(self.self_linear.weight)

        # asymmetric gate
        self.alpha     = nn.Parameter(torch.tensor(1.0))
        self.beta      = nn.Parameter(torch.tensor(1.0))
        self.alpha_self = nn.Parameter(torch.tensor(1.0))
        # learnable temperature
        self.temp      = nn.Parameter(torch.tensor(2.0))

    def forward(self, x, edge_index, sim_weight, rep, node_signal):
        row, col = edge_index

        h_j = self.W(x)[col]

        # gate = 신뢰도만 (rep 기반), asymmetric + temperature
        gate_input = self.alpha * rep[row] + self.beta * rep[col]
        gate = torch.sigmoid(gate_input / self.temp).unsqueeze(-1)

        # sim: row-wise normalize (이웃 간 상대적 중요도)
        sim_norm = torch.zeros_like(sim_weight)
        sim_norm.scatter_add_(0, row, sim_weight)
        sim = (sim_weight / (sim_norm[row] + 1e-6)).unsqueeze(-1)

        # signal: 원래 철학 유지 — negative는 repulsion, positive는 attraction
        s_j = torch.tanh(node_signal[col]).unsqueeze(-1)  # tanh로 bounded
        msg = sim * gate * s_j * h_j

        out = torch.zeros_like(x)
        out.scatter_add_(0, row.unsqueeze(-1).expand_as(msg), msg)

        # weighted degree normalization
        deg = torch.zeros(x.size(0), device=x.device)
        deg.scatter_add_(0, row, gate.squeeze(-1))
        out = out / (deg.unsqueeze(-1) + 1e-6)

        # self connection (신뢰도만)
        gate_self = torch.sigmoid(self.alpha_self * rep / self.temp).unsqueeze(-1)
        out = out + gate_self * self.self_linear(x)

        return F.leaky_relu(out)


class BehaviorAwareGCN(nn.Module):
    def __init__(self, dim, num_layers=1):
        super().__init__()
        self.layers = nn.ModuleList([
            BehaviorAwareGCNLayer(dim) for _ in range(num_layers)
        ])

    def forward(self, x, edge_index, sim_weight, rep, node_signal):
        for layer in self.layers:
            x = layer(x, edge_index, sim_weight, rep, node_signal)
        return x


class TimeAwarePE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(1, dim)

    def forward(self, timestamps):
        t_min = timestamps.min(dim=1, keepdim=True).values
        t_max = timestamps.max(dim=1, keepdim=True).values
        timestamps_norm = (timestamps - t_min) / (t_max - t_min + 1e-8)
        return self.proj(timestamps_norm.unsqueeze(-1))


class SequenceEncoder(nn.Module):
    def __init__(self, dim, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.time_pe = TimeAwarePE(dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.encoder   = nn.TransformerEncoder(encoder_layer, num_layers)
        self.norm      = nn.LayerNorm(dim)
        self.attn_pool = nn.Linear(dim, 1)

    def forward(self, x, timestamps):
        if x.size(1) == 1:
            return x.squeeze(1)
        pe = self.time_pe(timestamps)
        x  = x + pe
        x  = self.encoder(x)
        x  = self.norm(x)
        scores   = self.attn_pool(x).squeeze(-1)
        scores   = F.softmax(scores, dim=-1).unsqueeze(-1)
        user_emb = (x * scores).sum(dim=1)
        return user_emb


class CrossAttributeAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.dim       = dim
        self.W_q      = nn.Linear(dim, dim, bias=False)
        self.W_k      = nn.Linear(dim, dim, bias=False)
        self.W_v      = nn.Linear(dim, dim, bias=False)
        self.out_proj  = nn.Linear(dim, dim, bias=False)
        for w in [self.W_q, self.W_k, self.W_v, self.out_proj]:
            nn.init.xavier_normal_(w.weight)

    def forward(self, u_title, u_category):
        B, H, D = u_title.shape[0], self.num_heads, self.head_dim
        nodes = torch.stack([u_title, u_category], dim=1)
        Q = self.W_q(nodes).view(B, 2, H, D).permute(0, 2, 1, 3)
        K = self.W_k(nodes).view(B, 2, H, D).permute(0, 2, 1, 3)
        V = self.W_v(nodes).view(B, 2, H, D).permute(0, 2, 1, 3)
        attn = F.softmax(torch.matmul(Q, K.transpose(-2, -1)) / (D ** 0.5), dim=-1)
        out  = torch.matmul(attn, V)
        out  = out.permute(0, 2, 1, 3).contiguous().view(B, 2, self.dim)
        out  = self.out_proj(out) + nodes
        return out[:, 0, :], out[:, 1, :]


class MicroVideoRec(nn.Module):
    def __init__(self, num_items, title_feat, category_feat, dim=64,
                 num_heads=4, num_transformer_layers=2, num_gcn_layers=1,
                 dropout=0.1):
        super().__init__()
        self.num_items = num_items
        self.dim       = dim

        self.register_buffer('title_feat',    torch.tensor(title_feat,    dtype=torch.float))
        self.register_buffer('category_feat', torch.tensor(category_feat, dtype=torch.float))

        self.title_proj    = nn.Linear(title_feat.shape[1],    dim)
        self.category_proj = nn.Linear(category_feat.shape[1], dim)

        self.title_gcn    = BehaviorAwareGCN(dim, num_gcn_layers)
        self.category_gcn = BehaviorAwareGCN(dim, num_gcn_layers)

        self.title_transformer    = SequenceEncoder(dim, num_heads, num_transformer_layers, dropout)
        self.category_transformer = SequenceEncoder(dim, num_heads, num_transformer_layers, dropout)

        self.cross_attention = CrossAttributeAttention(dim, num_heads)

        # learnable lambda for signal aggregation
        self.lam_raw = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5

    def _aggregate_signals(self, item_ids_t, signals_t, reps_t, device):
        """signal = mean + λ·max_abs, rep = log1p + z-score"""
        signal_mean   = torch.zeros(self.num_items, dtype=torch.float, device=device)
        signal_maxabs = torch.zeros(self.num_items, dtype=torch.float, device=device)
        rep_full      = torch.zeros(self.num_items, dtype=torch.float, device=device)
        count_full    = torch.zeros(self.num_items, dtype=torch.float, device=device)

        for i, sig, rep in zip(item_ids_t, signals_t, reps_t):
            idx = i.item()
            signal_mean[idx]  += sig
            rep_full[idx]     += rep
            count_full[idx]   += 1
            if sig.abs() > signal_maxabs[idx].abs():
                signal_maxabs[idx] = sig

        mask = count_full > 0
        signal_mean[mask] /= count_full[mask]
        rep_full[mask]    /= count_full[mask]

        lam         = torch.sigmoid(self.lam_raw)
        signal_full = signal_mean + lam * signal_maxabs

        rep_log  = torch.log1p(rep_full)
        rep_mean = rep_log.mean()
        rep_std  = rep_log.std() + 1e-6
        rep_scaled = (rep_log - rep_mean) / rep_std

        return signal_full, rep_scaled

    def forward_user(self, user_sequence, rep_dict, graph_structure, device):
        item_ids     = []
        node_signals = []
        reps         = []
        timestamps   = []

        for interaction in user_sequence:
            i = interaction['item_id'] - 1
            u = interaction['user_id']
            item_ids.append(i)
            node_signals.append(interaction['node_signal'])
            reps.append(float(rep_dict.get((u, i), 0)))
            timestamps.append(interaction['timestamp'])

        if len(item_ids) == 0:
            base_title    = self.title_proj(self.title_feat)
            base_category = self.category_proj(self.category_feat)
            return base_title.mean(0), base_category.mean(0), base_title, base_category

        ids_t  = torch.tensor(item_ids,     dtype=torch.long,  device=device)
        sigs_t = torch.tensor(node_signals, dtype=torch.float, device=device)
        reps_t = torch.tensor(reps,         dtype=torch.float, device=device)
        ts_t   = torch.tensor(timestamps,   dtype=torch.float, device=device)

        signal_full, rep_scaled = self._aggregate_signals(ids_t, sigs_t, reps_t, device)

        base_title    = self.title_proj(self.title_feat)
        base_category = self.category_proj(self.category_feat)

        user_title = self.title_gcn(
            base_title,
            graph_structure['title']['edge_index'].to(device),
            graph_structure['title']['sim_weight'].to(device),
            rep_scaled, signal_full,
        )
        user_category = self.category_gcn(
            base_category,
            graph_structure['category']['edge_index'].to(device),
            graph_structure['category']['sim_weight'].to(device),
            rep_scaled, signal_full,
        )

        seq_title    = user_title[ids_t].unsqueeze(0)
        seq_category = user_category[ids_t].unsqueeze(0)

        u_title    = self.title_transformer(seq_title,    ts_t.unsqueeze(0)).squeeze(0)
        u_category = self.category_transformer(seq_category, ts_t.unsqueeze(0)).squeeze(0)

        u_title, u_category = self.cross_attention(
            u_title.unsqueeze(0), u_category.unsqueeze(0)
        )
        return u_title.squeeze(0), u_category.squeeze(0), user_title, user_category

    def compute_score(self, u_title, u_category, i_title, i_category):
        return (u_title * i_title).sum(-1) + (u_category * i_category).sum(-1)

    def compute_loss(self, batch_sequences, rep_dict, graph_structure, device):
        """BPR loss — 원래 구조 그대로"""
        batch_losses = []
        for user_seq, target, negative in batch_sequences:
            u_title, u_category, mod_title, mod_category = self.forward_user(
                user_seq, rep_dict, graph_structure, device
            )
            pos_t = mod_title[target - 1];    pos_c = mod_category[target - 1]
            neg_t = mod_title[negative - 1];  neg_c = mod_category[negative - 1]

            pos_score = self.compute_score(
                u_title.unsqueeze(0), u_category.unsqueeze(0),
                pos_t.unsqueeze(0),   pos_c.unsqueeze(0)
            )
            neg_score = self.compute_score(
                u_title.unsqueeze(0), u_category.unsqueeze(0),
                neg_t.unsqueeze(0),   neg_c.unsqueeze(0)
            )
            batch_losses.append(-torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8))

        return torch.stack(batch_losses).mean()
