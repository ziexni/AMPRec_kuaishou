import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity


def compute_node_signal(watch_ratio, watch_seconds, a=3.0, b=2.0, c=4.5):
    watch_ratio_normalized = watch_ratio / 100.0
    watch_ratio_clipped = np.clip(watch_ratio_normalized, 0.0, 2.0)
    raw_signal = a * watch_ratio_clipped + b * np.log1p(watch_seconds) - c
    return raw_signal


def build_repetition_dict_temporal(interactions_df, lambda_decay=0.1):
    rep_dict = defaultdict(float)

    interactions_df = interactions_df.sort_values(['user_id', 'timestamp'])

    for u, group in interactions_df.groupby('user_id'):
        group = group.sort_values('timestamp')
        max_t = group['timestamp'].max()
        min_t = group['timestamp'].min()
        time_range = max_t - min_t + 1e-6  # [0,1] 정규화용

        for _, row in group.iterrows():
            i = int(row['video_id'])
            t = row['timestamp']

            # delta_t를 [0,1]로 정규화 → lambda_decay 원래 값 유지 가능
            delta_t = (max_t - t) / time_range
            weight  = np.exp(-lambda_decay * delta_t)
            rep_dict[(u, i)] += weight

    print(f"  Unique (user, item) pairs: {len(rep_dict)}")
    print(f"  Max repetition (temporal): {max(rep_dict.values()):.2f}")
    print(f"  Mean repetition: {np.mean(list(rep_dict.values())):.2f}")

    return rep_dict


def build_similarity_graph(feat, epsilon=0.4, topk=30):
    print(f"  Computing similarity (epsilon={epsilon}, top-k={topk})...")

    sim = cosine_similarity(feat)
    sim[sim < epsilon] = 0

    N = sim.shape[0]
    sparse = np.zeros_like(sim)
    for i in range(N):
        idx = np.argsort(sim[i])[::-1][1:topk+1]
        sparse[i, idx] = sim[i, idx]

    sim = np.maximum(sparse, sparse.T)

    edges, weights = [], []
    for i in range(N):
        for j in range(i+1, N):
            if sim[i, j] > 0:
                edges.append([i, j])
                edges.append([j, i])
                weights.append(sim[i, j])
                weights.append(sim[i, j])

    if len(edges) == 0:
        print(f"  WARNING: No edges!")
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros(0, dtype=torch.float)

    edge_index = torch.tensor(edges, dtype=torch.long).T
    sim_weight = torch.tensor(weights, dtype=torch.float)
    print(f"  Edges: {len(edges)}")

    return edge_index, sim_weight


def preprocess_kuaishou(interactions_df, item_df, title_feat,
                        epsilon=0.4, topk=30, lambda_decay=0.1, signal_params=None):
    print("\n=== Behavior-aware Preprocessing ===\n")

    if signal_params is None:
        signal_params = {'a': 3.0, 'b': 2.0, 'c': 4.5}

    # 1. Node signal
    print("[1/4] Computing node signals...")
    interactions_df = interactions_df.copy()
    interactions_df['node_signal'] = compute_node_signal(
        interactions_df['watch_ratio'].values,
        interactions_df['watch_seconds'].values,
        **signal_params
    )
    signal_stats = interactions_df['node_signal'].describe()
    print(f"  Min: {signal_stats['min']:.3f}")
    print(f"  Max: {signal_stats['max']:.3f}")
    print(f"  Mean: {signal_stats['mean']:.3f}")
    print(f"  Negative ratio: {(interactions_df['node_signal'] < 0).mean():.2%}")

    # 2. Repetition (실제 timestamp 기반 정규화)
    print("\n[2/4] Computing repetition (temporal decay, normalized timestamp)...")
    rep_dict = build_repetition_dict_temporal(interactions_df, lambda_decay)

    # 3. Category feature
    print("\n[3/4] Preparing category features...")
    import ast

    all_cats = set()
    for cats in item_df['category']:
        if isinstance(cats, str):
            try:   cats = ast.literal_eval(cats)
            except: cats = [int(cats)]
        if isinstance(cats, (list, np.ndarray)):
            all_cats.update([int(c) for c in cats])
        else:
            all_cats.add(int(cats))

    num_items = len(item_df)
    num_cats  = max(all_cats) + 1
    category_feat = np.zeros((num_items, num_cats), dtype=np.float32)
    for idx, row in item_df.iterrows():
        cats = row['category']
        if isinstance(cats, str):
            try:   cats = ast.literal_eval(cats)
            except: cats = [int(cats)]
        if isinstance(cats, (list, np.ndarray)):
            for c in cats: category_feat[idx, int(c)] = 1.0
        else:
            category_feat[idx, int(cats)] = 1.0

    # 4. Similarity graphs
    print("\n[4/4] Building similarity graphs...")
    print("  [Title]")
    title_edge_index, title_sim_weight = build_similarity_graph(title_feat, epsilon, topk)
    print("  [Category]")
    category_edge_index, category_sim_weight = build_similarity_graph(category_feat, epsilon, topk)

    graph_structure = {
        'title':    {'edge_index': title_edge_index,    'sim_weight': title_sim_weight},
        'category': {'edge_index': category_edge_index, 'sim_weight': category_sim_weight},
    }

    print("\n=== Preprocessing Complete ===\n")
    return interactions_df, rep_dict, graph_structure
