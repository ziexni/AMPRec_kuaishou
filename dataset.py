import torch
import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict


class MicroVideoDataset(Dataset):
    """
    시청 시퀀스 기반 Dataset
    - video_id는 이미 0 ~ N-1로 reindex된 상태로 들어옴 (train.py에서 처리)
    - num_items를 외부에서 받아서 통일
    - 실제 timestamp 사용 (cumcount 제거)
    """
    def __init__(self, interactions_df, num_items, max_seq_len=50, mode='train'):
        self.max_seq_len = max_seq_len
        self.mode        = mode
        self.num_items   = num_items  # ✅ 외부에서 통일된 값 받음
        self.samples     = []
        self.user_train  = {}
        self.user_valid  = {}
        self.user_test   = {}
        self.train_item_dict = defaultdict(set)

        interactions_df = interactions_df.copy()
        # ✅ 실제 timestamp 유지 (cumcount 덮어쓰기 제거)
        interactions_df = interactions_df.sort_values(['user_id', 'timestamp'])

        self.user_sequences = defaultdict(list)
        for _, row in interactions_df.iterrows():
            u = int(row['user_id'])
            i = int(row['video_id'])  # ✅ 이미 0-indexed (train.py에서 reindex됨)
            self.user_sequences[u].append({
                'user_id':       u,
                'item_id':       i,  # 0-indexed
                'timestamp':     float(row['timestamp']),
                'watch_ratio':   float(row['watch_ratio']),
                'watch_seconds': float(row['watch_seconds']),
                'node_signal':   float(row['node_signal']),
            })

        for u, seq in self.user_sequences.items():
            if len(seq) < 3:
                self.user_train[u] = seq
                self.user_valid[u] = []
                self.user_test[u]  = []
            else:
                self.user_train[u] = seq[:-2]
                self.user_valid[u] = [seq[-2]]
                self.user_test[u]  = [seq[-1]]

            for s in self.user_train[u]:
                self.train_item_dict[u].add(s['item_id'])

        for u, train_seq in self.user_train.items():
            if len(train_seq) < 1:
                continue

            if mode == 'train':
                if len(train_seq) < 2:
                    continue
                target    = train_seq[-1]['item_id']
                input_seq = train_seq[:-1]
            elif mode == 'valid':
                if not self.user_valid[u]:
                    continue
                target    = self.user_valid[u][0]['item_id']
                input_seq = train_seq
            else:
                if not self.user_test[u]:
                    continue
                target    = self.user_test[u][0]['item_id']
                valid_seq = self.user_valid[u] if self.user_valid[u] else []
                input_seq = train_seq + valid_seq

            self.samples.append((u, input_seq, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        u, seq, target = self.samples[idx]

        rated = self.train_item_dict[u]
        neg = np.random.randint(0, self.num_items)
        while neg in rated:
            neg = np.random.randint(0, self.num_items)

        seq     = seq[-self.max_seq_len:]
        seq_len = len(seq)
        pad_len = self.max_seq_len - seq_len

        item_ids     = [s['item_id']     for s in seq] + [0] * pad_len
        timestamps   = [s['timestamp']   for s in seq] + [0.0] * pad_len
        node_signals = [s['node_signal'] for s in seq] + [0.0] * pad_len

        return {
            'user_id':     torch.tensor(u,            dtype=torch.long),
            'item_ids':    torch.tensor(item_ids,     dtype=torch.long),
            'timestamps':  torch.tensor(timestamps,   dtype=torch.float),
            'node_signals':torch.tensor(node_signals, dtype=torch.float),
            'seq_len':     torch.tensor(seq_len,      dtype=torch.long),
            'target':      torch.tensor(target,       dtype=torch.long),
            'negative':    torch.tensor(neg,          dtype=torch.long),
        }
