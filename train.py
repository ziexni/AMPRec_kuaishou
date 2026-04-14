import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import math
import torch
import numpy as np
import pandas as pd
from collections import defaultdict

from preprocess import preprocess_kuaishou
from dataset import MicroVideoDataset
from model import MicroVideoRec


def evaluate(model, dataset, rep_dict, graph_structure, device,
             num_neg=100, topk=10, mode='test'):
    model.eval()

    eval_dict   = dataset.user_valid if mode == 'valid' else dataset.user_test
    valid_users = [u for u, items in eval_dict.items()
                   if len(items) > 0 and len(dataset.user_train[u]) >= 1]

    NDCG = HR = MRR = 0.0
    count = 0

    for u in valid_users:
        if mode == 'valid':
            user_seq = dataset.user_train[u]
        else:
            valid_seq = dataset.user_valid[u] if dataset.user_valid[u] else []
            user_seq  = dataset.user_train[u] + valid_seq

        if len(user_seq) == 0:
            continue

        target = eval_dict[u][0]['item_id']

        rated = dataset.train_item_dict[u] | {0}
        candidates = [target]
        for _ in range(num_neg):
            neg = np.random.randint(0, dataset.num_items)
            while neg in rated:
                neg = np.random.randint(0, dataset.num_items)
            candidates.append(neg)

        with torch.no_grad():
            u_title, u_category, mod_title, mod_category = model.forward_user(
                user_seq, rep_dict, graph_structure, device
            )

        cand_ids = torch.tensor(candidates, dtype=torch.long, device=device)
        cand_t   = mod_title[cand_ids]
        cand_c   = mod_category[cand_ids]

        with torch.no_grad():
            u_t_exp = u_title.unsqueeze(0).expand(len(candidates), -1)
            u_c_exp = u_category.unsqueeze(0).expand(len(candidates), -1)
            scores  = model.compute_score(u_t_exp, u_c_exp, cand_t, cand_c)

        rank  = (scores[0] <= scores).sum().item() - 1
        count += 1
        MRR   += 1 / (rank + 1)
        if rank < topk:
            NDCG += 1 / np.log2(rank + 2)
            HR   += 1

    N = max(count, 1)
    return NDCG / N, HR / N, MRR / N


def train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    print("Loading data...")
    interactions_df = pd.read_pickle(config['interaction_path'])

    # ✅ item_df 기준으로 통일
    item_df = interactions_df[['video_id', 'title', 'category']]\
        .drop_duplicates('video_id')\
        .reset_index(drop=True)

    # ✅ 0 ~ N-1 reindex
    id_map     = {vid: idx for idx, vid in enumerate(item_df['video_id'])}
    num_items  = len(item_df)
    title_feat = np.stack(item_df['title'].values)
    print(f"Items: {num_items}\n")

    # id_map을 interactions_df에 적용
    interactions_df = interactions_df.copy()
    interactions_df['video_id'] = interactions_df['video_id'].map(id_map)

    interactions_df, rep_dict, graph_structure = preprocess_kuaishou(
        interactions_df, item_df, title_feat,
        epsilon=config.get('epsilon', 0.4),
        topk=config.get('topk_sim', 30),
        lambda_decay=config.get('lambda_decay', 0.1),
        signal_params=config.get('signal_params'),
    )

    # ✅ num_items를 dataset에 전달
    dataset = MicroVideoDataset(
        interactions_df, num_items=num_items,
        max_seq_len=config['max_seq_len'], mode='train'
    )

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

    num_cats      = max(all_cats) + 1
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

    assert title_feat.shape[0] == num_items, \
        f"title_feat mismatch: {title_feat.shape[0]} vs {num_items}"
    assert category_feat.shape[0] == num_items, \
        f"category_feat mismatch: {category_feat.shape[0]} vs {num_items}"

    model = MicroVideoRec(
        num_items=num_items,
        title_feat=title_feat,
        category_feat=category_feat,
        dim=config['dim'],
        num_heads=config['num_heads'],
        num_transformer_layers=config['num_transformer_layers'],
        num_gcn_layers=config['num_gcn_layers'],
        dropout=config['dropout'],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'],
                                 weight_decay=config['weight_decay'])

    def lr_lambda(epoch):
        warmup = config['warmup_epochs']
        if epoch < warmup:
            return float(epoch + 1) / float(warmup)
        progress = (epoch - warmup) / max(config['num_epochs'] - warmup, 1)
        return max(config['min_lr'] / config['lr'],
                   0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_ndcg  = 0.0
    best_test_ndcg = best_test_hr = best_test_mrr = 0.0
    num_decreases  = 0
    sample_indices = list(range(len(dataset.samples)))

    for epoch in range(config['num_epochs']):
        model.train()
        total_loss  = 0.0

        np.random.shuffle(sample_indices)
        num_batches = len(sample_indices) // config['batch_size']

        for batch_idx in range(num_batches):
            start = batch_idx * config['batch_size']
            end   = start + config['batch_size']
            batch_sample_indices = sample_indices[start:end]

            batch_sequences = []
            for idx in batch_sample_indices:
                u, input_seq, target = dataset.samples[idx]

                rated = dataset.train_item_dict[u] | {0}
                neg = np.random.randint(0, dataset.num_items)
                while neg in rated:
                    neg = np.random.randint(0, dataset.num_items)

                batch_sequences.append((input_seq, target, neg))

            optimizer.zero_grad()
            loss = model.compute_loss(
                batch_sequences, rep_dict, graph_structure, device
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                print(f"  Batch [{batch_idx+1}/{num_batches}] Loss: {loss.item():.4f}")

        scheduler.step()
        avg_loss = total_loss / max(num_batches, 1)
        cur_lr   = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{config['num_epochs']} | Loss: {avg_loss:.4f} | LR: {cur_lr:.2e}")

        if (epoch + 1) % config['eval_every'] == 0:
            print(f"\nEvaluating...")
            val_ndcg, val_hr, val_mrr = evaluate(
                model, dataset, rep_dict, graph_structure, device,
                num_neg=config['num_neg'], topk=config['topk'], mode='valid',
            )
            print(f"  [Valid] NDCG@{config['topk']}: {val_ndcg:.4f} | "
                  f"HR@{config['topk']}: {val_hr:.4f} | MRR: {val_mrr:.4f}")

            test_ndcg, test_hr, test_mrr = evaluate(
                model, dataset, rep_dict, graph_structure, device,
                num_neg=config['num_neg'], topk=config['topk'], mode='test',
            )
            print(f"  [Test]  NDCG@{config['topk']}: {test_ndcg:.4f} | "
                  f"HR@{config['topk']}: {test_hr:.4f} | MRR: {test_mrr:.4f}")

            if val_ndcg > best_val_ndcg:
                best_val_ndcg  = val_ndcg
                best_test_ndcg = test_ndcg
                best_test_hr   = test_hr
                best_test_mrr  = test_mrr
                num_decreases  = 0
                torch.save(model.state_dict(), config['save_path'])
                print(f"  ✓ Best saved")
            else:
                num_decreases += 1
                if num_decreases >= config['patience']:
                    print("Early stopping.")
                    break

    print(f"\n{'='*50}")
    print(f"Best Valid NDCG@{config['topk']}: {best_val_ndcg:.4f}")
    print(f"Best Test  NDCG@{config['topk']}: {best_test_ndcg:.4f} | "
          f"HR@{config['topk']}: {best_test_hr:.4f} | MRR: {best_test_mrr:.4f}")
    print(f"{'='*50}")


if __name__ == '__main__':
    config = {
        'interaction_path': '/kaggle/input/datasets/jieunl2/kuaishou/kuaishou_preprocess.pkl',
        'save_path':        '/kaggle/working/output/best_model.pt',

        'epsilon':      0.4,
        'topk_sim':     30,
        'lambda_decay': 0.1,
        'signal_params': {'a': 3.0, 'b': 2.0, 'c': 4.5},

        'dim':                    128,
        'num_heads':              4,
        'num_transformer_layers': 3,
        'num_gcn_layers':         2,
        'dropout':                0.1,
        'max_seq_len':            50,

        'lr':           2e-3,
        'weight_decay': 5e-6,
        'batch_size':   32,
        'num_epochs':   100,
        'eval_every':   5,
        'topk':         10,
        'patience':     10,
        'num_neg':      100,
        'warmup_epochs':5,
        'min_lr':       1e-6,
    }

    import os
    os.makedirs('/kaggle/working/output', exist_ok=True)
    train(config)
