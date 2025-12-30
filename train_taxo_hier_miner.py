import os
import re
import math
import json
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_impressions(imps: str) -> Tuple[List[str], List[int]]:
    # "N12345-1 N23456-0 ..." -> (ids, labels)
    items = imps.strip().split()
    nids, labels = [], []
    for it in items:
        nid, lab = it.rsplit("-", 1)
        nids.append(nid)
        labels.append(int(lab))
    return nids, labels


def safe_split_history(hist: str) -> List[str]:
    if hist is None or hist == "" or (isinstance(hist, float) and math.isnan(hist)):
        return []
    if hist.strip() == "":
        return []
    return hist.strip().split()


# -----------------------------
# Load MIND news
# -----------------------------
def load_news_tsv(news_path: str) -> pd.DataFrame:
    cols = ["NewsID", "Category", "SubCategory", "Title", "Abstract", "URL", "TitleEnt", "AbsEnt"]
    df = pd.read_csv(news_path, sep="\t", header=None, names=cols)
    df["Title"] = df["Title"].fillna("")
    df["Abstract"] = df["Abstract"].fillna("")
    return df


def build_taxonomy_maps(news_df: pd.DataFrame):
    # topic = Category, subtopic = SubCategory
    topics = sorted(news_df["Category"].fillna("").unique().tolist())
    subs = sorted(news_df["SubCategory"].fillna("").unique().tolist())
    topic2id = {t: i for i, t in enumerate(topics)}
    sub2id = {s: i for i, s in enumerate(subs)}
    return topic2id, sub2id


def build_news_index(news_df: pd.DataFrame, topic2id: Dict[str, int], sub2id: Dict[str, int]):
    news2row = {}
    meta = {}
    for _, r in news_df.iterrows():
        nid = r["NewsID"]
        news2row[nid] = True
        meta[nid] = {
            "topic": topic2id.get(r["Category"], 0),
            "sub": sub2id.get(r["SubCategory"], 0),
            "title": r["Title"],
            "abstract": r["Abstract"],
        }
    return meta


# -----------------------------
# Dataset: build training samples from behaviors
# -----------------------------
class MINDTrainDataset(Dataset):
    """
    Each item returns:
      history_nids: List[str] length <= his_len
      cand_nid: str
      label: int (1/0)
    with negative sampling within an impression.
    """
    def __init__(
        self,
        behaviors_path: str,
        his_len: int = 50,
        npratio: int = 4,
        drop_no_pos: bool = True,
    ):
        cols = ["ImpressionID", "UserID", "Time", "History", "Impressions"]
        df = pd.read_csv(behaviors_path, sep="\t", header=None, names=cols)
        self.rows = []
        self.his_len = his_len
        self.npratio = npratio

        for _, r in df.iterrows():
            hist = safe_split_history(r["History"])
            hist = hist[-his_len:] if len(hist) > his_len else hist
            nids, labs = parse_impressions(r["Impressions"])

            pos = [nid for nid, lab in zip(nids, labs) if lab == 1]
            neg = [nid for nid, lab in zip(nids, labs) if lab == 0]

            if drop_no_pos and len(pos) == 0:
                continue

            # store impression-level pool; sampling done in __getitem__
            self.rows.append((hist, pos, neg))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        hist, pos_pool, neg_pool = self.rows[idx]
        # sample 1 positive
        pos_nid = random.choice(pos_pool)
        # sample npratio negatives (with replacement if needed)
        if len(neg_pool) == 0:
            neg_samples = []
        else:
            neg_samples = random.choices(neg_pool, k=self.npratio) if len(neg_pool) < self.npratio else random.sample(neg_pool, self.npratio)

        # build (1 + npratio) candidates
        cands = [pos_nid] + neg_samples
        labels = [1] + [0] * len(neg_samples)

        return {
            "history": hist,
            "cands": cands,
            "labels": labels
        }


@dataclass
class Batch:
    # history: [B, L]
    hist_nids: List[List[str]]
    cand_nids: List[List[str]]  # [B, (1+npratio)]
    labels: torch.Tensor        # [B, (1+npratio)]


def collate_fn(batch_list: List[dict]) -> Batch:
    hist = [b["history"] for b in batch_list]
    cands = [b["cands"] for b in batch_list]
    # pad labels to same width (usually equal already)
    max_c = max(len(b["labels"]) for b in batch_list)
    lab = []
    for b in batch_list:
        l = b["labels"] + [0] * (max_c - len(b["labels"]))
        lab.append(l)
    labels = torch.tensor(lab, dtype=torch.float32)
    return Batch(hist_nids=hist, cand_nids=cands, labels=labels)


# -----------------------------
# News Encoder (PLM)
# -----------------------------
class NewsEncoder(nn.Module):
    def __init__(self, plm_name: str = "bert-base-uncased", out_dim: int = 256, use_abstract: bool = True):
        super().__init__()
        self.plm = AutoModel.from_pretrained(plm_name)
        self.hidden = self.plm.config.hidden_size
        self.use_abstract = use_abstract
        self.reduce = nn.Linear(self.hidden, out_dim)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.plm(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls = out.last_hidden_state[:, 0]  # [B, hidden]
        return self.reduce(cls)            # [B, out_dim]


# -----------------------------
# Taxonomy-guided Hierarchical Multi-Interest Model
# -----------------------------
class TaxoHierMiner(nn.Module):
    """
    Implements:
      - subtopic attention aggregation -> u_s
      - topic attention aggregation -> u_t
      - candidate-aware dynamic selection over topic interests
      - disagreement regularization on topic interests
    """
    def __init__(
        self,
        plm_name: str,
        news_dim: int,
        n_topics: int,
        n_subtopics: int,
        margin_delta: float = 0.2,
        lambda_dis: float = 0.1,
        freeze_plm: bool = False
    ):
        super().__init__()
        self.news_encoder = NewsEncoder(plm_name=plm_name, out_dim=news_dim)
        if freeze_plm:
            for p in self.news_encoder.plm.parameters():
                p.requires_grad = False

        self.n_topics = n_topics
        self.n_subtopics = n_subtopics
        self.news_dim = news_dim
        self.margin_delta = margin_delta
        self.lambda_dis = lambda_dis

        # learnable queries (attention) per subtopic/topic
        self.q_sub = nn.Embedding(n_subtopics, news_dim)
        self.q_topic = nn.Embedding(n_topics, news_dim)

    def disagreement_loss(self, topic_vecs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        topic_vecs: [B, K, D]
        mask: [B, K]  (1 for valid)
        """
        B, K, D = topic_vecs.shape
        # cosine sim matrix per batch: [B, K, K]
        v = F.normalize(topic_vecs, dim=-1)
        sim = torch.matmul(v, v.transpose(1, 2))
        # exclude diagonal
        eye = torch.eye(K, device=sim.device).unsqueeze(0)
        sim = sim * (1 - eye)

        # only keep pairs where both valid
        m = mask.unsqueeze(2) * mask.unsqueeze(1)  # [B,K,K]
        sim = sim * m

        # hinge: max(0, sim - delta)
        loss = F.relu(sim - self.margin_delta)
        # normalize by number of valid pairs
        denom = m.sum().clamp_min(1.0)
        return loss.sum() / denom

    def forward(
        self,
        # tensors for news encoding
        hist_input_ids, hist_attn_mask,
        cand_input_ids, cand_attn_mask,
        # taxonomy ids
        hist_topic_ids, hist_sub_ids,
        cand_topic_ids, cand_sub_ids,
        # masks
        hist_mask,  # [B, L]
    ):
        """
        hist_*: [B, L, T] for ids/mask after flattening for encoder
        cand_*: [B, C, T]
        """
        device = hist_input_ids.device
        B, L, T = hist_input_ids.shape
        C = cand_input_ids.shape[1]

        # --- Encode history news ---
        h_hist = self.news_encoder(
            hist_input_ids.view(B * L, T),
            hist_attn_mask.view(B * L, T)
        ).view(B, L, self.news_dim)  # [B,L,D]

        # --- Encode candidate news ---
        h_cand = self.news_encoder(
            cand_input_ids.view(B * C, T),
            cand_attn_mask.view(B * C, T)
        ).view(B, C, self.news_dim)  # [B,C,D]

        # -------- Subtopic-level aggregation --------
        # For each subtopic s, aggregate clicked news belonging to s.
        # We build per-user subtopic vectors for subtopics appeared in history.
        # Implementation: group-by via masking over unique subtopics in each batch item.
        # To keep it efficient, we restrict K_sub to distinct subtopics in each user history (<= L).
        sub_ids = hist_sub_ids  # [B,L]
        topic_ids = hist_topic_ids  # [B,L]
        valid = hist_mask.float()   # [B,L]

        # get unique subtopics per user (ragged -> pad)
        # We'll do a simple CPU loop for indices, then gather on GPU.
        sub_list = []
        for b in range(B):
            ids = sub_ids[b][hist_mask[b]].detach().cpu().tolist()
            uniq = list(dict.fromkeys(ids))  # keep order
            sub_list.append(uniq)

        Ksub = max(len(x) for x in sub_list) if B > 0 else 1
        sub_pad = torch.zeros(B, Ksub, dtype=torch.long, device=device)
        sub_mask = torch.zeros(B, Ksub, dtype=torch.float32, device=device)
        for b, uniq in enumerate(sub_list):
            if len(uniq) == 0:
                continue
            sub_pad[b, :len(uniq)] = torch.tensor(uniq, device=device)
            sub_mask[b, :len(uniq)] = 1.0

        # attention within each subtopic:
        # score_{b,k,l} = q_sub[sub_pad[b,k]] dot h_hist[b,l]
        qsub = self.q_sub(sub_pad)  # [B,Ksub,D]
        scores_sub = torch.einsum("bkd,bld->bkl", qsub, h_hist)  # [B,Ksub,L]

        # mask: only news with that subtopic and valid history
        same_sub = (sub_pad.unsqueeze(-1) == sub_ids.unsqueeze(1)).float()  # [B,Ksub,L]
        m_sub = same_sub * valid.unsqueeze(1)  # [B,Ksub,L]
        scores_sub = scores_sub.masked_fill(m_sub == 0, -1e9)
        attn_sub = torch.softmax(scores_sub, dim=-1)  # [B,Ksub,L]
        u_sub = torch.einsum("bkl,bld->bkd", attn_sub, h_hist)  # [B,Ksub,D]

        # -------- Topic-level aggregation --------
        # For each topic t, aggregate its subtopic vectors u_sub via topic attention.
        # Need topic id for each subtopic: take the most frequent topic among clicks in that subtopic.
        # We compute topic_of_sub[b,k] as mode over topic_ids where sub_ids match.
        topic_of_sub = torch.zeros(B, Ksub, dtype=torch.long, device=device)
        for b in range(B):
            for k in range(Ksub):
                if sub_mask[b, k].item() == 0:
                    continue
                s = sub_pad[b, k].item()
                idxs = (sub_ids[b] == s) & hist_mask[b]
                if idxs.sum().item() == 0:
                    continue
                tvals = topic_ids[b][idxs].detach().cpu().tolist()
                # mode
                t = max(set(tvals), key=tvals.count)
                topic_of_sub[b, k] = t

        # unique topics per user
        topic_list = []
        for b in range(B):
            ids = topic_of_sub[b][sub_mask[b].bool()].detach().cpu().tolist()
            uniq = list(dict.fromkeys(ids))
            topic_list.append(uniq)

        Ktop = max(len(x) for x in topic_list) if B > 0 else 1
        top_pad = torch.zeros(B, Ktop, dtype=torch.long, device=device)
        top_mask = torch.zeros(B, Ktop, dtype=torch.float32, device=device)
        for b, uniq in enumerate(topic_list):
            if len(uniq) == 0:
                continue
            top_pad[b, :len(uniq)] = torch.tensor(uniq, device=device)
            top_mask[b, :len(uniq)] = 1.0

        qtop = self.q_topic(top_pad)  # [B,Ktop,D]
        # score_{b,kt,ks} = q_topic dot u_sub
        scores_top = torch.einsum("btd,bsd->bts", qtop, u_sub)  # [B,Ktop,Ksub]

        # mask: only subtopics belonging to that topic and valid subtopic
        same_top = (top_pad.unsqueeze(-1) == topic_of_sub.unsqueeze(1)).float()  # [B,Ktop,Ksub]
        m_top = same_top * sub_mask.unsqueeze(1)  # [B,Ktop,Ksub]
        scores_top = scores_top.masked_fill(m_top == 0, -1e9)
        attn_top = torch.softmax(scores_top, dim=-1)  # [B,Ktop,Ksub]
        u_top = torch.einsum("bts,bsd->btd", attn_top, u_sub)  # [B,Ktop,D]

        # -------- Candidate-aware dynamic selection (MINER-style) --------
        # For each candidate: soft select topic interests
        # s_{b,c,t} = cosine(u_top[b,t], h_cand[b,c])
        u_top_norm = F.normalize(u_top, dim=-1)
        h_cand_norm = F.normalize(h_cand, dim=-1)
        s = torch.einsum("btd,bcd->bct", u_top_norm, h_cand_norm)  # [B,C,Ktop]
        s = s.masked_fill(top_mask.unsqueeze(1) == 0, -1e9)
        gamma = torch.softmax(s, dim=-1)  # [B,C,Ktop]
        u_uc = torch.einsum("bct,btd->bcd", gamma, u_top)  # [B,C,D]

        # click score
        y_hat = torch.einsum("bcd,bcd->bc", u_uc, h_cand)  # [B,C]

        # disagreement reg on u_top
        dis = self.disagreement_loss(u_top, top_mask) if self.lambda_dis > 0 else torch.tensor(0.0, device=device)

        return y_hat, dis


# -----------------------------
# Tokenization & batching for news ids
# -----------------------------
class NewsTextStore:
    def __init__(self, news_meta: Dict[str, dict], tokenizer, max_len: int = 96, use_abstract: bool = True):
        self.meta = news_meta
        self.tok = tokenizer
        self.max_len = max_len
        self.use_abstract = use_abstract
        self.cache = {}  # nid -> (input_ids, attn_mask) on CPU

    def encode_one(self, nid: str):
        if nid in self.cache:
            return self.cache[nid]
        m = self.meta.get(nid, None)
        if m is None:
            text = ""
            topic = 0
            sub = 0
        else:
            text = m["title"]
            if self.use_abstract and m.get("abstract", ""):
                text = (m["title"] + " " + m["abstract"]).strip()
            topic = m["topic"]
            sub = m["sub"]

        enc = self.tok(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"][0]       # [T]
        attn_mask = enc["attention_mask"][0]  # [T]
        # cache on CPU
        self.cache[nid] = (input_ids, attn_mask, topic, sub)
        return self.cache[nid]

    def batch_encode(self, nids_2d: List[List[str]]):
        B = len(nids_2d)
        L = max(len(x) for x in nids_2d) if B > 0 else 1
        T = self.max_len
        input_ids = torch.zeros(B, L, T, dtype=torch.long)
        attn_mask = torch.zeros(B, L, T, dtype=torch.long)
        topic_ids = torch.zeros(B, L, dtype=torch.long)
        sub_ids = torch.zeros(B, L, dtype=torch.long)
        mask = torch.zeros(B, L, dtype=torch.bool)

        for b, seq in enumerate(nids_2d):
            for i, nid in enumerate(seq):
                ids, am, t, s = self.encode_one(nid)
                input_ids[b, i] = ids
                attn_mask[b, i] = am
                topic_ids[b, i] = t
                sub_ids[b, i] = s
                mask[b, i] = True
        return input_ids, attn_mask, topic_ids, sub_ids, mask


def move_to_device(*tensors, device):
    return [x.to(device, non_blocking=True) for x in tensors]


# -----------------------------
# Train loop
# -----------------------------
def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load news + taxonomy
    train_news = load_news_tsv(args.train_news)
    topic2id, sub2id = build_taxonomy_maps(train_news)
    train_meta = build_news_index(train_news, topic2id, sub2id)

    tokenizer = AutoTokenizer.from_pretrained(args.plm)
    store = NewsTextStore(train_meta, tokenizer, max_len=args.max_len, use_abstract=args.use_abstract)

    ds = MINDTrainDataset(args.train_behaviors, his_len=args.his_len, npratio=args.npratio)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                    pin_memory=True, collate_fn=collate_fn, drop_last=True)

    model = TaxoHierMiner(
        plm_name=args.plm,
        news_dim=args.news_dim,
        n_topics=len(topic2id),
        n_subtopics=len(sub2id),
        margin_delta=args.delta,
        lambda_dis=args.lambda_dis,
        freeze_plm=args.freeze_plm,
    ).to(device)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.wd)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    model.train()
    step = 0
    for epoch in range(args.epochs):
        pbar = tqdm(dl, desc=f"Epoch {epoch}")
        for batch in pbar:
            # Build history/candidates id lists
            hist_nids = [h[-args.his_len:] for h in batch.hist_nids]
            # candidates is already list per user
            cand_nids = batch.cand_nids

            # Encode history/candidates into tensors
            hist_ids, hist_mask, hist_topic, hist_sub, hist_valid = store.batch_encode(hist_nids)
            cand_ids, cand_mask, cand_topic, cand_sub, _ = store.batch_encode(cand_nids)

            hist_ids, hist_mask, hist_topic, hist_sub, hist_valid = move_to_device(
                hist_ids, hist_mask, hist_topic, hist_sub, hist_valid, device=device
            )
            cand_ids, cand_mask, cand_topic, cand_sub = move_to_device(
                cand_ids, cand_mask, cand_topic, cand_sub, device=device
            )
            labels = batch.labels.to(device)

            with torch.cuda.amp.autocast(enabled=args.amp):
                y_hat, dis = model(
                    hist_ids, hist_mask,
                    cand_ids, cand_mask,
                    hist_topic, hist_sub,
                    cand_topic, cand_sub,
                    hist_valid
                )
                # BCE with logits
                rec = F.binary_cross_entropy_with_logits(y_hat, labels)
                loss = rec + args.lambda_dis * dis

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            step += 1
            pbar.set_postfix(loss=float(loss.detach().cpu()), rec=float(rec.detach().cpu()), dis=float(dis.detach().cpu()))

    # save
    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.out_dir, "taxo_hier_miner.pt"))
    with open(os.path.join(args.out_dir, "taxonomy_maps.json"), "w", encoding="utf-8") as f:
        json.dump({"topic2id": topic2id, "sub2id": sub2id}, f, ensure_ascii=False, indent=2)

    print("Saved to", args.out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_news", type=str, required=True)
    ap.add_argument("--train_behaviors", type=str, required=True)
    ap.add_argument("--plm", type=str, default="bert-base-uncased")
    ap.add_argument("--use_abstract", action="store_true")
    ap.add_argument("--max_len", type=int, default=96)
    ap.add_argument("--news_dim", type=int, default=256)

    ap.add_argument("--his_len", type=int, default=50)
    ap.add_argument("--npratio", type=int, default=4)

    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=1)

    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--wd", type=float, default=0.01)

    ap.add_argument("--delta", type=float, default=0.2)
    ap.add_argument("--lambda_dis", type=float, default=0.1)

    ap.add_argument("--freeze_plm", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--out_dir", type=str, default="outputs")
    args = ap.parse_args()

    train(args)


if __name__ == "__main__":
    main()
