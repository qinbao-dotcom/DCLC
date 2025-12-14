import math, torch
from torch import nn
from torch.nn import functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):                      # x:(B,L,D)
        return self.dropout(x + self.pe[:, :x.size(1)])


class TransformerChunkEncoder(nn.Module):
    def __init__(self, vocab_size, ngram_size, d_model=128,
                 nhead=8, n_layers=2, dim_ff=512, dropout=0.1, tau=0.07):
        super().__init__()
        self.word_emb = nn.Embedding(vocab_size, d_model)
        self.ngram_emb = nn.Embedding(ngram_size, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
        self.pos = PositionalEncoding(d_model, dropout)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.tau = tau
        self.word2idx, self.ngram2idx = {}, {}

    # ---------- InfoNCE ----------
    def forward(self, seq_q, seq_k_pos, seq_k_neg, ng_q, ng_k):
        B, N, L = seq_k_neg.shape

        def encode(seq, ng):
            ctx = self.pos(self.word_emb(seq))             # (B,L,D)
            h = self.encoder(ctx)[:, L // 2, :]            # 中心输出
            ng = self.ngram_emb(ng).mean(1)                # (B,D)
            return F.normalize(self.proj(0.5 * (h + ng)), dim=1)

        q   = encode(seq_q, ng_q)                          # (B,D)
        k_p = encode(seq_k_pos, ng_k[:, 0, :])             # (B,D)

        seq_k_neg = seq_k_neg.view(B * N, L)
        ng_neg    = ng_k[:, 1:, :].reshape(B * N, -1)
        k_n = encode(seq_k_neg, ng_neg).view(B, N, -1)     # (B,N,D)

        pos_sim = (q * k_p).sum(1, keepdim=True)           # (B,1)
        neg_sim = torch.bmm(k_n, q.unsqueeze(2)).squeeze(2)# (B,N)
        logits  = torch.cat([pos_sim, neg_sim], 1) / self.tau
        labels  = torch.zeros(B, dtype=torch.long, device=q.device)
        return F.cross_entropy(logits, labels)

    # ---------- 推理 ----------
    @torch.no_grad()
    def chunk2vec(self, seq_q, ng_q):
        ctx = self.pos(self.word_emb(seq_q))
        h   = self.encoder(ctx)[:, seq_q.size(1)//2, :]
        ng  = self.ngram_emb(ng_q).mean(1)
        return self.proj(0.5 * (h + ng)).squeeze(0)        # (D,)

    def set_index_file(self, word2idx, ngram2idx):
        self.word2idx, self.ngram2idx = word2idx, ngram2idx
