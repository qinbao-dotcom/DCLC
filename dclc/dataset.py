import torch
from torch.utils import data as tud
from card.utils import get_n_grams


class EmbeddingDatasetCL(tud.Dataset):
    """
    输出 5 个张量:
      seq_q      : (2K+1)       —— 查询序列 (含中心)
      seq_k_pos  : (2K+1)       —— 正样本序列 (轻扰动中心)
      seq_k_neg  : (N,2K+1)     —— 负样本序列集合
      ngram_q    : (M=12)       —— 查询 token 的 n-gram 索引
      ngram_k    : (N+1,M)      —— 正+负的 n-gram 索引
    """

    def __init__(self, text, word2idx, ngram2idx, idx2word,
                 K=4, N_neg=5, ngram_len=5):
        super().__init__()
        self.tokens = torch.LongTensor(
            [word2idx.get(t, word2idx["<UNK>"]) for t in text]
        )
        self.word2idx, self.ngram2idx = word2idx, ngram2idx
        self.idx2word = idx2word
        self.K, self.N_neg, self.ngram_len = K, N_neg, ngram_len
        self.win = 2 * K + 1

    # -------- n-gram helper --------
    def _token_ngrams(self, tok_idx):
        word = self.idx2word[tok_idx.item()]
        grams = get_n_grams(word, n=self.ngram_len, horizon=False,
                            single_word=True)
        grams = [g for g in grams if g != "UNK"][:12]
        grams += ["UNK"] * (12 - len(grams))
        return torch.LongTensor(
            [self.ngram2idx.get(g, self.ngram2idx["<UNK>"]) for g in grams]
        )

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        # ---- query 序列 ----
        left = [self.tokens[(idx - i - 1) % len(self.tokens)]
                for i in range(self.K)][::-1]
        right = [self.tokens[(idx + i + 1) % len(self.tokens)]
                 for i in range(self.K)]
        seq_q = torch.stack(left + [self.tokens[idx]] + right)          # (2K+1)

        # ---- 轻扰动中心 token 作为正样本 ----
        word = self.idx2word[self.tokens[idx].item()]
        pert = word[:-1] if len(word) > 1 else word                     # 删 1 字符
        pert_idx = self.word2idx.get(pert, self.word2idx["<UNK>"])
        seq_k_pos = seq_q.clone()
        seq_k_pos[self.K] = torch.tensor(pert_idx)

        # ---- 负样本序列 ----
        neg_seq, neg_ng = [], []
        for _ in range(self.N_neg):
            j = torch.randint(0, len(self.tokens), ()).item()
            neg_left = [self.tokens[(j - i - 1) % len(self.tokens)]
                        for i in range(self.K)][::-1]
            neg_right = [self.tokens[(j + i + 1) % len(self.tokens)]
                         for i in range(self.K)]
            neg_seq.append(torch.stack(neg_left + [self.tokens[j]] + neg_right))
            neg_ng.append(self._token_ngrams(self.tokens[j]))
        seq_k_neg = torch.stack(neg_seq)                                # (N,2K+1)

        # ---- n-gram 张量 ----
        ngram_q = self._token_ngrams(self.tokens[idx])                  # (12)
        ngram_pos = self._token_ngrams(torch.tensor(pert_idx))
        ngram_k = torch.stack([torch.cat(
            [ngram_pos.unsqueeze(0), torch.stack(neg_ng)])])            # (1+N,12)

        return seq_q, seq_k_pos, seq_k_neg, ngram_q, ngram_k.squeeze(0)
