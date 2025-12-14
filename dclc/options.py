#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.6+

import argparse


def args_parser():
    parser = argparse.ArgumentParser(
        description="Transformer-InfoNCE Chunk-Dedup training args")

    # ------------------------- 基础训练设置 -------------------------
    parser.add_argument('--epochs', type=int, default=20,
                        help="total training epochs")
    parser.add_argument('--bs', type=int, default=256,
                        help="mini-batch size")
    parser.add_argument('--lr', type=float, default=3e-4,
                        help="AdamW learning rate")
    parser.add_argument('--seed', type=int, default=1,
                        help="random seed")

    # ------------------------- 语料 / 数据处理 -------------------------
    parser.add_argument('--corpus_dir', type=str,
                        default='/mnt/tmp/sum/nltk_data/tokenizers/punkt',
                        help="raw corpus directory (chunked files)")
    parser.add_argument('--shingles', action='store_true',
                        help="use prime-ranking shingles (default on)")
    parser.add_argument('--average_size', type=int, default=32768,
                        help="FastCDC avg chunk size (Bytes)")
    parser.add_argument('--feature_window', type=int, default=48,
                        help="sliding-window size when hashing")
    parser.add_argument('--num_perm', type=int, default=128,
                        help="MinHash num_perm")

    # ------------------------- 词表设置 -------------------------
    parser.add_argument('--max_vocab', type=int, default=40000,
                        help="max token vocab size")
    parser.add_argument('--max_ngram', type=int, default=200000,
                        help="max n-gram vocab size")
    parser.add_argument('--ngram_window', type=int, default=5,
                        help="n of n-gram tokens")

    # ------------------------- InfoNCE 采样 -------------------------
    parser.add_argument('--neighbor', type=int, default=4,
                        help="K (±K) context size")
    parser.add_argument('--negative_sample', type=int, default=15,
                        help="number of negative blocks per query")
    parser.add_argument('--tau', type=float, default=0.07,
                        help="InfoNCE temperature")

    # ------------------------- Transformer 超参 -------------------------
    parser.add_argument('--hidden', type=int, default=128,
                        help="embedding / model dimension (d_model)")
    parser.add_argument('--n_layers', type=int, default=4,
                        help="Transformer encoder layers")
    parser.add_argument('--ff_dim', type=int, default=512,
                        help="feed-forward inner dim")

    # ------------------------- I/O -------------------------
    parser.add_argument('--model', type=str,
                        default='/mnt/tmp/sum/models/card_transformer.pt',
                        help="path to save model weights")

    # ------------------------- 计算资源 -------------------------
    parser.add_argument('--num_workers', type=int, default=4,
                        help="DataLoader workers")
    parser.add_argument('--gpu', type=int, default=0,
                        help="GPU id, -1 for CPU")
    args = parser.parse_args()
    return args
