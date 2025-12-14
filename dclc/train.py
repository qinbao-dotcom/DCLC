import os, sys, pickle, logging
sys.path.append('../')
import torch
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
from card import data_prepare, options
from card.dataset import EmbeddingDatasetCL
from model import TransformerChunkEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    args = options.args_parser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device=%s", device)

    # ---------- 语料准备 ----------
    data_prepare.get_corpus_for_train(
        args.corpus_dir, avg_size=args.average_size,
        sliding_window=args.feature_window, num_perm=args.num_perm)
    text, w2i, n2i, freqs, i2w = data_prepare.establish_index(
        args.corpus_dir + "_learning_corpus",
        max_vocab=args.max_vocab, max_ngram=args.max_ngram,
        ngram_window=args.ngram_window)
    ds = EmbeddingDatasetCL(text, w2i, n2i, i2w,
                            K=args.neighbor, N_neg=args.negative_sample,
                            ngram_len=args.ngram_window)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True,
                    num_workers=args.num_workers, pin_memory=True)

    # ---------- 模型 ----------
    model = TransformerChunkEncoder(
        vocab_size=args.max_vocab, ngram_size=args.max_ngram,
        d_model=args.hidden, nhead=8,
        n_layers=args.n_layers, dim_ff=args.ff_dim).to(device)
    model.set_index_file(w2i, n2i)
    opt = optim.AdamW(model.parameters(), lr=args.lr)

    # ---------- 训练 ----------
    best_state, min_loss, patience = {}, float("inf"), 0
    for epoch in range(args.epochs):
        model.train(); run_loss = 0
        for seq_q, seq_k_p, seq_k_n, ng_q, ng_k in tqdm(dl):
            seq_q, seq_k_p   = seq_q.to(device), seq_k_p.to(device)
            seq_k_n, ng_q    = seq_k_n.to(device), ng_q.to(device)
            ng_k             = ng_k.to(device)
            opt.zero_grad()
            loss = model(seq_q, seq_k_p, seq_k_n, ng_q, ng_k).mean()
            loss.backward(); opt.step()
            run_loss += loss.item()

        avg = run_loss / len(dl)
        logger.info("Epoch %d  loss=%.5f", epoch, avg)
        if avg < min_loss - 0.0005:
            min_loss, patience = avg, 0
            best_state = {"model": model.state_dict(),
                          "word2idx": w2i, "ngram2idx": n2i}
        else:
            patience += 1
        if patience >= 5:
            logger.info("Early stop."); break

    # ---------- 保存 ----------
    torch.save(best_state["model"], args.model)
    base = os.path.splitext(args.model)[0]
    pickle.dump(best_state["word2idx"], open(base + "_word2idx.pkl", "wb"))
    pickle.dump(best_state["ngram2idx"], open(base + "_ngram2idx.pkl", "wb"))
