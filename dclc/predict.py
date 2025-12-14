import sys
sys.path.append('../')
import os
import shutil
import time
from hashlib import md5
from fastcdc import fastcdc
import torch
from collections import OrderedDict
#from model import Fasttext
from annoy import AnnoyIndex
import xdelta3
import pickle
#import subprocess, tempfile
import logging
from model import TransformerChunkEncoder
from data_prepare  import get_minhash_from_chunk
from utils import get_n_grams
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# 创建一个日志记录器
logger = logging.getLogger(__name__)
train_files = ['thinkmobile_20200505.sql', 'thinkmobile_20200730.sql', 'linux-4.14.210', 'linux-4.4.247', 'CentOS 8-s001.vmdk']

def get_model(model_path, vocab_size, ngram_size,
              hidden, n_layers, ff_dim, tau):
    model = TransformerChunkEncoder(
        vocab_size=vocab_size,
        ngram_size=ngram_size,
        d_model=hidden,
        n_layers=n_layers,
        dim_ff=ff_dim,
        tau=tau
    )

    state = torch.load(model_path,
                       map_location='cuda' if torch.cuda.is_available() else 'cpu')
    filtered_state = {k: v for k, v in state.items() if not k.startswith('pos.pe')}
    missing_keys, unexpected_keys = model.load_state_dict(filtered_state, strict=False)

    if missing_keys:
        print(f"[INFO] Missing keys when loading state_dict: {missing_keys}")
    if unexpected_keys:
        print(f"[INFO] Unexpected keys in state_dict: {unexpected_keys}")

    prefix = os.path.splitext(model_path)[0]
    with open(prefix + "_word2idx.pkl", "rb") as f:
        word2idx = pickle.load(f)
    with open(prefix + "_ngram2idx.pkl", "rb") as f:
        ngram2idx = pickle.load(f)

    model.set_index_file(word2idx, ngram2idx)
    model.eval()
    return model

def handle_vector_cash(chunk_data, vector_cash):
    diff_data = chunk_data
    similar_chunk_index = -1
    for cash_index, cash_chunk_data in vector_cash.items():
        cash_delta_data = delta_encoding(chunk_data, cash_chunk_data)
        if len(cash_delta_data) < len(diff_data):
            diff_data = cash_delta_data
            similar_chunk_index = cash_index
    return diff_data, get_md5(similar_chunk_index)

def calculate_similar_chunk(chunk_data, annoy_tree, sliding_window, ngram_window, num_perm, model, top_n, is_add=False, shingles=True):
    current_chunk_index, similar_chunk_index_list, chunk_vector = (-1, -1, None)
    if annoy_tree.get_n_items() != 0: # 如果树中不为0
        current_chunk_index, similar_chunk_index_list, chunk_vector = get_similar_chunk_index(chunk_data, sliding_window=sliding_window, ngram_window=ngram_window, num_perm=num_perm, model=model, annoy_tree=annoy_tree, top_n=top_n, is_add=False, shingles=shingles)
    return current_chunk_index, similar_chunk_index_list, chunk_vector

def find_best_node(chunk_data, similar_chunk_index_list, to_dir_name):
    diff_data = chunk_data
    similar_chunk_hash = str()
    for similar_chunk_index in similar_chunk_index_list:
        temp_similar_chunk_hash = get_md5(similar_chunk_index)
        temp_similar_chunk_data = read_hash_table(temp_similar_chunk_hash, to_dir_name)
        temp = delta_encoding(chunk_data, temp_similar_chunk_data)
        if len(temp) < len(diff_data):  # if current chunk is similar chunk ---- len(temp) < len(chunk_data)
            diff_data = temp
            similar_chunk_hash = temp_similar_chunk_hash
    return diff_data, similar_chunk_hash

def deduplicationWithNoRecursion(dir_name, sliding_window, ngram_window, num_perm, shingles=True, model=None, annoy_tree=None, avg_size=512, fat=True, hf=md5, top_n=3):
    cache_queries = 0
    dce_annoy_sum = 0.0
    dce_cnt = 0
    annoy_queries = 0
    annoy_hits = 0
    min_size = avg_size // 2
    max_size = avg_size * 2
    to_dir_name = dir_name + "_card"
    folder_existed = os.path.exists(to_dir_name)
    before_size = 0
    after_size = 0
    base_file_size = 0
    annoy_time = 0
    find_best_node_time = 0
    vector_cash = OrderedDict() # {1:chunk_data, }
    if not folder_existed:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(to_dir_name)
    else:
        shutil.rmtree(to_dir_name)
        os.makedirs(to_dir_name)
    delta_index = 0
    origin_index = 0 # origin_index和cache对应，因为cache是一个先进先出的列表，每次弹出一个元素就放进annoy里面
    for home, _, files in os.walk(dir_name):
        for filename in files:
            file = os.path.join(home, filename)
            before_size += os.path.getsize(file)
            chunk_list = list(fastcdc(file, min_size=min_size, avg_size=avg_size, max_size=max_size, fat=fat, hf=hf))
            with open(file=file, mode="rb") as r:
                for chunk in chunk_list:
                    chunk_data = r.read(chunk.length)
                    annoy_time_start = time.time()
                    current_chunk_index, similar_chunk_index_list, chunk_vector = calculate_similar_chunk(chunk_data, annoy_tree, sliding_window, ngram_window, num_perm, model, top_n, shingles=shingles)
                    annoy_time_end = time.time()
                    annoy_time += (annoy_time_end - annoy_time_start)

                    if similar_chunk_index_list != -1 and len(similar_chunk_index_list) > 0: # 找到相似块
                        annoy_queries += 1
                        min_diff_annoy = chunk_data
                        for cand_idx in similar_chunk_index_list:
                            cand_hash = get_md5(cand_idx)
                            cand_data = read_hash_table(cand_hash, to_dir_name)
                            diff_tmp = delta_encoding(chunk_data, cand_data)
                            if len(diff_tmp) < len(min_diff_annoy):
                                min_diff_annoy = diff_tmp
                        dce_annoy_sum += len(min_diff_annoy) / len(chunk_data)
                        dce_cnt += 1
                        find_best_node_time_start = time.time()
                        diff_data, similar_chunk_hash = find_best_node(chunk_data, similar_chunk_index_list, to_dir_name)
                        find_best_node_time_end = time.time()
                        find_best_node_time += (find_best_node_time_end - find_best_node_time_start)
                        if len(diff_data) < len(chunk_data):
                            annoy_hits += 1
                        cash_diff_data = chunk_data
                        cash_similar_chunk_hash = None
                        if len(vector_cash) != 0:
                            cache_queries += 1
                            cash_diff_data, cash_similar_chunk_hash = handle_vector_cash(chunk_data, vector_cash)

                        _diff_data, _similar_chunk_hash = (diff_data, similar_chunk_hash) if len(diff_data) < len(cash_diff_data) else (cash_diff_data, cash_similar_chunk_hash)

                        if len(_diff_data) == len(chunk_data): # 并没有发现能够delta encoding的块
                            vector_cash[origin_index] = chunk_data
                            write_hash_table(get_md5(origin_index), chunk_data, to_dir_name)
                            origin_index += 1
                        else:
                            write_hash_table(get_md5(delta_index + 1), _diff_data, to_dir_name, is_delta=True, similar_chunk_hash=_similar_chunk_hash)
                            delta_index += 1
                    else:
                        if len(vector_cash) != 0:
                            cache_queries += 1
                        diff_data, similar_chunk_hash = handle_vector_cash(chunk_data, vector_cash)
                        if len(diff_data) < len(chunk_data):
                            write_hash_table(get_md5(delta_index + 1), diff_data, to_dir_name, is_delta=True, similar_chunk_hash=similar_chunk_hash)
                            delta_index += 1
                        else:
                            vector_cash[origin_index] = chunk_data
                            write_hash_table(get_md5(origin_index), chunk_data, to_dir_name)
                            origin_index += 1
                    if len(vector_cash) >=1:
                        cash_index, cash_chunk_data = vector_cash.popitem(last=False)
                        # cash_chunk_vector = get_vector(cash_chunk_data, model)
                        cash_chunk_vector = get_vector(model=model, data=cash_chunk_data, sliding_window=sliding_window, num_perm=num_perm,ngram_window=ngram_window,shingles=shingles)
                        add_annoy_node(cash_chunk_vector, annoy_tree)
    for root, dirs, files in os.walk(to_dir_name):
        after_size += sum([os.path.getsize(os.path.join(root, name)) for name in files])
    print("before size : " + str(before_size))
    print("after size : " + str(after_size))
    print("base file size : " + str(base_file_size))
    try:
        dcr = before_size / after_size
    except ZeroDivisionError:
        dcr = 0
    print("DCR : " + str(dcr))

    try:
        dcr_without_base_file = before_size / (after_size - base_file_size)
    except ZeroDivisionError:
        dcr_without_base_file = 0
    print("DCR without base file: " + str(dcr_without_base_file))
    print("Annoy time: " + str(annoy_time))
    print("find best node time: " + str(find_best_node_time))
    logger.info(f"Cache lookups (calls) : {cache_queries}")
    if annoy_queries:
        hit_rate = annoy_hits / annoy_queries
        logger.info(f"Annoy Hit Rate : {hit_rate:.2%} ({annoy_hits}/{annoy_queries})")
    else:
        logger.info("Annoy Hit Rate : n/a (no Annoy query)")
    if dce_cnt:
            avg_dce_annoy = dce_annoy_sum / dce_cnt
            logger.info(f"Avg DCE_annoy (no cache) : {avg_dce_annoy:.4f}")
    else:
            logger.info("Avg DCE_annoy (no cache) : n/a")
    return annoy_time, find_best_node_time

def delta_encoding(chunk_data, base_chunk_data):
    diff_data = chunk_data
    try:
        diff_data = xdelta3.encode(base_chunk_data, chunk_data)
    except BaseException:
        pass
    return diff_data
def write_hash_table(hash_value, data, base_dht, is_delta=False, similar_chunk_hash=None):
    key = hash_value[-3:-1]
    write_path = os.path.join(base_dht, 'delta', key, hash_value) if is_delta else os.path.join(base_dht, 'origin', key, hash_value)
    if not is_delta:
        dir_name = os.path.dirname(write_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        with open(write_path, 'wb') as writer:
            writer.write(data)
    else:
        os.makedirs(write_path)
        with open(os.path.join(write_path, hash_value), 'wb') as writer:
            writer.write(data)
        with open(os.path.join(write_path, similar_chunk_hash), 'wb') as create_new:
            pass

def read_hash_table(hash_value, base_dht):
    key = hash_value[-3:-1]
    read_path = os.path.join(base_dht, 'origin',  key, hash_value)
    if os.path.exists(read_path):
        with open(read_path, 'rb') as reader:
            return reader.read()
    else:
        return decode_data(hash_value, base_dht)
def decode_data(hash_value, base_dht):
    """
    迭代 返回原始数据块
    :param hash_value:
    :param base_dht:
    :return:
    """
    delta_stack = []
    key = hash_value[-3:-1]
    read_dir = os.path.join(base_dht, 'delta', key, hash_value)
    similar_chunk_hash = ''
    for file_name in os.listdir(read_dir):
        if file_name != hash_value:
            similar_chunk_hash = file_name
            break
    with open(os.path.join(read_dir, hash_value), 'rb') as reader:
        delta_chunk_data = reader.read()
        delta_stack.append(delta_chunk_data)

    while not is_origin_file(similar_chunk_hash, base_dht):
        with open(os.path.join(base_dht, 'delta', similar_chunk_hash[-3:-1], similar_chunk_hash, similar_chunk_hash), 'rb') as reader:
            delta_chunk_data = reader.read()
            delta_stack.append(delta_chunk_data)
        for file_name in os.listdir(os.path.join(base_dht, 'delta', similar_chunk_hash[-3:-1], similar_chunk_hash)):
            if file_name != similar_chunk_hash:
                similar_chunk_hash = file_name
                break
    origin_chunk_data = bytes()
    read_path = os.path.join(base_dht, 'origin', similar_chunk_hash[-3:-1], similar_chunk_hash)
    with open(read_path, 'rb') as reader:
        origin_chunk_data = reader.read()
    while len(delta_stack) > 0:
        origin_chunk_data = xdelta3.decode(origin_chunk_data, delta_stack.pop())

    return origin_chunk_data

def is_origin_file(hash_value, base_dht):
    key = hash_value[-3:-1]
    read_path = os.path.join(base_dht, 'origin', key, hash_value)
    if os.path.exists(read_path):
        return True
    return False

def get_md5(number):
    digest = md5()
    digest.update(str(number).encode('utf-8'))
    return digest.hexdigest()

def get_vector(model, data,
               sliding_window, num_perm,
               ngram_window, shingles):
    # 1) MinHash → token list
    _, tokens = get_minhash_from_chunk(
        chunk_data=data,
        sliding_window=sliding_window,
        num_perm=num_perm,
        shingles=shingles
    )
    # 2) n-gram list
    ngrams = get_n_grams(tokens, n=ngram_window)

    # 3) token → idx
    seq_idx = torch.LongTensor([
        model.word2idx.get(t, model.word2idx['<UNK>'])
        for t in tokens
    ]).unsqueeze(0).to(next(model.parameters()).device)

    ng_idx = torch.LongTensor([
        model.ngram2idx.get(g, model.ngram2idx['<UNK>'])
        for g in ngrams
    ]).unsqueeze(0).to(seq_idx.device)

    # 4) 向量
    return model.chunk2vec(seq_idx, ng_idx)     # (D,)
def get_similar_chunk_index(data, sliding_window, ngram_window, num_perm, model, annoy_tree, top_n, is_add=True, shingles=True):
    """
    主函数，接收数据块，返回最相似的index
    :param is_add:
    :param top_n:
    :param ngram_window:
    :param num_perm:
    :param sliding_window:
    :param n: 找最近的n个
    :param model: 模型
    :param data: 数据块内容
    :param annoy_tree: annoy 索引的加载路径
    :return: 最相似的数据块的index
    """
    vector = get_vector(model, data,sliding_window=sliding_window,num_perm=num_perm,ngram_window=ngram_window,shingles=shingles)
    return get_most_similar_index(vector, annoy_tree, n=top_n, is_add=is_add)
def add_annoy_node(vector, annoy_tree):
    annoy_tree.add_item(annoy_tree.get_n_items(), vector)
    return annoy_tree.get_n_items()
def get_most_similar_index(vector, annoy_tree, n, is_add=True):
    """
    :param is_add:
    :param n: 找最近的n个
    :param vector: 向量
    :param annoy_tree: annoy 索引
    :return: 当前数据块的index, 最相似的数据块的index
    """
    if annoy_tree.get_n_items() != 0:
        most_similar_index = annoy_tree.get_nns_by_vector(n=n, vector=vector)
        if is_add:
            annoy_tree.add_item(annoy_tree.get_n_items(), vector) # table.get_n_items() 为当前的annoy树中最后一个索引 + 1
            return annoy_tree.get_n_items() - 1, most_similar_index, vector
        else:
            return annoy_tree.get_n_items(), most_similar_index, vector
    else:
        annoy_tree.add_item(annoy_tree.get_n_items(), vector)
        return 0, -1, vector
if __name__ == '__main__':

    import argparse
    import sys

    # parser
    parser = argparse.ArgumentParser(description='return the most similar item of vector to the chunk')
    parser.add_argument('--n_layers', type=int, default=4,
                        help='Transformer encoder layers')
    parser.add_argument('--ff_dim', type=int, default=512,
                        help='feed-forward dimension')
    parser.add_argument('--tau', type=float, default=0.07,
                        help='InfoNCE temperature')
    parser.add_argument('--dataset', type=str, help="chunk bytes")
    parser.add_argument('--max_vocab', type=int, default=40000, help='max vocab size') # 50000
    parser.add_argument('--max_ngram', type=int, default=200000, help='max n-gram bag size') # 400000
    parser.add_argument('--hidden', type=int, default=128, help='hidden embedding dimension')
    parser.add_argument('--shingles', type=bool, default=True, help="use the new shingles algoritm")
    parser.add_argument('--sliding_window', type=int, default=512, help='window size for extract feature')
    parser.add_argument('--ngram_window', type=int, default=5, help='window size for extract feature')
    parser.add_argument('--num_perm', type=int, default=128, help='minhash size')
    parser.add_argument('--avg_size', type=int, default=262144, help='hidden embedding dimension')
    parser.add_argument('--top_n', type=int, default=1, help='the most top n tree node')
    parser.add_argument('--model_path', type=str, help="model path")
    args = parser.parse_args()

    annoy = AnnoyIndex(args.hidden, 'angular')
    annoy.build(1000)#  cos

    print("load model ...")
    start = time.time()
    model = get_model(args.model_path, args.max_vocab, args.max_ngram, args.hidden,args.n_layers,args.ff_dim, args.tau)
    end = time.time()
    print("cost: {}".format(end - start))
    print("deduplication ...".format(end-start))
    start = time.time()
    annoy_time, find_best_node_time = deduplicationWithNoRecursion(args.dataset, sliding_window=args.sliding_window, ngram_window=args.ngram_window,num_perm = args.num_perm, model=model, shingles=args.shingles, avg_size=args.avg_size, annoy_tree=annoy, top_n=args.top_n)
    end = time.time()
    print("cost: {}".format(end - start))
    print("cost - annoy: {}".format(end - start - annoy_time))
    print("cost - annoy - find best node: {}".format(end - start - annoy_time - find_best_node_time))
    print("done!")
    # print(get_similar_chunk_index(args.chunk, args.model_path, args.annoy_path, args.max_vocab, args.max_ngram, args.hidden, args.n)[0])