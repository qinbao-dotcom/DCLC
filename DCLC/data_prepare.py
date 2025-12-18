from collections import Counter
import numpy as np
from datasketch import MinHash
import base64
import os
from fastcdc import fastcdc

from card.utils import get_n_grams
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# 创建一个日志记录器
logger = logging.getLogger(__name__)

USE_RABIN = False
logger.info(f'use rabin: {USE_RABIN}')

class RabinFingerprint:
    def __init__(self, window_size, base=26, mod=10**9+7):
        """
        Initializes the Rabin Fingerprint object with the given parameters.
        :param window_size: the size of the rolling window
        :param base: the base for the polynomial hash function
        :param mod: the modulus for the polynomial hash function  
        """
        logger.info(f'Rabin parameters:')
        logger.info(f'  --window_size: {window_size}')
        logger.info(f'  --base: {base}')
        logger.info(f'  --mod: {mod}')
        logger.info(f'**' * 20)
        self.window_size = window_size
        self.base = base
        self.mod = mod
        self.power = 1  # Power of base to the window size
        for _ in range(window_size-1):
            self.power = (self.power * base) % mod
        self.hash = 0  # Current hash value
        self.window = []  # Current binary data in the window

    def update(self, new_byte):
        """
        Updates the hash value by sliding the window: remove oldest byte if needed and add new byte.
        :param new_byte: the new byte to add to the window and hash  
        """
        if not isinstance(new_byte, int):
            raise TypeError("Input must be an integer representing a byte.")
        
        if new_byte < 0 or new_byte >= 256:
            raise ValueError("Input must be an integer between 0 and 255 (a byte).")
        
        if len(self.window) == self.window_size:
            # Remove the leftmost byte's contribution  
            old_byte = self.window.pop(0)
            self.hash = (self.hash - self.power * old_byte) % self.mod
        # Add new byte
        self.window.append(new_byte)
        self.hash = (self.hash * self.base + new_byte) % self.mod

    def clear(self):
        """Resets the hash and the window."""
        self.hash = 0
        self.window = []

    def get_hash(self):
        """Returns the current hash value.
        :return: the current hash value
        """
        return self.hash

def get_rabin_from_chunk(chunk_data, sliding_window, num_perm, shingles=True):
    """
    ...
    :param num_perm: hash count
    :param chunk_data: chunk's byte content
    :param sliding_window: the numbers of bits for minhash each update
    :return: minhash: for debug   |    and base64 of minhash
    """
    primes = [1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069, 1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153, 1163, 1171, 1181, 1187, 1193, 1201, 1213, 1217, 1223, 1229, 1231, 1237, 1249, 1259, 1277, 1279, 1283, 1289, 1291, 1297, 1301, 1303, 1307, 1319, 1321, 1327, 1361, 1367, 1373, 1381, 1399, 1409, 1423, 1427, 1429, 1433, 1439, 1447, 1451, 1453, 1459, 1471, 1481, 1483, 1487, 1489, 1493, 1499, 1511, 1523, 1531, 1543, 1549, 1553, 1559, 1567, 1571, 1579, 1583, 1597, 1601, 1607, 1609, 1613, 1619, 1621, 1627, 1637, 1657, 1663, 1667, 1669, 1693, 1697, 1699, 1709, 1721, 1723, 1733, 1741, 1747, 1753, 1759, 1777, 1783, 1787, 1789, 1801, 1811, 1823, 1831, 1847, 1861, 1867, 1871, 1873, 1877, 1879, 1889, 1901, 1907, 1913, 1931, 1933, 1949, 1951, 1973, 1979, 1987, 1993, 1997, 1999]
    
    rf = RabinFingerprint(sliding_window)
    rabins = []
    avg_size = max(int(len(chunk_data) / num_perm), 256)
    print(f"avg_size: {avg_size}")
    sub_chunk_list = fastcdc(chunk_data, avg_size=avg_size, fat=True)
    for sub_chunk_info in sub_chunk_list:
        sub_chunk = chunk_data[sub_chunk_info.offset: sub_chunk_info.offset + sub_chunk_info.length]
        max_hash = 99999999
        for char in sub_chunk:
            rf.update(char)
            cur_hash = rf.get_hash()
            max_hash = max(cur_hash, max_hash)
        hex_rabin = hex(max_hash)[2:]
        logger.info(f'rabin: {hex_rabin}')
        rabins.append(max_hash)
        rf.clear()
    # RANKING
    if shingles:
        hex_rabins = []
        for rabin in rabins:
            # 计算每个质数的余数
            remainders = [int(rabin % prime) for prime in primes]

            # 使用字典统计每个余数出现的次数
            remainder_counts = {}
            for remainder in remainders:
                if remainder in remainder_counts:
                    remainder_counts[remainder] += 1
                else:
                    remainder_counts[remainder] = 1

            # 找到出现次数最多的余数，作为它的ID
            max_count_remainder = max(remainder_counts, key=remainder_counts.get)
            # number_bytes = struct.pack('i', minhash_digest)
            hex_rabin = hex(rabin)[2:]
            hex_rabins.append([hex_rabin, max_count_remainder])
        hex_rabins.sort(key=lambda x: x[1])
        hex_rabins = [x[0] for x in hex_rabins]
        
    else:
        hex_rabins = []
        for rabin in rabins:
            hex_rabin = hex(rabin)[2:]
            hex_rabins.append(hex_rabin)
    logger.info(f'hex_rabins: {hex_rabins}')
    return 0, hex_rabins

def get_minhash_from_chunk(chunk_data, sliding_window, num_perm, shingles=True):
    """
    ...
    :param num_perm: hash count
    :param chunk_data: chunk's byte content
    :param sliding_window: the numbers of bits for minhash each update
    :return: minhash: for debug   |    and base64 of minhash
    """
    if USE_RABIN: # 套壳
        return get_rabin_from_chunk(chunk_data, sliding_window, num_perm, shingles)
    primes = [1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069, 1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153, 1163, 1171, 1181, 1187, 1193, 1201, 1213, 1217, 1223, 1229, 1231, 1237, 1249, 1259, 1277, 1279, 1283, 1289, 1291, 1297, 1301, 1303, 1307, 1319, 1321, 1327, 1361, 1367, 1373, 1381, 1399, 1409, 1423, 1427, 1429, 1433, 1439, 1447, 1451, 1453, 1459, 1471, 1481, 1483, 1487, 1489, 1493, 1499, 1511, 1523, 1531, 1543, 1549, 1553, 1559, 1567, 1571, 1579, 1583, 1597, 1601, 1607, 1609, 1613, 1619, 1621, 1627, 1637, 1657, 1663, 1667, 1669, 1693, 1697, 1699, 1709, 1721, 1723, 1733, 1741, 1747, 1753, 1759, 1777, 1783, 1787, 1789, 1801, 1811, 1823, 1831, 1847, 1861, 1867, 1871, 1873, 1877, 1879, 1889, 1901, 1907, 1913, 1931, 1933, 1949, 1951, 1973, 1979, 1987, 1993, 1997, 1999]

    minhash = MinHash(num_perm=num_perm)
    minhash_digest_base64 = []
    minhash_digest_set = []
    avg_size = max(int(len(chunk_data) / num_perm), 256)
    logger.info(f"avg_size: {avg_size}")
    sub_chunk_list = fastcdc(chunk_data, avg_size=avg_size, fat=True)
    for sub_chunk_info in sub_chunk_list:
        logger.debug(f"sub chunk length: {sub_chunk_info.length}")
        sub_chunk = chunk_data[sub_chunk_info.offset: sub_chunk_info.offset + sub_chunk_info.length]
        for i in range(0, len(sub_chunk) - sliding_window + 1, sliding_window):
            minhash.update(sub_chunk[i:i+sliding_window])
        minhashs = minhash.digest()
        minhash_digest_set.append(max(minhashs))
        minhash.clear()
    
    if shingles:
        for minhash_digest in minhash_digest_set:
            # 计算每个质数的余数
            remainders = [int(minhash_digest % prime) for prime in primes]

            # 使用字典统计每个余数出现的次数
            remainder_counts = {}
            for remainder in remainders:
                if remainder in remainder_counts:
                    remainder_counts[remainder] += 1
                else:
                    remainder_counts[remainder] = 1

            # 找到出现次数最多的余数，作为它的ID
            max_count_remainder = max(remainder_counts, key=remainder_counts.get)
            # number_bytes = struct.pack('i', minhash_digest)
            # base64_str = base64.b64encode(minhash_digest).decode('utf8')
            hex_minhash_digest = hex(minhash_digest)[2:]
            
            minhash_digest_base64.append([hex_minhash_digest, max_count_remainder])
        minhash_digest_base64.sort(key=lambda x: x[1])
        
        minhash_digest_return = [x[0] for x in minhash_digest_base64] # 去除里面的ID
        return minhash, minhash_digest_return
    else:
        for minhash_digest in minhash_digest_set:
            # base64_str = base64.b64encode(minhash_digest).decode('utf8')
            hex_minhash_digest = hex(minhash_digest)[2:]
            minhash_digest_base64.append(hex_minhash_digest)
        return minhash, minhash_digest_base64

def get_corpus_for_train(dir_name, avg_size, num_perm, sliding_window, shingles=True):
    file_number = 0
    current_size = 0
    for home, _, files in os.walk(dir_name):
        for file in files:
            filename = os.path.join(home, file) # filename 为最终的文件名字，加了文件夹前缀
            # 配置fastcdc的分块大小
            min_size = avg_size / 2
            max_size = avg_size * 2

            # 存存放训练数据的文件夹
            learning_corpus_dir = dir_name + "_learning_corpus"
            if not os.path.exists(learning_corpus_dir): # 如果不存在则创建
                os.makedirs(learning_corpus_dir)

            # 准备工作完成后准备CDC分块
            chunk_set = fastcdc(filename, min_size=min_size, avg_size=avg_size, max_size=max_size, fat=True)
            write_file_name = os.path.join(learning_corpus_dir, file + "learning_file_" + str(file_number))
            with open(file=filename, mode='rb') as file_handler:
                with open(write_file_name, "a") as learning_file_handler:
                    for chunk in chunk_set:
                        # print(chunk) # for debug
                        file_handler.seek(chunk.offset)
                        chunk_data = file_handler.read(chunk.length)
                        # print(chunk_data) # for debug()
                        _, minhash_digest_base64 = get_minhash_from_chunk(chunk_data=chunk_data, sliding_window=sliding_window, num_perm=num_perm, shingles=shingles)
                        write_data = ','.join(minhash_digest_base64)
                        learning_file_handler.write(write_data + '\n')
                        current_size += chunk.length
            if current_size > 20 * 1024:
                file_number += 1
                current_size = 0

def establish_index(file_path, max_vocab, max_ngram, ngram_window):
    text = []
    if os.path.isdir(file_path):
        for home, _, file_list in os.walk(file_path):
            for file in file_list:
                file_name = os.path.join(home, file)
                with open(file_name, 'r') as f:
                    for line in f.readlines():
                        text.extend(line.split(','))
    else:
        with open(file_path, 'r') as f:
            for line in f.readlines():
                text.extend(line.split(','))
    # We can only use MAX_VOCAB - 1 words we use <UNK> as a word.
    vocab = dict(Counter(text).most_common(max_vocab - 1))
    # the count of <UNK> is text length - other words' count
    vocab['<UNK>'] = len(text) - np.sum(list(vocab.values()))
    print(type(vocab.keys()))
    ngram = dict(Counter(get_n_grams(list(vocab.keys()), n=ngram_window)).most_common(max_ngram - 1))
    ngram['<UNK>'] = len(ngram.keys()) - np.sum(list(vocab.values()))
    # save the mapping pair of word to index
    # bitmap
    ngram2idx = {word: i for i, word in enumerate(ngram.keys())}
    # idx2ngram = {i: word for i, word in enumerate(ngram.keys())}
    word2idx = {word: i for i, word in enumerate(vocab.keys())}
    idx2word = {i: word for i, word in enumerate(vocab.keys())}

    word_count = np.array([count for count in vocab.values()], dtype=np.float32)
    word_freqs = word_count / np.sum(word_count)
    # refer to original paper
    word_freqs = word_freqs ** (3. / 4.)
    return text, word2idx, ngram2idx, word_freqs, idx2word

# if __name__ == '__main__':  
#     import secrets

#     # 生成一个长度为4096字节的随机字节串
#     random_bytes = secrets.token_bytes(4096)
#     get_minhash_from_chunk(random_bytes, 5, 128)  
#     pass