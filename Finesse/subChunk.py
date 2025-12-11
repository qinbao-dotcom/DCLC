#from rabin_origin.utils import RabinFingerprint
import rabin
import time
from hashlib import md5
from rabin import Rabin
from finesse_chunk import log_v1 as log
import rabin as rb
rb.set_max_block_size(2 ** 40)
rb.set_min_block_size(2 ** 40)


def extracting_sfs(string, n):
    length = len(string)
    subChunkSize = length // n
    window = 100
    remnant = length % n
    feature = [0] * n
    rabin = Rabin()
    rb.set_window_size(window)
    for m in range(n):
        ranges = subChunkSize
        if m == n - 1:
            ranges += remnant
        for i in range(ranges):
            seek = m * subChunkSize + i
            rabin.update(string[seek])
            FP = rabin.fingerprints()[0][2] % (2 ** 32)
            if feature[m] <= FP:
                feature[m] = FP
        rabin.clear()
    return feature


def get_fixed_chunks_by_bytes(data, method=md5, chunk_count=12):
    start_extract = time.perf_counter()
    features = extracting_sfs(data, chunk_count)
    end_extract = time.perf_counter()
    log.LogSystem.print_log("extract feature time : %s" % (end_extract - start_extract))
    file_size = len(data)
    avg_size = file_size // chunk_count
    over_size = file_size % chunk_count
    for i in range(chunk_count):
        start = i * avg_size
        end = (i + 1) * avg_size
        if i == chunk_count:
            end += over_size
        sub_data = data[start:end]
        yield SubChunk(data=sub_data, feature=features[i])





class SubChunk:
    identify = 1

    def __init__(self, data, feature):
        self.id = SubChunk.identify
        self.data = data
        self.feature = feature
        SubChunk.identify += 1

    def __str__(self):
        return "{\nid: " + str(self.id) + "\ndata: " + self.data + "\nfeature: " + str(self.feature) + "\n}"

    @staticmethod
    def get_features(fixed_chunks, method=md5, chunk_count=12, group_count=4):
        start_sort = time.perf_counter()
        features = []
        sub_count = chunk_count // group_count
        for group in range(group_count):
            tmp = []
            for sub in range(sub_count):
                chunk = next(fixed_chunks)
                tmp.append(chunk.feature)
            features.append(tmp)
            features[group].sort()
        end_sort = time.perf_counter()
        log.LogSystem.print_log("sort time : %s" % (end_sort - start_sort))
        SFs = []
        start_hash = time.perf_counter()
        for i in range(sub_count):
            hash_method = method()
            for k in range(group_count):
                hash_method.update(str(features[k].pop()).encode("UTF-8"))
            SFs.append(hash_method.hexdigest())
        end_hash = time.perf_counter()
        log.LogSystem.print_log("hash sfs time : %s" % (end_hash - start_hash))
        return SFs


# if __name__ == '__main__':

    # for i in range(191):
    #     sub_chunks = get_fixed_chunks_by_file("chinese" + str(i) + ".txt")
    #     result = SubChunk.get_features(sub_chunks, method=md5)
    #     print(result)
    # f = open("../test4.txt", "r")
    # content = f.read()
    # sfs = extracting_N_sf(content, 12)

    # chunks = get_fixed_chunks_by_file("/home/ubuntu/Public/test2.txt", chunk_count=16)
    # sfs = SubChunk.get_features(chunks, chunk_count=16, group_count=8)
    # for s in sfs:
    #     print(s)

