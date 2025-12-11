
from hashlib import md5
from rabin import Rabin
import rabin as rb


def extracting_N_sf(string, n):
    length = len(string)
    subChunkSize = length // n
    window = 100
    # rabin = RabinFingerprint(window)
    rb.set_max_block_size(2 ** 40)
    rb.set_min_block_size(2 ** 40)
    rabin = Rabin()
    rb.set_window_size(window)
    feature = [0] * n
    for m in range(length):
        rabin.update(string[m])
        # FP = rabin.fingerprints()[0][2] % (2 ** 32)
        for i in range(n):
            FP = rabin.fingerprints()[0][2] % (2 ** 32)
            if feature[i] <= FP:
                feature[i] = FP
    return feature

def get_fixed_chunks_by_bytes(data, method=md5, chunk_count=12):
    features = extracting_N_sf(data, chunk_count)
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
        sub_count = chunk_count // group_count
        SFs = []
        for group in range(sub_count):
            tmp = []
            rabin_value = object()
            window = 100
            # rabin = RabinFingerprint(window)
            rabin = Rabin()
            rb.set_max_block_size(2 ** 40)
            rb.set_min_block_size(2 ** 40)
            rb.set_window_size(window)
            for sub in range(group_count):
                chunk = next(fixed_chunks)
                tmp.append(chunk.feature)
                rabin.update(str(chunk.feature))
                rabin_value = rabin.fingerprints()[0][2] % (2 ** 32)
            hash_method = method()
            hash_method.update(str(rabin_value).encode("UTF-8"))
            SFs.append(hash_method.hexdigest())
        return SFs
        # for i in range(sub_count):
        #     hash_method = method()
        #     for k in range(group_count):
        #         hash_method.update(str(features[k].pop()).encode("UTF-8"))
        #     SFs.append(hash_method.hexdigest())
        # return SFs

