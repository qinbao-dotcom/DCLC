import os
from hashlib import md5
from fastcdc import fastcdc
from finesse_chunk import subChunk, delta


def cdc_get(dir_name, avg_size=4096, fat=True, hf=md5, subChunk_count = 12, per_package_count=4, maxsize=None, result_file="E:/finesse"):
    result = open(result_file, "a+")
    min_size = avg_size // 2
    count = 0
    if maxsize is None:
        max_size = avg_size * 2
    else:
        max_size = maxsize
    for home, dirs, files in os.walk(dir_name):
        for filename in files:
            fp = os.path.join(home, filename)
            cdc = list(fastcdc(fp, min_size=min_size, avg_size=avg_size, max_size=max_size, fat=fat, hf=hf))
            with open(file=fp, mode="r", encoding='gb18030', errors='ignore') as r:
                for i in cdc:
                    local_similarity = 0
                    r.seek(i.offset)
                    content = r.read(i.length)
                    chunk = open("E:/finesse_result/corpus"+str(count), "w+")
                    chunk.write(content)
                    # 得到每个子块的super-features
                    sub_chunks = subChunk.get_fixed_chunks_by_bytes(str(content), chunk_count=subChunk_count)
                    sf = subChunk.SubChunk.get_features(sub_chunks, chunk_count=subChunk_count,
                                                        group_count=per_package_count)
                    sfs_str = str()
                    for k in sf:
                        sfs_str += k
                        sfs_str += " "
                    result.write(sfs_str+"\n")
                    count += 1


if __name__ == '__main__':
    cdc_get("kernel", avg_size=512, maxsize=1024)
