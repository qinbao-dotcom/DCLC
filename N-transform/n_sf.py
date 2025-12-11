import shutil
from fastcdc import fastcdc
from hashlib import md5
import os
import gc
def cdc_chunking(dir_name, avg_size=4096, fat=True, hf=md5, subChunk_count = 12, per_package_count=4, maxsize=None, similarity=0.3):

    min_size = avg_size // 2
    if maxsize is None:
        max_size = avg_size * 2
    else:
        max_size = maxsize
    diff_count = 0
    origin_count = 0
    to_dir_name = dir_name + "_sfs"
    folder = os.path.exists(to_dir_name)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(to_dir_name)
    else:
        shutil.rmtree(to_dir_name)
        os.makedirs(to_dir_name)
    group_count = subChunk_count//per_package_count
    before_size = 0
    for home, dirs, files in os.walk(dir_name):
        for filename in files:
            file = os.path.join(home, filename)
            before_size += os.path.getsize(file)
            cdc = list(fastcdc(file, min_size=min_size, avg_size=avg_size, max_size=max_size, fat=fat, hf=hf))
            with open(file=file, mode="rb") as r:
                for i in cdc:
                    local_similarity = 0
                    r.seek(i.offset)
                    content = r.read(i.length)
                    # 得到每个子块的super-features
                    sub_chunks = subChunk.get_fixed_chunks_by_bytes(str(content), chunk_count=subChunk_count)
                    sf = subChunk.SubChunk.get_features(sub_chunks, chunk_count=subChunk_count, group_count=per_package_count)
                    del sub_chunks
                    gc.collect()
                    flag = False
                    similarity_chunk_index = 0
                    count = 0
                    try:
                        with open(to_dir_name + "/features", "r") as features:
                            for line in features:
                                for k in range(group_count):
                                    if sf[k] in line:
                                        local_similarity += 1
                                    if local_similarity/group_count >= similarity:
                                        flag = True
                                        similarity_chunk_index = count
                                        break
                                if flag:
                                    break
                                count += 1
                    except BaseException:
                        pass

                    if flag:
                        # TODO 使用delta encoding 保存文件
                        try:
                            a_file = open(to_dir_name+"/origin"+str(similarity_chunk_index), "rb")
                            a_data = a_file.read()
                            a_file.close()
                            state = delta.get_diff_file(a_data, content, fp=to_dir_name+"/"+"diff"+str(diff_count))
                            if state == 101:
                                with open(to_dir_name + "/origin" + str(origin_count), "wb+") as origin:
                                    origin.write(content)
                                origin_count += 1
                                with open(to_dir_name+"/features", "a") as write:
                                    write_data = str()
                                    for sf_index in range(len(sf) - 1):
                                        write_data += sf[sf_index] + ","
                                    write_data += sf[sf_index + 1]
                                    write.write(write_data + "\n")
                        except BaseException:
                            print("no such file ")
                    else:
                        # TODO 保存文件，以及super-feature
                        with open(to_dir_name + "/features", "a") as write:
                            write_data = str()
                            for sf_index in range(len(sf) - 1):
                                write_data += sf[sf_index] + ","
                            write_data += sf[sf_index + 1]
                            write.write(write_data + "\n")
                        with open(to_dir_name+"/origin"+str(origin_count), "wb+") as origin:
                            origin.write(content)
                        origin_count += 1
                    try:
                        if os.path.getsize(to_dir_name + "/" + "diff" + str(diff_count)) > 10 * (2 ** 20):
                            diff_count += 1
                    except BaseException:
                        pass
                    gc.collect()
    after_size = 0
    for root, dirs, files in os.walk(to_dir_name):
        after_size += sum([os.path.getsize(os.path.join(root, name)) for name in files])
    print("before size : " + str(before_size))
    print("after size : " + str(after_size))
    try:
        DCR = before_size / after_size
    except ZeroDivisionError:
        DCR = 0
    print("DCR : " + str(DCR))



class Chunk:
    def __init__(self, name, offset, data, length, features=None):
        self.name = name
        self.features = features
        self.offset = offset
        self.data = data
        self.length = length
        filename = "result_file/" + self.features
        with open(filename, "wb+") as f:
            f.write(self.data)


if __name__ == '__main__':
    import argparse
    import sys
    import time
    # import the pkg
    sys.path.extend(["../"])
    from N_SF_chunk import subChunk, delta
    import N_SF_chunk.log_v1 as log
    # parser
    parser = argparse.ArgumentParser(description='from dataset get corpus and model')
    parser.add_argument('--corpus_dir', type=str, help=" the dataset path")
    parser.add_argument('--average_size', type=int, default=524288, help="the chunk's average size (unit: B)")
    parser.add_argument('--subChunk_count', type=int, default=12, help="the chunk's count")
    parser.add_argument('--per_package_count', type=int, default=4, help="the per chunk's count")
    parser.add_argument('--similarity', type=float, default=0.3, help="if the similarity more than the threshold then it's a similar chunk")

    args = parser.parse_args()

    # get params
    corpus_dir = args.corpus_dir
    average_size = args.average_size
    subChunk_count = args.subChunk_count
    per_package_count = args.per_package_count
    similarity = args.similarity

    # log
    l = log.LogSystem()
    l.print_log("corpus_dir : %s" % corpus_dir)
    l.print_log("average_size : %d" % average_size)
    l.print_log("subChunk_count : %d" % subChunk_count)
    l.print_log("per_package_count : %d" % per_package_count)
    l.print_log("similarity : %f" % similarity)

    # get corpus
    start = time.perf_counter()

    # train
    cdc_chunking(corpus_dir, avg_size=average_size, subChunk_count=subChunk_count, per_package_count=per_package_count, similarity=similarity)
    end = time.perf_counter()
    print(str(end - start))
