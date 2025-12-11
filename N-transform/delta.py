import xdelta3 as delta


def get_diff_file(a, b, fp=None):
    diff_data = str()
    try:
        diff_data = delta.encode(a, b)
    except BaseException:
        print("delta error")
    length_b = len(b)
    length_r = len(diff_data)
    if diff_data is "":
        return -101
    elif length_r >= length_b:
        # path = str(fp).replace("diff", "origin")
        # diff_file = open(path, "ab+")
        # diff_file.write(bytes(b))
        # print("diff file is too big")
        return 101
    else:
        diff_file = open(fp, "ab+")
        diff_file.write(bytes(diff_data))
        diff_file.close()
        return 0

