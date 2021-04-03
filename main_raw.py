from pprint import pprint
import bitarray
import copy
import math
import json


def save_rgb_arr(rgb_arr):
    with open('./image.json', 'w') as image:
        image.write(json.dumps(rgb_arr))


def YCbCrToRGB(Y, Cb, Cr):
    R = round(Y + 1.402 * (Cr-128))
    G = round(Y - 0.34414 * (Cb-128) - 0.71414 * (Cr-128))
    B = round(Y + 1.772 * (Cb-128))

    R = min(max(0, R), 255)
    G = min(max(0, G), 255)
    B = min(max(0, B), 255)

    return R, G, B


def fix_channel_tables(channel_tables):
    for t in range(1, len(channel_tables)):
        channel_tables[t][0][0] += channel_tables[t - 1][0][0]


markers = {
    "start": "ffd8",
    "comment": "fffe",
    "quant": 'ffdb',
    "baseline": 'ffc0',
    "progressive_baseline": 'ffc2',
    "huffman": "ffc4",
    "start_of_scan": 'ffda',
    "end": 'ffd9',

    "exif": 'ffe',
    "DRI": 'ffdd',
    # last n from 0 to 7 (3 bits)
    "rst": 'ffd'  # ff_dn
}


def to_zigzag_order(arr):
    assert len(arr) == 64, f'Zig zag got weird array with length {len(arr)}'
    level = 0
    pos = [0, 0]
    size = int(len(arr)**0.5)
    res = [[0 for j in range(size)] for i in range(size)]
#     print(f"{size=}")
    for i in range(len(arr)):
        index = pos[0] + pos[1]*size
        val = arr[i]
        res[pos[1]][pos[0]] = val
#         print(f"{index=}")
#         res.append(arr[index])
        if (level % 2 == 0 and pos[1] == 0) or (level % 2 == 1 and pos[1]+1 == size):
            pos[0] += 1
            level += 1
        elif (level % 2 == 1 and pos[0] == 0) or (level % 2 == 0 and pos[0]+1 == size):
            pos[1] += 1
            level += 1
        elif level % 2 == 0:
            pos[0] += 1
            pos[1] -= 1
        elif level % 2 == 1:
            pos[0] -= 1
            pos[1] += 1
        else:
            print("Weird")
    return res


def dfs(n, code=''):
    if n.letter is not None:
        yield n.letter, code
    if n.left:
        yield from dfs(n.left, code + '0')
    if n.right:
        yield from dfs(n.right, code + '1')


class LetterNode:
    letter = None
    value = 0
    left = None
    right = None
    parent = None

    def __lt__(self, other):
        return self.value < other.value

    def dict_repr(self):
        #         res = f"Letter Node\n\tLetter {self.letter}"
        #         if self.left:
        #             res += f"\n\tLeft {self.left}"
        #         if self.right:
        #             res += f"\n\tReft {self.right}"
        res = {
            'letter': self.letter
        }
        if self.left:
            res['left'] = self.left.dict_repr()
        if self.right:
            res['right'] = self.right.dict_repr()
        return res


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def bytes2int(b):
    return int.from_bytes(b, "big")


def get_len(f):
    return bytes2int(f.read(2))


def get_quants(quant_raw):
    mute = True
    tables_raw = [int(i) for i in quant_raw]
    tables = []
    for i in range(0, len(tables_raw), 64 + 1):
        meta = quant_raw[i]
        quant = {}
        quant["table_id"] = meta & 0x0f
        quant["item_len"] = (meta & 0xf0) >> 4
        quant["item_len_repr"] = 1 if quant["item_len"] == 0 else 2
        if not mute:
            print(
                f"Get quant. Meta. Id: {quant['table_id']}, Item len: {quant['item_len_repr']} bytes")
        quant["table"] = to_zigzag_order(
            [int(i) for i in quant_raw[i+1: i+1+64]])
        tables.append(quant)
    # quant["table"] = [
    #     [int(item, 16) for item in chunks(line, 2)]
    #     for line in chunks(quant_raw[1:].hex(), 16)
    # ]
    return tables


def get_baseline(meta):
    baseline = {}
    baseline["precision"] = meta[0]
    baseline["height"] = bytes2int(meta[1:3])
    baseline["width"] = bytes2int(meta[3:5])
    baseline["num_channels"] = bytes2int(meta[5:6])
    baseline["channels"] = []
    channels_data = meta[6:]
    for i in range(baseline["num_channels"]):
        channel_meta = bytes2int(channels_data[i*3:i*3+3])
        channel_id = (channel_meta & 0xff_00_00) >> 4*4
        hor_dwn = (channel_meta & 0x00_f0_00) >> 3*4
        ver_dwn = (channel_meta & 0x00_0f_00) >> 2*4
        quant_id = channel_meta & 0x00_00_ff
        baseline["channels"].append({
            "id": channel_id,
            "hor_dwn": hor_dwn,
            "ver_dwn": ver_dwn,
            "quant_id": quant_id
        })
    return baseline


def build_tree(code_lens, codes):
    root = LetterNode()
    cur = root
    code_index = 0
    for code_len, codes_count in enumerate(code_lens):
        #         print(f"Doing huffman for code length {code_len+1} for amout {codes_count}")
        if codes_count == 0:
            continue
        for count in range(codes_count):
            visited = []
            level = 0
            safe_n = 0
            while level < code_len + 1 and safe_n < 1000:
                temp = cur
                if cur.left is None:
                    cur.left = LetterNode()
                    cur = cur.left
                    cur.parent = temp
                    level += 1
                elif (cur.left is not None) and (cur.left.letter is None) and (cur.left not in visited):
                    cur = cur.left
                    cur.parent = temp
                    level += 1
                elif cur.right is None:
                    cur.right = LetterNode()
                    cur = cur.right
                    cur.parent = temp
                    level += 1
                elif (cur.right is not None) and (cur.right.letter is None) and (cur.right not in visited):
                    cur = cur.right
                    level += 1
                elif cur in visited:
                    level -= 1
                    cur = cur.parent

                if (cur.left is not None) and (cur.left.letter is not None):
                    visited.append(cur.left)
                if (cur.right is not None) and (cur.right.letter is not None):
                    visited.append(cur.right)
                if (cur.left in visited) and (cur.right in visited):
                    visited.append(cur)
                safe_n += 1
            if safe_n == 1000:
                print("SOME ERROR OCCURED ON HUFFMAN")
            cur.letter = codes[code_index]
            cur = root
            code_index += 1
#     pprint(root.dict_repr())
    h = list(dfs(root))
    return h


def get_huffmans(meta):
    int_meta = [int(i) for i in meta]
    safe_n = 0
    trees = []
    while len(int_meta) > 0 and safe_n < 1000:
        info = int_meta[0]
        h_class = (info & 0xf0) >> 4
        h_id = info & 0x0f
        code_lens = int_meta[1:17]
        last_index = 17 + sum(code_lens)
        codes = int_meta[17:last_index]
        int_meta = int_meta[last_index:]
        tree = build_tree(code_lens, codes)
        trees.append({"class": h_class, "class_repr": "DC" if h_class ==
                      0 else "AC", "id": h_id, "table": tree})
    return trees


def get_scan_info(meta):
    scan_info = {"channels": []}
    scan_info["num_channels"] = meta[0]
    # print(f"num_channels: {meta[0]}")
    channels_data_raw = meta[1: 1 + scan_info["num_channels"]*2]
    for i in range(scan_info["num_channels"]):
        channel_data = bytes2int(channels_data_raw[i*2:i*2+2])
        channel_id = (channel_data & 0xff_00) >> 2*4
        huff_dc = (channel_data & 0x00_f0) >> 1*4
        huff_ac = (channel_data & 0x00_0f) >> 0*4
        scan_info["channels"].append({
            "channel_id": channel_id,
            "huff_dc": huff_dc,
            "huff_ac": huff_ac
        })
#         print(f"{channel_data=}")
    return scan_info


def get_mat(reverse_dict_dc, reverse_dict_ac, byte_rest):
    assert len(byte_rest) != 0, f"Get mat got byte_rest with zero length"
    mute = True
    Mat = []
    safe_n = 0
# DC
    buffer = bitarray.bitarray(endian='big')
    while True and safe_n < 10000:
        safe_n += 1
        str_repr = buffer.to01()
        if str_repr in reverse_dict_dc:
            value = reverse_dict_dc[str_repr]
            if not mute:
                print(
                    f"Found DC coeff in huffman {str_repr} with value {value}")
            if value == 0:
                Mat.append(0)
                if not mute:
                    print(f"DC is equal to zero. Proceeding")
            else:
                koeff_str = byte_rest[:value].to01()
                koeff = int(koeff_str, 2)
                if koeff_str[0] == '0':
                    koeff = koeff - 2**len(koeff_str) + 1
                byte_rest = byte_rest[value:]
                Mat.append(koeff)
                if not mute:
                    print(
                        f"DC is non-zero. Reading {value} bits. Got koeff {koeff} from {koeff_str}. Rest {byte_rest.to01()[:10]}...")
            buffer.clear()
            break
        else:
            buffer += byte_rest[0:1]
            byte_rest = byte_rest[1:]
    if len(byte_rest) == 0:
        print("Weird thing in huffman. Byte_rest empty after dc koeff")
    # print(f"Mat len {len(Mat)}")
# AC
    safe_n = 0
    while True and safe_n < 1000:
        safe_n += 1
        str_repr = buffer.to01()
        if str_repr in reverse_dict_ac:
            # first_reset = byte_rest.to01().find('1111111111010000')
            # print(f"From get mat {first_reset=}")
            value = reverse_dict_ac[str_repr]
            if not mute:
                print(
                    f"Found AC koeff in huffman {str_repr} with value {value}. Mat len was {len(Mat)}")
            if value == 0:
                if not mute:
                    print(f"Zero AC koeff. Filling rest with zeros")
                while len(Mat) < 8*8:
                    Mat.append(0)
            else:
                zeros_count = (value & 0xf0) >> 4
                meaning_len = (value & 0x0f)
                koeff_str = byte_rest[:meaning_len].to01()
                assert koeff_str != '', f"koeff_str empty. meaning len {meaning_len}, zeros {zeros_count}"
                byte_rest = byte_rest[meaning_len:]
                koeff = int(koeff_str, 2)
                if koeff_str[0] == '0':
                    koeff = koeff - 2**len(koeff_str) + 1
                for _ in range(zeros_count):
                    Mat.append(0)
                    if len(Mat) == 8*8:
                        break
                Mat.append(koeff)
                if not mute:
                    print(
                        f"AC is non-zero. Reading {meaning_len} bits. Got {zeros_count} zeros and {koeff} meaning. Rest {byte_rest.to01()[:50]}...")
            buffer.clear()
            if len(Mat) == 8*8:
                if not mute:
                    print(f"Table is 64 in len")
                break
        else:
            buffer += byte_rest[0:1]
            byte_rest = byte_rest[1:]
    # print(f"Mat:\n{Mat}\n\n")
    return to_zigzag_order(Mat), byte_rest


def get_huff_table(huffmans, h_class, h_id):
    return [x for x in huffmans if x["class"] == h_class and x["id"] == h_id][0]["table"]


def mat_mut(A, B):
    assert len(A) == len(B), "Matricies must be same size"
    res = copy.deepcopy(A)
    for l in range(len(res)):
        for i in range(len(res[l])):
            res[l][i] *= B[l][i]
    return res


def process(file_name):
    with open(file_name, 'rb') as f:
        marker = f.read(2)
        if marker.hex() != markers["start"]:
            print(f"Seems like its not a JPG image. Got {marker}. Aborting!")
            return
        else:
            print("Looks like JPG image. Trying to process")

#         quant = {
#             "table_id": None,
#             "item_len": None,
#             "table": None
#         }
#         channel = {
#             "id": None,
#             "hor_dwn": None,
#             "ver_dwn": None,
#             "quant_id": None
#         }
#         huffman = {
#             "class": None,
#             "id": None
#         }
        quants = []
        baseline = {
            "precision": None,
            "height": None,
            "width": None,
            "num_channels": None,
            "channels": []
        }
        huffmans = []
        scan_info = {
            "num_channels": None,
            "channels": [],
            #             ..... Start of spectral or predictor selection, End of spectral selection, Successive approximation bit position
        }
        data_tables = []
        data_tables_quant_mul = []
        data_tables_idct = []
        data_tables_rgb = []
        reset_interval = None
        resest_count = 0

        while True and marker != b'':
            marker = f.read(2)
            hex_repr = marker.hex()
            if hex_repr == markers['comment']:
                l = get_len(f)
                print(f"Comment section of length {l}")
                comment = f.read(l - 2)
                print(f"Comment: {comment}")
            elif hex_repr.startswith(markers['exif']):
                l = get_len(f)
                print(f"Exif section {hex_repr} of length {l}")
                comment = f.read(l - 2)
                print(f"Exif: {comment}")
            elif hex_repr == markers['DRI']:
                l = get_len(f)
                print(f"DRI section of length {l}")
                reset_interval = bytes2int(f.read(l - 2))
                print(f"DRI: {reset_interval}")
            elif hex_repr.startswith(markers['rst']) and '01234567'.find(hex_repr[-1]) > -1:
                # l = get_len(f)
                circl = bytes2int(marker) & 0b111
                print(f"RST section, number: {circl}")
                # reset_interval = bytes2int(f.read(l - 2))
                # print(f"DRI: {reset_interval}")
            elif hex_repr == markers['quant']:
                l = get_len(f)
                print(f"Quant tables sector with length {l}")
                quant_raw = f.read(l - 2)
                quants += get_quants(quant_raw)
#                 pprint(f"Meta quant {meta}")
                # print(f"Table id: {quant['table_id']}")
                # print(f"Item len: {quant['item_len']}")
                print(f'Got {len(quants)} quant tables.')
            elif hex_repr == markers['baseline']:
                l = get_len(f)
                print(f"baseline with length {l}")
                meta = f.read(l - 2)
                baseline = get_baseline(meta)
                pprint(baseline)
            elif hex_repr == markers["progressive_baseline"]:
                print("Codec does not support progressive images. Aborting.")
                return
            elif hex_repr == markers['huffman']:
                l = get_len(f)
                print(f"huffman with length {l}")
                meta = f.read(l - 2)
                huffmans += get_huffmans(meta)
            elif hex_repr == markers['start_of_scan']:
                l = get_len(f)
                print(f"start_of_scan with length {l}")
                meta = f.read(l - 2)
                scan_info = get_scan_info(meta)
                print(f"{scan_info=}")
                rest = f.read()
                end_mark = rest[-2:].hex()
                # print(
                # f"Reading rest of length {len(rest)}\n\n\n\n\n{rest.hex()}")
                if end_mark == markers['end']:
                    print("Successfully ended reading file")
                rest = rest[:-2]
                # print(rest)
                filter_stuffing = [rest[i: i+2].hex()
                                   for i in range(0, len(rest) + 1, 2)]
                print(
                    f"Len before pruning {len(''.join(filter_stuffing))/2} bytes")
                # filter_stuffing = [(i if i != 'ff00' else 'ff')
                #                    for i in filter_stuffing]
                # filter_stuffing = ''.join(filter_stuffing)
                filter_stuffing = ','.join(
                    filter_stuffing).replace('ff,00', 'ff').replace(',','')
                filter_stuffing = [filter_stuffing[i:i+2]
                                   for i in range(0, len(filter_stuffing)+1, 2)]
                filter_stuffing = [i for i in filter_stuffing if i != '']
                filter_stuffing = [int(i, base=16).to_bytes(
                    1, "big") for i in filter_stuffing]
                filter_stuffing = b''.join(filter_stuffing)
                print(f"Len afteer pruning {len(filter_stuffing)/2} bytes")

                print(f"{filter_stuffing=}")
                byte_rest = bitarray.bitarray(endian='big')
                byte_rest.frombytes(filter_stuffing)
                first_reset = byte_rest.to01().find('1111111111010000')
                secret_stuffing = len(
                    byte_rest.to01().split('1111111100000000')) - 1
                print(f"{first_reset=}")
                print(f"{secret_stuffing=}")
                # pprint(byte_rest)

                current_tables_id = 0
                current_class = 0
                buffer = bitarray.bitarray(endian='big')
#                 while True and safe_n < 1000:
#                     safe_n += 1
                # print(f"{huffmans}")
                total_mats_count = 0
                for b in baseline["channels"]:
                    total_mats_count += b['hor_dwn'] * b['ver_dwn']
                print(f"{total_mats_count=}")
                safe_n = 0
                while len(byte_rest) > 7 and safe_n < 400:
                    if reset_interval is not None:
                        print(f"{reset_interval=}")
                        # (safe_n % (reset_interval // 3)) == 0
                        is_last_in_reset = ((safe_n+1) % (reset_interval)) == 0
                        print(f"{is_last_in_reset=}")
                    print(f"byte_rest len {len(byte_rest)}")
                    safe_n += 1
                    print(
                        f"Decoding {safe_n}th Block. Rest: {byte_rest.to01()[:30]}...")
                    for chan in scan_info["channels"]:
                        first_reset = byte_rest.to01().find('1111111111010000')
                        huff_dc = get_huff_table(huffmans, 0, chan["huff_dc"])
                        huff_ac = get_huff_table(huffmans, 1, chan["huff_ac"])
                        # print(f"{huff_dc=}\n{huff_ac=}")
                        reverse_dict_dc = dict([(code, letter)
                                                for letter, code in huff_dc])
                        reverse_dict_ac = dict([(code, letter)
                                                for letter, code in huff_ac])
                        chan_mat_info = [i for i in baseline["channels"]
                                         if i['id'] == chan["channel_id"]][0]
                        chan_mat_count = chan_mat_info['hor_dwn'] * \
                            chan_mat_info['ver_dwn']
                        channel_tables = []
                        channel_tables_quant_mul = []
                        # pprint(quants)
                        quant_table = [i for i in quants if i["table_id"]
                                       == chan_mat_info["quant_id"]][0]["table"]
                        # print(f"{quant_table=}")
                        for chan_mat in range(chan_mat_count):
                            first_reset = byte_rest.to01().find('1111111111010000')
                            print(f"Before get mat {first_reset=}")
                            Mat, byte_rest = get_mat(
                                reverse_dict_dc, reverse_dict_ac, byte_rest)
                            channel_tables.append(Mat)
                            # channel_tables_quant_mul.append(
                            #     mat_mut(Mat, quant_table))
                            # pprint(Mat)
                        fix_channel_tables(channel_tables)
                        for ch_t in channel_tables:
                            channel_tables_quant_mul.append(
                                mat_mut(ch_t, quant_table))
                        data_tables.append(channel_tables)
                        data_tables_quant_mul.append(channel_tables_quant_mul)
                    if reset_interval is not None and is_last_in_reset:
                        index = format(resest_count % 8, '03b')
                        print(f"Reset index bin: {index}")
                        resest_count += 1
                        shift_len = byte_rest.to01().find('1111111111010' + index)
                        if shift_len == -1:
                            print(f"Reset marker not fount proceeding further.")
                        else:
                            reset_marker = byte_rest[shift_len: shift_len +
                                                     2*8].tobytes().hex()
                            print(f"{shift_len=}")
                            print(f"{reset_marker=}")
                            print(f"Rest before: {byte_rest[:18]}...")
                            byte_rest = byte_rest[shift_len + 2*8:]
                            print(f"Rest after: {byte_rest[:18]}...")
                # pprint(data_tables_quant_mul)
                # print(f"{byte_rest=}")
            elif hex_repr == markers['end']:
                print(f"End of image")
            else:
                print(f"Unrecognized byte {hex_repr} from {marker}")
                # break
        # end parse
        # Inverse descrete cosine transform
        print(
            f"Performing inverse descrete cosine transform for {len(data_tables_quant_mul)} blocks*channels")
        one_over_sqrt_2 = 2**(-1/2)
        for chan in data_tables_quant_mul:
            print(f"Performing idct for {len(chan)} tables in channel*block")
            channel_idct_tables = []
            for table in chan:
                idct_table = copy.deepcopy(table)
                for y_row in range(len(table)):
                    for x_column in range(len(table[y_row])):
                        s_yx = 0
                        for v_row in range(len(table)):
                            for u_column in range(len(table[v_row])):
                                C_row = one_over_sqrt_2 if v_row == 0 else 1
                                C_column = one_over_sqrt_2 if u_column == 0 else 1
                                s_yx += table[v_row][u_column] * C_row * C_column * \
                                    math.cos((2*x_column + 1) *
                                             u_column * math.pi / 16) * \
                                    math.cos((2*y_row + 1) *
                                             v_row * math.pi / 16)
                        s_yx /= 4
                        # print(f"{y_row=}, {x_column=}")
                        idct_table[y_row][x_column] = min(
                            max(round(s_yx) + 128, 0), 255)
                channel_idct_tables.append(idct_table)
            data_tables_idct.append(channel_idct_tables)
        pprint(data_tables_idct)
        print("Performing RGB conversion")
        for channel_block_index in range(0, len(data_tables_idct), 3):
            ychan = data_tables_idct[channel_block_index]
            print(f"Lume tables count {len(ychan)}")
            y_chan_size = 2  # int(len(ychan)**(1/2))
            # for row in range(baseline['height']):
            #     for col in range(baseline['width']):
            #         index = row * baseline['height'] + col
            #         Y = ychan[row]
            for table_index in range(len(ychan)):
                table = ychan[table_index]
                # print(f"HUUUUI {table=}")
                table_rgb = copy.deepcopy(table)
                for y in range(len(table)):
                    for x in range(len(table[y])):
                        dwn_y = 4*(table_index // y_chan_size) + (y // 2)
                        dwn_x = 4*(table_index % y_chan_size) + (x // 2)
                        R, G, B = YCbCrToRGB(
                            table[y][x],
                            data_tables_idct[channel_block_index +
                                             1][0][dwn_y][dwn_x],
                            data_tables_idct[channel_block_index +
                                             2][0][dwn_y][dwn_x]
                        )
                        # print(f"{B=}")
                        table_rgb[x][y] = [R, G, B]
                data_tables_rgb.append(table_rgb)
        # pprint(data_tables_rgb)
        save_rgb_arr(data_tables_rgb)


process('lein.jpeg')
