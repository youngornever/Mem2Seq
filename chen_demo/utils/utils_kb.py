from config import *

DATA_NAME = ['R_phone', 'R_cuisine', 'R_address', 'R_location', 'R_number', 'R_price', 'R_rating']
task = 5

if task != 6:
    file_name = '/home/chenxiuyi/project/Mem2Seq/chenxiuyi/Mem2Seq/data/dialog-bAbI-tasks/dialog-babi-kb-all.txt'
else:
    file_name = '/home/chenxiuyi/project/Mem2Seq/chenxiuyi/Mem2Seq/data/dialog-bAbI-tasks/dialog-babi-task6-dstc2-kb.txt'


class KB_data(object):
    def __init__(self):
        self.data = {}
        for name_ in DATA_NAME:
            self.data[name_] = None


def kb_reader(file_):
    logging.info('reading KB file to a dict')
    kb_dict = {}
    cnt_line = 0    # total lines in the file
    cnt_kb = 0      # kb numbers of the kb file, one kb_item may occupy many lines
    show_times = 0  # show some lines for debug
    with open(file_) as fin:
        for line_ in fin:
            cnt_line += 1
            part1,part2 = line_.strip().split('\t')

            _, R_name, P_name = part1.split(' ')
            P_value = part2
            # logging.info(P_value)
            if R_name in kb_dict.keys():
                kb_dict[R_name].data[P_name] = P_value
            else:
                cnt_kb += 1
                kb_dict[R_name] = KB_data()
                kb_dict[R_name].data[P_name] = P_value

            show_times += 1
            if show_times == 10:        # let it to be a negative number to forbidden
                for kk in kb_dict.keys():
                    print(kk)
                    print(kb_dict[kk].data)
    logging.info('total {} lines'.format(cnt_line))
    logging.info('total {} KB items'.format(cnt_kb))
    logging.info('read done')
    return kb_dict


def generate_memory(sent, speaker, time):
    # generate_memory(r, "", "")
    '''copy from utils/utils_babi_mem2seq.py'''
    sent_new = []
    sent_token = sent.split(' ')
    if sent_token[1]=="R_rating":       # R_rating��һ����־; end of һ��KB��Ϣ
        sent_token = sent_token + ["PAD"]*(MEM_TOKEN_SIZE-len(sent_token))
    else:
        # TODO: list���򣿣���
        sent_token = sent_token[::-1] + ["PAD"]*(MEM_TOKEN_SIZE-len(sent_token))
    sent_new.append(sent_token)
    return sent_new


def api_call(cuisine, location, size, price, kb_dict):
    candidate_name = set()  # we use set as retrieve not in order
    for name_ in kb_dict.keys():
        if (cuisine in name_) and (location in name_) and (price in name_):
            if kb_dict[name_].data['R_number'] == size:
                candidate_name.add(name_)

    data = []
    cnt_kb = 0      # kb items in this api call
    for res_name in candidate_name:
        # temp_data =[]
        for property_name in DATA_NAME:
            if property_name=="R_rating":   # R_rating��һ����־; end of һ��KB��Ϣ
                cnt_kb += 1
                temp_data = ' '.join((res_name,property_name,kb_dict[res_name].data[property_name]))
                # temp_data = list((res_name,property_name,kb_dict[res_name].data[property_name])) + \
                #                                                                         ["PAD"] * (MEM_TOKEN_SIZE - 3)
            else:
                temp_data = ' '.join((res_name, property_name, kb_dict[res_name].data[property_name]))
                # temp_data = list((kb_dict[res_name].data[property_name],property_name,res_name)) + \
                #                                                                         ["PAD"] * (MEM_TOKEN_SIZE - 3)
            data.append([temp_data])
    logging.info('retrieve {} kb items in this api call'.format(cnt_kb))
    for res_name in candidate_name:
        logging.info(res_name)
    return data

kb_dict = kb_reader(file_name)


if __name__ == '__main__':
    kb_dict = kb_reader(file_name)

    kb_data = api_call(cuisine='spanish', location='madrid', size='two', price='cheap', kb_dict=kb_dict)
    print(kb_data)


