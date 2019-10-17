import random
from random import shuffle
from collections import defaultdict
import tqdm
from sql_parse import get_incorrect_sqls

def shuffle_list(some_list):
    randomized_list = some_list[:]
    while True:
        random.shuffle(randomized_list)
        again = False
        for a, b in zip(some_list, randomized_list):
            if a == b:
                again = True
                break
        if not again:
            return randomized_list

if __name__ =='__main__':
    splits = ['train', 'test', 'dev']
    random.seed(1)
    ddir = 'wikisql_data/'
    for split in splits:
        sql_file = open(ddir + split + '.in', 'r', encoding='utf-8')
        nl_file = open(ddir + split + '.out', 'r', encoding='utf-8')
        table_no_file = open(ddir + split + '_table_no.txt', 'r', encoding='utf-8')
        col_file = open(ddir + split + '_columns.txt', 'r', encoding='utf-8')
        tsv_file = open(split + '.tsv', 'w', encoding='utf-8')
        sql_list = []
        nl_list = []
        no_list = []
        for row in sql_file:
            sql_list.append(row.rstrip())
        for row in nl_file:
            nl_list.append(row.rstrip())
        for row in table_no_file:
            no_list.append(row.rstrip())

        assert(len(sql_list) == len(nl_list))
        assert(len(sql_list) == len(no_list))

        correct = []
        for i in range(len(sql_list)):
            correct.append('1' + '\t' + sql_list[i] + '\t' + nl_list[i] + '\n')

        incorrect = []

        no_dict = defaultdict(list)
        for i in range(len(no_list)):
            no_dict[no_list[i]].append((sql_list[i], nl_list[i]))
        
        for no, pair_list in tqdm.tqdm(no_dict.items()):
            if(len(pair_list) > 1 and (len(pair_list) == len(set(pair_list)))):
                s_list = []
                n_list = []
                for pair in pair_list:
                    s_list.append(pair[0])
                    n_list.append(pair[1])
                if(len(s_list) != len(set(s_list)) or len(n_list) != len(set(n_list))):
                    continue
                n_list = shuffle_list(n_list)
                for i in range(len(s_list)):
                    incorrect.append('0' + '\t' + s_list[i] + '\t' + n_list[i] + '\n')

        columns_list = []
        for row in col_file:
            columns = row.rstrip().split('||')
            del columns[-1]
            columns = [x.lower() for x in columns]
            columns_list.append(columns)

        for i in tqdm.tqdm(range(len(sql_list))):
            incorrect_sqls = get_incorrect_sqls(sql_list[i], columns_list[i])
            for incorrect_sql in incorrect_sqls:
                incorrect.append('0' + '\t' + incorrect_sql + '\t' + nl_list[i] + '\n')

        nl_list = shuffle_list(nl_list)
        for i in range(len(sql_list)):
            incorrect.append('0' + '\t' + sql_list[i] + '\t' + nl_list[i] + '\n')

        write_list = correct + incorrect
        shuffle(write_list)

        for line in write_list:
            tsv_file.write(line)
