import random
from random import shuffle

def shuffle_list(some_list):
    randomized_list = some_list[:]
    while True:
        random.shuffle(randomized_list)
        for a, b in zip(some_list, randomized_list):
            if a == b:
                break
            else:
                return randomized_list

if __name__ =='__main__':
    splits = ['train', 'test', 'dev']
    random.seed(1)
    ddir = 'data/'
    for split in splits:
        sql_file = open(ddir + split + '.in', 'r', encoding='utf-8')
        nl_file = open(ddir + split + '.out', 'r', encoding='utf-8')
        tsv_file = open(ddir + split + '.tsv', 'w', encoding='utf-8')
        sql_list = []
        nl_list = []
        for row in sql_file:
            sql_list.append(row.rstrip())
        for row in nl_file:
            nl_list.append(row.rstrip())

        assert(len(sql_list) == len(nl_list))

        correct = []
        for i in range(len(sql_list)):
            correct.append('1' + '\t' + sql_list[i] + '\t' + nl_list[i] + '\n')

        incorrect = []
        shuffle_list(nl_list)
        for i in range(len(sql_list)):
            incorrect.append('0' + '\t' + sql_list[i] + '\t' + nl_list[i] + '\n')

        write_list = correct + incorrect
        shuffle(write_list)

        for line in write_list:
            tsv_file.write(line)
