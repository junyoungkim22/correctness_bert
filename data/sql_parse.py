import sys
import re
import random

random.seed(1)


def sql_parse(sql):
    #print(sql)
    words = sql.split()
    #print(words)
    length = len(words)

    i = 0

    assert(words[i].lower() == 'symselect')

    i = i + 1

    assert(words[i].lower() == 'symagg')
    i = i + 1

    agg = words[i]

    if(agg.lower() == 'symcol'):
        agg = None
    else:
        i = i + 1

    assert(words[i].lower() == 'symcol')
    i = i + 1

    sel_col_words = []

    while((words[i].lower() != 'symwhere') & (words[i].lower() != 'symend')):
        sel_col_words.append(words[i])
        i = i + 1


    #assert(words[i] == 'symwhere')
    conds = []
    while(words[i].lower() != 'symend'):
        i = i + 1
        if(words[i].lower() == 'symcol'):
            cond_sel_col_words = []
            i = i + 1
            while(words[i].lower() != 'symop'):
                cond_sel_col_words.append(words[i])
                i = i + 1
            cond_sel_col = ' '.join(cond_sel_col_words)
            
            assert(words[i].lower() == 'symop')
            i = i + 1
            op = words[i]
            i = i + 1
            assert(words[i].lower() == 'symcond')
            i = i + 1
            cond_words = []
            while((words[i].lower() != 'symand') & (words[i].lower() != 'symend')):
                cond_words.append(words[i])
                i = i + 1
            cond_sent = ' '.join(cond_words)
            conds.append((cond_sel_col, op, cond_sent))
    '''
    print("SELECT column") 
    print(sel_col)
    print("Conditions")
    print(conds)
    '''

    nl = ''
    if(agg != None):
        if(agg.lower() == 'count'):
            nl = 'how many '
        if(agg.lower() == 'max'):
            nl = 'what is the maximum '
        if(agg.lower() == 'min'):
            nl = 'what is the minimum '
        if(agg.lower() == 'sum'):
            nl = 'what is the total '
    else:
        nl = 'What is the ' 
        
    sel_col = ' '.join(sel_col_words)

    '''
    nl = nl + sel_col + ' where the'
    cond_strings = []
    for cond in conds:
        op_string = ''
        if(cond[1] == '='):
            op_string = ' is '
        if(cond[1] == '>'):
            op_string = ' is greater than '
        if(cond[1] == '<'):
            op_string = ' is less than '
        cond_string = cond[0] + op_string + cond[2]
        cond_strings.append(cond_string)
    full_cond_string = ' and '.join(cond_strings)
    nl = nl + ' ' + full_cond_string + ' ?'
    '''
    return (agg, sel_col, conds)

def sql_reconstruct(parse_tuple):
    agg, sel_col, conds = parse_tuple
    words = []
    words.append('symselect symagg')
    if agg != None:
        words.append(agg)
    words.append('symcol')
    words.append(sel_col)
    words.append('symwhere')
    for cond in conds:
        words.append('symcol')
        words.append(cond[0])
        words.append('symop')
        words.append(cond[1])
        words.append('symcond')
        words.append(cond[2])
        words.append('symand')
        
    words[-1] = 'symend'
    
    sql = ' '.join(words)
    sql = re.sub(' +', ' ', sql)
    return sql

def get_incorrect_column(col_name, columns):
    i = 0
    while True:
        incorrect_column = random.choice(columns)
        if incorrect_column == col_name:
            pass
        else:
            return incorrect_column

def get_incorrect_sqls(sql, columns):
    agg, sel_col, conds = sql_parse(sql)
    incorrect_sqls = []
    incorrect_sel_col = get_incorrect_column(sel_col, columns)
    incorrect_sqls.append(sql_reconstruct((agg, incorrect_sel_col, conds)))
    changed_conds = []
    for cond in conds:
        #incorrect_cond = cond[:]
        #incorrect_cond[0] = get_incorrect_column(incorrect_cond[0], columns)
        incorrect_col = get_incorrect_column(cond[0], columns)
        changed_conds.append((incorrect_col, cond[1], cond[2]))
    incorrect_conds_list = []
    for i in range(len(changed_conds)):
        cond = []
        for j in range(len(conds)):
            if j == i:
                cond.append(changed_conds[j])
            else:
                cond.append(conds[j])
        incorrect_conds_list.append(cond)

    for incorrect_conds in incorrect_conds_list:
        incorrect_sqls.append(sql_reconstruct((agg, sel_col, incorrect_conds)))
    return incorrect_sqls
                

    

if __name__ =='__main__':
    splits = ['data/train', 'data/test', 'data/dev']
    for split in splits:
        sql_file = open(split + '.in', 'r', encoding='utf-8')
        for row in sql_file:
            sql = row.rstrip()
            parse_tuple = sql_parse(sql)
            re_sql = sql_reconstruct(parse_tuple)
            if(sql != re_sql):
                print("Error!")
                print(sql)
                print(re_sql)

