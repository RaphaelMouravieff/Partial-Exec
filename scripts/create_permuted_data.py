import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import os
import sys
import sqlite3
import json

import pandas as pd
import numpy as np
import copy


from transformers import TapexTokenizer
from datasets import load_dataset, Dataset



# Get the absolute path to the directory containing the script
dir_path = os.path.dirname(os.path.abspath(__file__))
# Move up one level to get the project root
project_root = os.path.dirname(dir_path)
# Add the project root to the sys.path
sys.path.insert(0, project_root)


from data_processing.metrics import  flexible_denotation_accuracy
from data_processing.nodes import create_nodes
from data_processing.utils import to_pandas, common_dataset


np.random.seed(42)


def get_squall_table(tbl):
    db_path = f"/home/raphael.gervillie/sql_graph/data/tables/db/{tbl}.db"
    conn = sqlite3.connect(db_path)
    squall_tbl = pd.read_sql_query(f"SELECT * FROM w", conn)
    return squall_tbl

def permuted(squall_tbl, table, sql, tbl, Npermuted_rows):
    G = create_nodes(sql, tbl)
    cols = list(set([node.parameter for name, node in G.nodes.items() if node.prefix=="P"])-set(["id"]))
    header = table.columns.tolist()
    mapping = mapping_cols(cols, header)
    
    for col in cols:
        permuted_indices = np.random.permutation(Npermuted_rows)
        permuted_indices = np.random.permutation(Npermuted_rows)
        squall_tbl_col_permuted = squall_tbl[col].iloc[:Npermuted_rows].iloc[permuted_indices].tolist() + squall_tbl[col].iloc[Npermuted_rows:].tolist()
        squall_tbl[col] = squall_tbl_col_permuted
        
        table_col_permuted = table[mapping[col]].iloc[:Npermuted_rows].iloc[permuted_indices].tolist() + table[mapping[col]].iloc[Npermuted_rows:].tolist()
        table[mapping[col]] = table_col_permuted
    
    return squall_tbl, table


def execute(table, sql):
    conn = sqlite3.connect(':memory:')
    table.to_sql('w', conn, index=False, if_exists='replace')
    result = pd.read_sql_query(sql, conn)
    conn.close()
    result = result.values.flatten().tolist()
    result = [str(i) for i in result]
    return result 

def mapping_cols(cols, header):
    mapping = {f"c{idx+1}":i for idx, i in enumerate(header)}
    new_mapping = {}
    for col in cols:
        base_header = col.split('_')[0]
        if base_header in mapping:
            new_mapping_key = col if '_' in col else base_header
            new_mapping[new_mapping_key] = mapping[base_header]
            
    #cols_update = list(new_mapping.keys())
    return new_mapping


def dataframe_to_dict(df):
    return {
        "header": df.columns.tolist(),
        "rows": df.values.tolist()
    }



wtq = load_dataset("wikitablequestions")


file_path = "../data/squall.json"
with open(file_path, 'r') as json_file:
    squall = json.load(json_file)
squall_targets = {example["nt"]:" ".join([i[1] for i in example["sql"]])  for example in squall}
squall_tables = {example["nt"]: example["tbl"] for example in squall}

eval_dataset = common_dataset()


tokenizer_filter = TapexTokenizer.from_pretrained("microsoft/tapex-large")



counter_except=0
counter_new=0

data_permuted=[]
data_nopermuted=[]

for idx, item in enumerate(eval_dataset):
    new_item =  copy.deepcopy(item)
    previous_table = copy.deepcopy(item["table"])
    previous_answer = copy.deepcopy(item["answers"])


    sql = squall_targets[item['id']]
    question = item['question']
    table  = to_pandas(item)
    answers_exec = item["answers"]
    tbl = squall_tables[item['id']]

    shapes = tokenizer_filter(table=table, query=question,
                                return_tensors="pt",
                                padding=False,
                                truncation=False)["input_ids"].shape[1]

    Npermuted_rows = table.shape[0]
    while shapes>1024:
        table = table.iloc[:-1]
        Npermuted_rows -= 1
        shapes = tokenizer_filter(table=table,
                                  query=question,
                                  return_tensors="pt",
                                  padding=False,
                                  truncation=False)["input_ids"].shape[1]
        
        logging.info(f'shape = {shapes}, Npermuted_rows = {Npermuted_rows}')

    logging.info(f'LAST  : shape = {shapes}, Npermuted_rows = {Npermuted_rows}')

    table  = to_pandas(item)

    
    try :
        table_squall = get_squall_table(tbl)
        table_squall, table = permuted(table_squall, table, sql, tbl, Npermuted_rows)
        answers_exec_squall = execute(table_squall, sql)
        check = flexible_denotation_accuracy(answers_exec_squall, answers_exec)
    except Exception as e:
        logging.info(f'Exception : {e}')
        counter_except+=1
        check=False

    
        
    nopermuted = {"table":previous_table, "answers":previous_answer,
                    "question":question, "id":item['id']}
    data_nopermuted.append(nopermuted)

    if not check:
        counter_new+=1
        logging.info('New permuted table ')
        table = table.applymap(str)
        new_item['table'] = dataframe_to_dict(table)
        new_item['answers'] = answers_exec_squall
        data_permuted.append(new_item)

    else:
        data_permuted.append(nopermuted)

    logging.info('Table not permuted')
    logging.info(to_pandas(data_nopermuted[idx]))

    logging.info('Table permuted')
    logging.info(to_pandas(data_permuted[idx]))
    logging.info(f"data_permuted['answers'] : {data_permuted[idx]['answers']}, data_nopermuted['answers'] : {data_nopermuted[idx]['answers']}")

def list_of_dicts_to_dict(list_of_dicts):
    # Initialize a dictionary to hold column-wise data
    columnar_data = {}
    # Ensure there is at least one dictionary in the list to proceed
    if list_of_dicts:
        # Initialize the dictionary with lists for each key in the first row
        for key in list_of_dicts[0].keys():
            columnar_data[key] = []
        # Populate the lists with data from each row
        for row in list_of_dicts:
            for key, value in row.items():
                columnar_data[key].append(value)
    return columnar_data

# Convert the list of dictionaries to a dictionary of lists
data_permuted_dict = list_of_dicts_to_dict(data_permuted)
data_nopermuted_dict = list_of_dicts_to_dict(data_nopermuted)

# Now, you can create Dataset objects without encountering the error



dataset_permuted = Dataset.from_dict(data_permuted_dict)
dataset_nopermuted = Dataset.from_dict(data_nopermuted_dict)


logging.info(f"total valid data = {len(eval_dataset)}, total new valid = {len(dataset_nopermuted)}, total permuted valid = {len(dataset_permuted)}, total permuted = {counter_new}")


dataset_permuted.save_to_disk(f"../data/permuted_data")
dataset_nopermuted.save_to_disk(f"../data/nopermuted_data")
