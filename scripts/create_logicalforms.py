

from datasets import load_dataset, DatasetDict
import json
from argparse import ArgumentParser

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


import os
import sys
from tqdm import tqdm

# Get the absolute path to the directory containing the script
dir_path = os.path.dirname(os.path.abspath(__file__))
# Move up one level to get the project root
project_root = os.path.dirname(dir_path)
# Add the project root to the sys.path
sys.path.insert(0, project_root)

from data_processing.metrics import target_values_map, flexible_denotation_accuracy, to_value_list
from data_processing.nodes import create_nodes
from data_processing.seq_to_graph import parse





def main(args):

    wtq = load_dataset("wikitablequestions")

    file_path = "../data/squall.json"
    logging.info("Attempting to open the file: %s", file_path)
    with open(file_path, 'r') as json_file:
        squall = json.load(json_file)

    squall_lf = []
    Omega_includes = [
                    ["P"],
                    ["P","C"],
                    ["P","C","S"],
                    ["P","C","S","GB","H"],
                    ["P","C","S","GB","H","OB"],
                    ["P","C","S","GB","H","OB","A"],
                    ["P","C","S","GB","H","OB","A","OP"],
                    ["P","C","S","GB","H","OB","A","OP","L"]]


    traversal = args.flatten_mode
    print("\n"*3)
    logging.info(f'Processing example {traversal}')
    print("\n"*3)
    example = squall[0]
    sql = " ".join([i[1] for i in example["sql"]])
    result_squall = target_values_map[example['nt']]
    logging.info("Target: %s", result_squall)
    counter=0
    # check if every flat sequence give the gold answer
    for Omega_include in Omega_includes:

        logging.info("\nProcessing Omega_include: %s", Omega_include)
        counter+=1
        G0 = create_nodes(sql, example["tbl"])
        header = G0.header
        result0 = to_value_list(G0.executed_last_node())
        flatten_sequence = G0.linearize_graph(Omega_include, flatten_mode=traversal)
        G1 = parse(flatten_sequence, header, flatten_mode=traversal)


        logging.info("Flatten sequence: %s", flatten_sequence)
        result1 = to_value_list(G1.executed_last_node())
        name = "lf_"+"".join(Omega_include).lower()
        example[name] = flatten_sequence
        if not flexible_denotation_accuracy(result_squall, result1):
            break


            
    logging.info(f"Starting processing {traversal}")


    count_errors = 0
    for idx in tqdm(range(0, len(squall))):
        try:

            if idx % 100 == 0:
                logging.info(f'Processing {idx}/{len(squall)} - Completed {len(squall_lf)} - Erros {count_errors}')
        
            example = squall[idx]
            sql = " ".join([i[1] for i in example["sql"]])
            result_squall = target_values_map[example['nt']]
            counter=0
            # check if every flat sequence give the gold answer
            for Omega_include in Omega_includes:
                counter+=1


                G0 = create_nodes(sql, example["tbl"])
                header = G0.header
                result0 = to_value_list(G0.executed_last_node())
                
                flatten_sequence = G0.linearize_graph(Omega_include, flatten_mode=traversal)
                G1 = parse(flatten_sequence, header, flatten_mode=traversal)
    

                result1 = to_value_list(G1.executed_last_node())
                name = "lf_"+"".join(Omega_include).lower()
                example[name] = flatten_sequence
                if not flexible_denotation_accuracy(result_squall, result1):
                    break

            if counter==len(Omega_includes):
                example['sql']=sql
                squall_lf.append(example)
            else:
                count_errors += 1
        except Exception as e: 
            #logging.info(e)
            count_errors += 1
            pass
        
    logging.info(f'Processing {idx}/{len(squall)} - Completed {len(squall_lf)} - Erros {count_errors}')
    logging.info(f"Successfully processed {len(squall_lf)} records from {file_path} for {traversal}")


    # Filter wikitablequestions observations.
    train_ids = [example["nt"] for example in squall_lf]
    wtq_train_indices = [i for i, example in enumerate(wtq["train"]) if example['id'] in set(train_ids)]
    wtq_val_indices = [i for i, example in enumerate(wtq["validation"]) if example['id'] in set(train_ids)]
    train_dataset = wtq["train"].select(wtq_train_indices)
    validation_dataset = wtq["validation"].select(wtq_val_indices)
    datasets = DatasetDict({ 'train': train_dataset, 'validation': validation_dataset, "test": wtq['test']})


    # Add logical form 
    logical_forms = ["lf_"+"".join(Omega_include).lower() for Omega_include in Omega_includes] + ["sql"]
    squall_targets = {example["nt"]:[example[lf] for lf in logical_forms] for example in squall_lf}

    def add_target_squall(example):
        idx = example["id"]
        targets = squall_targets[idx]
        
        return {lf : targets[i] for i, lf in enumerate(logical_forms)}

    datasets["train"] = datasets["train"].map(add_target_squall)
    datasets["validation"] = datasets["validation"].map(add_target_squall)

    datasets.save_to_disk(f"../data/wtq_lf_{traversal}")



if __name__ == "__main__":

    parser = ArgumentParser()

    
    parser.add_argument('--flatten_mode', help='the maximum length for the flattened table plus input SQL query',
                type=str,
                default="preorder")

    args = parser.parse_args()


    main(args)