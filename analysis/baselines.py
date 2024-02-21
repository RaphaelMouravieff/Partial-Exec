
from argparse import ArgumentParser
import json
import os
import sys
from transformers import pipeline
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



# Get the absolute path to the directory containing the script
dir_path = os.path.dirname(os.path.abspath(__file__))
# Move up one level to get the project root
project_root = os.path.dirname(dir_path)
# Add the project root to the sys.path
sys.path.insert(0, project_root)


from data_processing.metrics import *
from data_processing.utils import to_pandas, init_counts, convert_to_numeric, models_d
from data_processing.seq_to_graph import parse
from data_processing.nodes import create_nodes

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import TapexTokenizer, AutoConfig, BartForConditionalGeneration

from datasets import load_from_disk

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def update_counts(operation_counts, sql, acc_ex_flex):
    try:
        # Create nodes once and reuse
        G = create_nodes(sql, None) 
        elements_list = [n.prefix for _, n in G.nodes.items()]
        
        # Define prefixes to check in elements_list
        prefixes = ["P", "C", "S", "GB", "H", "OB", "A", "OP", "L"]
        
        # Update totals for all prefixes found
        for prefix in prefixes:
            if prefix in elements_list:
                total_key = f"{prefix}_total"
                operation_counts[total_key] = operation_counts.get(total_key, 0) + 1
                
                # If not acc_ex_flex, also update the non-total counts
                if not acc_ex_flex:
                    operation_counts[prefix] = operation_counts.get(prefix, 0) + 1

    except:
        print('no update')
        
    return operation_counts



def tapas_executor(pred):
    try:
        if " > " in pred:
            aggr, pred = pred.split(' > ')
            pred = pred.split(', ')
            if aggr == "NONE":
                prediction_exec = pred
            if aggr == "COUNT":
                prediction_exec = [str(len(pred))]
            if aggr in ["SUM","AVERAGE"]:
                if len(pred)==1:
                    prediction_exec = pred
                else:
                    pred, check = convert_to_numeric(pred)
                    if check:
                        if aggr == "SUM":
                            prediction_exec = [str(sum(pred))]
                        if aggr == "AVERAGE":
                            prediction_exec = [str(sum(pred)/len(pred))]
                    else :
                        prediction_exec = pred
        else:
            prediction_exec = pred.split(', ')
    except:
        prediction_exec = str(pred).split(', ')
    return prediction_exec





def main(args):


    model_hf = models_d[args.model]
 
    logging.info(f'model_hf {model_hf}')
    file_path = "data/squall.json"
    with open(file_path, 'r') as json_file:
        squall = json.load(json_file)
    squall_targets = {example["nt"]:" ".join([i[1] for i in example["sql"]])  for example in squall}

    if args.model in ["tapex","omnitab"]:
        tokenizer = AutoTokenizer.from_pretrained(model_hf)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_hf).to(device)
    
    elif args.model == "tapas":
        tqa = pipeline(task="table-question-answering", model=model_hf)

    else :
        tokenizer = TapexTokenizer.from_pretrained("microsoft/tapex-large")
        config = AutoConfig.from_pretrained("microsoft/tapex-large")
        config.no_repeat_ngram_size = 0
        config.max_length = 1024
        config.early_stopping = False
        model = BartForConditionalGeneration.from_pretrained(model_hf, config=config)
        model.to(device)

    perfs={"Fuzzy_Matchs":[],
            "Strict_Denotation_Accuracy_Execs":[],
            "Flexible_Denotation_Accuracy_Execs":[]}

    if args.permuted:
        name = "permuted"
        logging.info(f"load : data/permuted_data")
        eval_dataset = load_from_disk(f"data/permuted_data")

    if not args.permuted:
        name = "not_permuted"
        logging.info(f"load : data/nopermuted_data")
        eval_dataset = load_from_disk(f"data/nopermuted_data")

    logging.info(eval_dataset)

    _, operation_counts = init_counts()

    for idx, item in enumerate(eval_dataset):
            
        logging.info("")
        logging.info(f'idx {idx}')

        sql = squall_targets[item['id']]
        question = item['question']
        table  = to_pandas(item)
        answers_exec = item["answers"]

        
   
        
        if args.model in ["tapex","omnitab"]:
            
            encoding = tokenizer(table=table, query=question,
                                return_tensors="pt", max_length=1024,
                                padding=False, truncation=True).to(device)
            
            outputs =  model.generate(**encoding, max_length=512)
            prediction_exec = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            prediction_exec = [i.strip().lower() for i in prediction_exec]
            answers_exec = item["answers"]
            fuzzy_match = fuzzy_matching(", ".join(answers_exec).lower(), ", ".join(prediction_exec).lower())
            acc_ex_stric = strict_denotation_accuracy(to_value_list(answers_exec), to_value_list(prediction_exec))
            acc_ex_flex = flexible_denotation_accuracy(answers_exec, prediction_exec)

        elif args.model == "tapas":

            try:
                pred = tqa(table=table, query=question, max_length=512, truncation=True)['answer']
            except:
                pred = "None"
            prediction_exec = tapas_executor(pred)
            fuzzy_match = fuzzy_matching(", ".join(answers_exec).lower(), ", ".join(prediction_exec).lower())
            acc_ex_stric = strict_denotation_accuracy(to_value_list(answers_exec), to_value_list(prediction_exec))
            acc_ex_flex = flexible_denotation_accuracy(answers_exec, prediction_exec)

        else:

            encoding = tokenizer(table=table, query=question,
                                return_tensors="pt", max_length=1024,
                                padding=False, truncation=True).to(device)
            
            generated_ids =  model.generate(**encoding)
            prediction_lf = tokenizer.decode(generated_ids.squeeze(), skip_special_tokens=True).strip()

            header=tokenizer.decode(encoding["input_ids"].squeeze(), skip_special_tokens=True)
            header=header.split(' col : ')[1].split(' row 1 : ')[0].split(' |')
            header=[h.strip() for h in header]

            try:

                G=parse(prediction_lf, header, flatten_mode="preorder")
                prediction_exec = G.executed_last_node()
                fuzzy_match = fuzzy_matching(", ".join(answers_exec).lower(), ", ".join(prediction_exec).lower())
                acc_ex_stric = strict_denotation_accuracy(to_value_list(answers_exec), to_value_list(prediction_exec))
                acc_ex_flex = flexible_denotation_accuracy(answers_exec, prediction_exec)

            except Exception as e:
                logging.info(f"Exception {e}")
                fuzzy_match = 0
                acc_ex_stric = False
                acc_ex_flex = False

        logging.info(f"{fuzzy_match}, {acc_ex_stric}, {acc_ex_flex}")

        perfs["Fuzzy_Matchs"].append(fuzzy_match)
        perfs["Strict_Denotation_Accuracy_Execs"].append(acc_ex_stric)
        perfs["Flexible_Denotation_Accuracy_Execs"].append(acc_ex_flex)
    
        operation_counts = update_counts(operation_counts, sql, acc_ex_flex)
        logging.info(f"operation_counts : {operation_counts}")

    Fuzzy_Match  = sum(perfs["Fuzzy_Matchs"])/len(perfs["Fuzzy_Matchs"])
    Strict_Denotation_Accuracy_Exec = sum(perfs["Strict_Denotation_Accuracy_Execs"])/len(perfs["Strict_Denotation_Accuracy_Execs"])
    Flexible_Denotation_Accuracy_Exec = sum(perfs["Flexible_Denotation_Accuracy_Execs"])/len(perfs["Flexible_Denotation_Accuracy_Execs"])
    logging.info(f"Fuzzy_Match={Fuzzy_Match}, Strict_Denotation_Accuracy_Exec={Strict_Denotation_Accuracy_Exec}, Flexible_Denotation_Accuracy_Exec={Flexible_Denotation_Accuracy_Exec})")


    # Write the dictionary to a file in JSON format
    path = f'analysis/counts/{name}_{args.model}.json'
    logging.info(f"error decomposition for {args.model} at {path}")
    with open(path, 'w') as json_file:
        json.dump(operation_counts, json_file)


    def print_ratios(data):
        # Iterate over each key in the dictionary
        for key in data:
            # Check if the current key has a corresponding "_total" key
            if key + "_total" in data:
                # Calculate the ratio
                ratio = (1-(data[key] / data[key + "_total"]))*100
                # Print the ratio in a formatted way
                logging.info(f"{key}/{key}_total: {ratio:.2f}")


    print_ratios(operation_counts)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--model', help='the maximum length for the flattened table plus input SQL query',
                        type=str,
                        default="tapex")

    

    parser.add_argument('--permuted', help='the maximum length for the flattened table plus input SQL query',
            type=bool,
            default=True)

    args = parser.parse_args()


    main(args)