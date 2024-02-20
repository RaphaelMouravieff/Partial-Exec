import os
import json 
import sys
import torch

print(os.getcwd())
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_from_disk, concatenate_datasets
from functools import partial
import math
from data_processing.metrics import (to_value_list,
                                     to_value,
                                     plot_and_save_model_performance,
                                     fuzzy_matching,
                                     strict_denotation_accuracy,
                                     flexible_denotation_accuracy)

from data_processing.seq_to_graph import parse

import random
    
from transformers import (
    AutoConfig,
    BartForConditionalGeneration,
    TapexTokenizer,
    AutoModelForSeq2SeqLM
)








def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    path_model_folder = f"models/{args.flatten_mode}/fine-tuned/{args.path_model_folder}"
        

    
    output_file = f'{path_model_folder}/model_evaluation_results.json'
    ckpt_fig_file = f'{path_model_folder}/checkpoints_perfs.png'
    
    print(f"path_model_folder : {path_model_folder}")

    checkpoints = [os.path.join(path_model_folder, i) for i in os.listdir(path_model_folder) if i.startswith("checkpoint")]
    Omega_include = path_model_folder.split('/')[-1].split('_')[1]
    target_col_name = f"lf_{Omega_include.lower()}"


    datasets = load_from_disk(f"data/wtq_lf_{args.flatten_mode}")
    checkpoints = list(set(checkpoints) - set([ckpt_fig_file]))

    print("all ckpt",checkpoints)
        
    if os.path.exists(output_file):
        print(f'output file {output_file} already exist')
        with open(output_file, 'r') as file:
            results = json.load(file)
        results = {k: v for k, v in results.items() if not k.startswith("test")}
        remove_checkpoints = [i for i in list(results.keys()) if i.split('/')[-1].startswith("checkpoint") ]
        
        re_eval_checkpoints = [key for key, value in results.items() if "Fuzzy_Match" in value and value["Fuzzy_Match"] == 0]
        
        print(f're_eval_checkpoints : {re_eval_checkpoints}')
        remove_checkpoints = list(set(remove_checkpoints) - set(re_eval_checkpoints))

        print(f'all checkpoints : {checkpoints}')
        print(f"remove_checkpoints : {remove_checkpoints}")
        checkpoints = list(set(checkpoints) - set(remove_checkpoints))
        print(f'checkpoints to evaluate : {checkpoints}')
    
    if len(checkpoints) == 0:
        print("No checkpoints to evaluate. ")

    if not os.path.exists(output_file):
        print(f'output file {output_file} not exist start creating')
        
        
    path_tokenizer_config = "microsoft/tapex-large"
    tokenizer = TapexTokenizer.from_pretrained(path_tokenizer_config)
    config = AutoConfig.from_pretrained(path_tokenizer_config)
    config.no_repeat_ngram_size = 0
    config.max_length = 1024
    config.early_stopping = False

    features = ["question", "table"]
    removed_cols = set(datasets["validation"].features.keys())-set([target_col_name]+features+["answers"])
    datasets["validation"] = datasets["validation"].remove_columns(removed_cols)
    datasets["validation"] = datasets["validation"].rename_column(target_col_name, 'answers_lf')
    datasets["validation"] = datasets["validation"].rename_column("answers", 'answers_exec')
    datasets["test"] = datasets["test"].rename_column("answers", 'answers_exec')

    rnd = random.randint(0, len(datasets["validation"]))
    print('random validation id', rnd)
    print("*"*10)
    print(f'target ({target_col_name}) answers_lf: ')
    print(datasets["validation"][rnd]["answers_lf"])
    print('table :')
    print(datasets["validation"][rnd]["table"]["header"])
    for r in range(len(datasets["validation"][rnd]["table"]["rows"])):
        print(datasets["validation"][rnd]["table"]["rows"][r])
    print('answer exec')
    print(datasets["validation"][rnd]["answers_exec"])
    print("*"*10)
    
    rnd = random.randint(0, len(datasets["test"]))
    print('random validation id', rnd)
    print("*"*10)
    print('table :')
    print(datasets["test"][rnd]["table"]["header"])
    for r in range(len(datasets["test"][rnd]["table"]["rows"])):
        print(datasets["test"][rnd]["table"]["rows"][r])
    print('answer exec')
    print(datasets["test"][rnd]["answers_exec"])
    print("*"*10)
    
    


    def preprocess_tableqa_function(examples, is_test=False):

        questions = [question.lower() for question in examples["question"]]
        example_tables = examples["table"]
        tables = [
            pd.DataFrame.from_records(example_table["rows"], columns=example_table["header"])
            for example_table in example_tables
        ]
        
        model_inputs = tokenizer(
            table=tables, query=questions, max_length=1024, padding=False, truncation=True
        )
        model_inputs["answers_exec"] = examples["answers_exec"] 
        if not is_test:
            model_inputs["answers_lf"] = examples["answers_lf"]
            
        return model_inputs

    


    preprocess_tableqa_function_valid = partial(preprocess_tableqa_function, is_test=False)
    eval_dataset = datasets["validation"]
    eval_dataset = eval_dataset.map(
                    preprocess_tableqa_function_valid,
                    batched=True)

    preprocess_tableqa_function_test = partial(preprocess_tableqa_function, is_test=True)
    predict_dataset = datasets["test"]
    predict_dataset = predict_dataset.map(
        preprocess_tableqa_function_test,
        batched=True,
    )

    
    
    def evaluate_model(model, dataset, is_test=False):
        model.eval()
        perfs={"Denotation_Accuracys":[], "Fuzzy_Matchs":[],
                "Strict_Denotation_Accuracy_Execs":[], "Flexible_Denotation_Accuracy_Execs":[]}
        for idx, example in tqdm(enumerate(dataset), total=len(dataset)):
            print(f"idx {idx}")
            input_ids = example["input_ids"]
            attention_mask = example["attention_mask"]


            answers_exec = example["answers_exec"]

            input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
            attention_mask = torch.tensor(attention_mask).unsqueeze(0).to(device)
            generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask)
            prediction_lf = tokenizer.decode(generated_ids.squeeze(), skip_special_tokens=True).strip()

            header=tokenizer.decode(input_ids.squeeze(), skip_special_tokens=True)
            header=header.split(' col : ')[1].split(' row 1 : ')[0].split(' |')
            header=[h.strip() for h in header]

            if not is_test:
                answers_lf = example["answers_lf"]
                acc_lf = answers_lf ==  prediction_lf
                print(f'answers_lf {answers_lf} prediction_lf {prediction_lf}')
            if is_test:
                acc_lf = True

            #G=parse(prediction_lf, header, flatten_mode=args.flatten_mode)
            try:
                print('1')
                G=parse(prediction_lf, header, flatten_mode=args.flatten_mode)
                print('2')
                prediction_exec = G.executed_last_node()
                print(f'answers_exec {answers_exec} prediction_exec {prediction_exec}')
                fuzzy_match = fuzzy_matching(", ".join(answers_exec).lower(), ", ".join(prediction_exec).lower())
                acc_ex_stric = strict_denotation_accuracy(to_value_list(answers_exec), to_value_list(prediction_exec))
                acc_ex_flex = flexible_denotation_accuracy(answers_exec, prediction_exec)

            except Exception as e:
                print(e)
                fuzzy_match = 0
                acc_ex_stric = False
                acc_ex_flex = False



            print(f'Denotation_Accuracys {acc_lf} Strict_Denotation_Accuracy_Execs {acc_ex_stric} Flexible_Denotation_Accuracy_Execs {acc_ex_flex} Fuzzy_Matchs {fuzzy_match}')
            perfs["Denotation_Accuracys"].append(acc_lf)
            perfs["Fuzzy_Matchs"].append(fuzzy_match)
            perfs["Strict_Denotation_Accuracy_Execs"].append(acc_ex_stric)
            perfs["Flexible_Denotation_Accuracy_Execs"].append(acc_ex_flex)

        Denotation_Accuracy = sum(perfs["Denotation_Accuracys"])/len(perfs["Denotation_Accuracys"])
        Fuzzy_Match  = sum(perfs["Fuzzy_Matchs"])/len(perfs["Fuzzy_Matchs"])
        Strict_Denotation_Accuracy_Exec = sum(perfs["Strict_Denotation_Accuracy_Execs"])/len(perfs["Strict_Denotation_Accuracy_Execs"])
        Flexible_Denotation_Accuracy_Exec = sum(perfs["Flexible_Denotation_Accuracy_Execs"])/len(perfs["Flexible_Denotation_Accuracy_Execs"])

        return {'Denotation_Accuracy': Denotation_Accuracy,
                'Fuzzy_Match': Fuzzy_Match,
                    'Strict_Denotation_Accuracy_Exec': Strict_Denotation_Accuracy_Exec,
                    'Flexible_Denotation_Accuracy_Exec':Flexible_Denotation_Accuracy_Exec}  # example


    if not os.path.exists(output_file):
        
        results = {}

        
    for checkpoint in checkpoints:
        print(f'Validation performance ckpt = {checkpoint}')
        # Load the model from checkpoint
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(device)

        # Evaluate the model
        metrics = evaluate_model(model, eval_dataset)
        print("Validation result  = ",metrics)
        # Store the results
        results[checkpoint] = metrics

        del model
        torch.cuda.empty_cache()
        print(f'maj results : {output_file}')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
            
    
            
    print(f'Save plot perfs at : {ckpt_fig_file}') 
    plot_and_save_model_performance(results, ckpt_fig_file)
    
    results_print = {key.split('/')[-1]: value for key, value in results.items()}
    for key, r in results_print.items():

        print(f"Key: {key}, Result: {r}")
    
    
    best_checkpoint = max(results, key=lambda k: results[k]['Flexible_Denotation_Accuracy_Exec'])
    print(f"Best model checkpoint: {best_checkpoint}")
    best_checkpoint_label = best_checkpoint.split('/')[-1]
    best_model = BartForConditionalGeneration.from_pretrained(best_checkpoint)
    best_model.to(device)

    
    # Evaluate the best model on test data
    test_metrics = evaluate_model(best_model, predict_dataset, is_test=True)
    print(f"Test metrics for the best model: {test_metrics}")

    results[f"test_{best_checkpoint_label}"] = test_metrics


    # Save results to a JSON file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {output_file}")
    
    
    
    

    

if __name__ == "__main__":


    parser = ArgumentParser()
        # TODO: if you want to provide your SQL templates, you could organize your file with the format of SQUALL data
        #  and you should also prepare the corresponding database files / csv files for tables.
    parser.add_argument('--path_model_folder', type=str)
    parser.add_argument('--flatten_mode', type=str)


    args = parser.parse_args()

    main(args)