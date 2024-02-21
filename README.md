

# Partial Execution

Table Question-Answering involves both understanding the natural language query and grounding it in the context of the input table to extract the relevant information. 
In this context, many methods have highlighted the benefits of intermediate pre-training from SQL queries. 
However, while most approaches aim at 
generating final answers from inputs directly, 
we claim that there is better to do with  SQL queries during training.
By learning to imitate a restricted portion of SQL-like algebraic operations, we show that their execution flow provides intermediate supervision steps that allow increased generalization and structural reasoning compared with classical approaches of the field. 
Our study bridges the gap between semantic parsing and direct answering methods and provides useful insights regarding what types of operations should be predicted by a generative architecture or be preferably executed by an external algorithm.

# Installation

### Data

Instructions for installation

Installation of Squall data.
```bash
git clone https://github.com/tzshi/squall.git && \
mkdir -p ~/partial_execution/data && \
mv squall/data ~/partial_execution && \
mv squall/tables ~/partial_execution/data && \
rm -rf squall
```

Creating the logical form for the 'preorder' flattening method.
```bash
cd scripts
python create_logicalforms.py --flatten_mode preorder
```
Pas encore dispo
```bash
tar -xzvf models.tar.gz
```
### Models

```bash
tar -xzvf models_pcs.tar.gz
```

# Train Your Own Model

### finetuned_model.py
This script evaluates a pre-trained model on the Question + Table -> Omega_include data, on the WikiTableQuestions dataset.

For example, here we aim to evaluate the model using the intermediate training "tapex-base" with the "preorder" flattening method whyle executing only the Projection, Selection, and Comparison (PCS) nodes. This fine-tuning is designed to run on a GPU, with a total batch size of around 128.

```bash
python finetune_model.py \
  --do_train \
  --do_eval \
  --Omega_include "pcs"  \
  --dataset_name data/wtq_lf_preorder \
  --output_dir "models/tapex-base_pcs" \
  --config_name "microsoft/tapex-base" \
  --tokenizer_name "microsoft/tapex-base"\
  --model_name_or_path "microsoft/tapex-base"  \
  --overwrite_output_dir \
  --per_device_train_batch_size 6 \
  --gradient_accumulation_steps 21 \
  --per_device_eval_batch_size 6 \
  --learning_rate 3e-5 \
  --logging_steps 10 \
  --eval_steps 1000 \
  --save_steps 1000 \
  --warmup_steps 2000 \
  --evaluation_strategy steps \
  --predict_with_generate \
  --num_beams 5 \
  --weight_decay 1e-2 \
  --label_smoothing_factor 0.1 \
  --max_steps 20000
```




### perf_model.py
After fine-tuning, you can evaluate the model's performance on the test dataset with the following code. This process identifies the best checkpoint during validation and then applies this checkpoint to the test set for a comprehensive performance assessment.

```bash
python perf_model.py \
 --path_model_folder "tapex-base_pcs" \
 --flatten_mode "preorder"
```


# Test the Best Model
Pas dispo encore
```bash
python perf_model.py \
 --path_model_folder "tapex-large_pcs" \
 --flatten_mode "preorder"
```

# Teste permuted performances


Creating the datasets: non-permuted and permuted.
This will create two datasets (HF) each of size 506. One will consist of 316 permuted tables, and the other will be the original from the official validation set.



```bash
cd scripts
python create_permuted_data.py 
```

Example of using the baseline file for Tapex with permuted data.



```bash
python analysis/baselines.py --model tapex --permuted 1
```

# Test ensemble models
Pas dispo encore
