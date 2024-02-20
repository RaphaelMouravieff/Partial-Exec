

# Project Title

One Paragraph of project description goes here

## Installation
Instructions for installation


```bash
git clone https://github.com/tzshi/squall.git && \
mkdir -p ~/partial_execution/data && \
mv squall/data ~/partial_execution && \
mv squall/tables ~/partial_execution/data && \
rm -rf squall
```

```bash
cd scripts
python create_logicalforms.py --flatten_mode preorder
```

```bash
tar -xzvf models.tar.gz
```

# Entrainer votre propre modèle

# finetuned_model.py
permet d'évaluer un modèle pré-entrainé sur les données Question + table -> Omega_include et ainsi tester les performances sur wikitablequestions

Par exempl ici on veut évaluer le modèle avec entrainement intermédiaire tapex-base avec la méthode d'applatissement preorder et en exécutant seulement les noeuds Projections, Selection et Comparison (pcs). Ce fine-tuned est fait pour tourner sur un GPU, le batch size total doit être vers les 128. 

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




# perf_model.py
Permet de valider les données en test après avoir sélectionné le meilleur checkpoint en validation.


```bash
python perf_model.py \
 --path_model_folder "tapex-base_pcs" \
 --flatten_mode "preorder"
```


# Tester le meilleur modèle : 

```bash
python perf_model.py \
 --path_model_folder "tapex-large_pcs" \
 --flatten_mode "preorder"
```


# Tester les modèles ensembles : 
