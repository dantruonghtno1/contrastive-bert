for i in {0..9}
do
    python xla_spawn.py --num_cores 8 supTrainer.py \
        --model_path "vinai/phobert-base" \
        --data_path "truongpdd/text_clf_fold_$i" \
        --epochs 3 \

    rm -rf results_contrastive

    python xla_spawn.py --num_cores 8 finetune.py \
        --model_path "vinai/phobert-base" \
        --data_path "truongpdd/text_clf_fold_$i" \
        --epochs 3 \

    rm -rf results  
done