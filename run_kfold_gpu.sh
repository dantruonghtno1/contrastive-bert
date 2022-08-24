for i in {0..9}
do
    python supTrainer.py \
        --model_path "vinai/phobert-base" \
        --data_path "truongpdd/text_clf_fold_$i" \
        --epochs 3 \

    rm -rf results_contrastive

    python finetune.py \
        --model_path "vinai/phobert-base" \
        --data_path "truongpdd/text_clf_fold_$i" \
        --epochs 3 \

    rm -rf results  
done