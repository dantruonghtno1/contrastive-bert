python xla_spawn.py --num_cores 8 \
    --model_path "vinai/phobert-base" \
    --data_path "truongpdd/text_clf_fold_9" \
    --epochs 1 \
    --sampled_for_model_test True

rm -rf results_contrastive

python xla_spawn.py --num_cores 8 \
    --model_path "vinai/phobert-base" \
    --data_path "truongpdd/text_clf_fold_9" \
    --epochs 1 \
    --sampled_for_model_test True