SPLIT=(0 1 2 3 4)
GPU=0
gene_list_filename=selected_gene_list.txt
huggingface_token= # please provide your huggingface_token
for sp in "${SPLIT[@]}"; do
    python ./main.py \
        --dataset 'kidney' \
        --gpu $GPU \
        --model_name hyperst \
        --gene_list_filename $gene_list_filename\
        --split_dir './split/kidney/flod_5/sample_split_flod_'$sp'.json' \
        --huggingface_token $huggingface_token\
        --experiment_name 'bs128_lr1e4_HMHVG'
done


