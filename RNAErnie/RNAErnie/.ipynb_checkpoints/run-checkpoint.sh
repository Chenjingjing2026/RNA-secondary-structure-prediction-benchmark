python run_ssp.py \
    --task_name=Filtered \
    --dataset_dir=/root/lanyun-tmp/RNAErnie/mydata/Filtered \
    --model_name_or_path="/root/lanyun-tmp/RNAErnie/output/BERT,ERNIE,MOTIF,PROMPT/checkpoint_final" \
    --train=True \
    --num_train_epochs=50 \
    --lr=0.001 \
    --output=/root/lanyun-tmp/RNAErnie/output_ft/output_ft/Filtered