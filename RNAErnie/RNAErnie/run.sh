export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
nohup python run_ssp.py \
    --task_name=crossfamily \
    --dataset_dir=/home/cjj/RNAErnie/mydata/Crossfamily/tmRNA \
    --model_name_or_path="/home/cjj/RNAErnie/output/BERT,ERNIE,MOTIF,PROMPT/checkpoint_final" \
    --train=True \
    --num_train_epochs=50 \
    --lr=0.001 \
    --output=/home/cjj/RNAErnie/output_ft/output_ft/Crossfamily/tmRNA/ > /home/cjj/RNAErnie/output_ft/output_ft/Crossfamily/tmRNA/training.log 2>&1 &
