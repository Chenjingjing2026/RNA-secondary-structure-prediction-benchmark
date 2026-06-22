dataset="telomerase"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
nohup python /home/cjj/Spot-RNA/SPOT-RNA-DL/main.py \
  --datadir /home/cjj/Spot-RNA/SPOT-RNA-DL/data/Crossfamily_1024/${dataset} \
  --sessName result/Crossfamily_1024/${dataset} \
  > log/train_${dataset}_1024.log 2>&1 &