import os
import sys
sys.path.append('/home/chenjingjing/Models/REDfold1')

from redfold import *
# 设置使用 GPU 1
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def main():

  [args,file1,file2]= get_args(default_config)

  if not os.path.exists(default_path):
    os.makedirs(default_path)

  if not os.path.exists(file2):
    set_config(redfold_config,file2)
  
  process_data(args, file1,file2)
  test_data(args,file2)

if __name__ == '__main__':
  main()
