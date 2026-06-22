# REDfold
The users are welcome to use REDfold webserver available at https://redfold.ee.ncyu.edu.tw for RNA structure prediction. REDfold is implemented in Python code and cross-platform compatible.

#System Requirement
	python (>=3.7)
	biopython (>=1.79)
	numpy (>=1.21)
	pandas(>=1.3.5)
	scipy (>=1.7.3)
	torch (>=1.9+cu111)


#Install
  %pip install redfold-1.14a0-py2.py3-none-any.whl


#Predict RNA Structure
REDfold test for predicting RNA secondary structure with fasta-formatted RNA sequences.

  %python redfold.py directory_containing_fasta_files

#Train parameters
REDfold can train the parameters with BPSEQ-formatted RNA sequences.

  %python redfold.py -train directory_containing_bpseq_files

#Configure parameters
Configure parameters in json format.
  BATCH_SIZE: test batch size
	batch_size_stage_1: dataload batch size
	load_model: read training model
  test_data: test data file
	data_type: test data directory
  model_type: test model file
	epoches_first: training iteration
	set_gamma: optimization hyperparm gamma
	set_L1: optimization constraint L1


# Web Server
REDfold web server is available at https://redfold.ee.ncyu.edu.tw

#References
Chun-Chi Chen, Yi-Ming Chan, "REDfold: Accurate RNA Secondary Structure Prediction using Residual Encoder-Decoder Network"

