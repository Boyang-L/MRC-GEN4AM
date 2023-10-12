# MRC-GEN4AM

# Prerequisites
python                    3.9.0   
argparse                  1.4.0  
dgl                       1.0.1   
numpy                     1.24.2   
scikit-learn              1.2.2  
torch                     2.0.0  
transformers              4.27.4  
tqdm                      4.65.0  

# Usage
The code for different task are in different directory. For example, rct_acc contains the code for the ACC task on the AbstRCT dataset.  
To conduct the experiments, please run the batch_train.sh in scripts file. For the PE dataset, you should prepare the data using the code in *datapreprocessing*
