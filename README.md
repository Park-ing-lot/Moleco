# Moleco

## Set up
1. Clone our repository.
   ```
   git clone https://github.com/Park-ing-lot/Moleco.git
   ```
2. Download sets of similar molecules from [here](https://1drv.ms/f/s!Av7zLRuxiWW_kLs96rYdzCEkzr04jA?e=CpBmn1) and unzip it in the current directory.
   ```
   unzip feature_extraction.zip
   ```   
3. Clone and set the environments by following the instructions from [MoLFormer](https://github.com/IBM/molformer/tree/main). \
   You also need to download the dataset and pre-trained checkpoints.
   ```
   git clone https://github.com/IBM/molformer.git
   conda activate MolTran_CUDA11
   cp Moleco/* molformer/finetune/
   cd molformer/finetune/
   ```
   
## Train with Moleco
1. Run Moleco (Substructure Prediction & Fingerprint-based Contrastive Learning). \
   We first run Moleco before fine-tuning the model.
   ```
   bash run_continue_qm9.sh
   ```
   You can change tasks by modifying task-related arguments.
   
3. We then fine-tune the model on downstream tasks.
   ```
   bash run_finetune_Moleco_r2.sh
   ```
   You can change tasks by modifying task-related arguments.
