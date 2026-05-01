
# IRAP
The repository for the ACL paper "Conjecture and Inquiry: Quantifying Software Performance Requirements via Interactive Retrieval-Augmented Preference Elicitation"

## Environment Setup
Execute the following command in the root directory:
```bash
pip install -e .
```
All dependency packages listed in `install_requires` of `setup.py` will be installed automatically.

## Code Overview
The `IRAP` directory contains the following contents:
```
Anal_reason  
Base_gen  
Query
base_op.py  infer.py  KM.py  
```
Among them, `Base_gen` is the code implementation for the basic quantification form generation phase of IRAP; `Anal_reason` is the code implementation for the analogical reasoning phase of IRAP; `Query` is the code implementation for the iterative interaction phase of IRAP.

The `sota` directory contains the following contents:
```
LLM_api 
RAG  
RLHF
```
These directories implement the three types of SOTA methods involved in this paper, respectively.

The `dataset` folder is used to store datasets.

The `expriments` folder stores experimental results and experimental code.

## IRAP Usage Example
The `IRAPE/infer.py` file provides a usage example of IRAP.

## Experiment Preparation
### Training the IRAP Model
Execute the following commands to fine-tune the Roberta model and GPT2 model, respectively. The training results will be saved in `IRAPE/Base_gen/models`.
```bash
python IRAPE/Base_gen/retrieval_classify.py
python IRAPE/Base_gen/extract_num.py
```

### Reinforcement Learning with DPO
```bash
python sota/RLHF/DPO/train.py
```
The trained model will be saved in `sota/RLHF/models`.

### Reinforcement Learning with WPO
```bash
python sota/RLHF/WPO/train.py
```
The trained model will be saved in `sota/RLHF/models`.

## Running Experiments
### Main Experiments
Measure the performance of each method on the preference inference task in performance requirement quantification, respectively.
```bash
python expriments/main_exp/IRAPE.py
python expriments/main_exp/LLM_api.py
python expriments/main_exp/RAG.py
python expriments/main_exp/DPO.py
python expriments/main_exp/WPO.py
```
Experimental results will be saved in the `expriments/main_exp/res` folder.

### Ablation Experiments
```bash
python expriments/ablation/ablation.py
```
Experimental results will be saved in the `expriments/ablation/res` folder.

### Trade-off Between Interaction Cost and Performance
```bash
python expriments/trade_off/trade_off.py
```
Experimental results will be saved in the `expriments/trade_off/res` folder.


