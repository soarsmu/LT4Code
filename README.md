# LT4Code

This repository is the replication package of the research work **"The Devil is in the Tails: How Long-Tailed Code Distributions Impact Large Language Models"**.

In our work we investigate the exsitence and impacts of long-tailed distributions in three downstream tasks, each one using a specific dataset obtained from the state-of-the-art approach of each task. Here we provide everything needed to replicate our experiments. 

There are two ways of replicating our results:
* Use the predictions of each models to replicate the results in RQ1, RQ2, and RQ3;
* Train your own models from scratch and perform the inference to generate predictions.









## Resources

* In the folder for each `task`, we provide:
  * the scripts we used to:
    * `01_train.sh`: fine-tune the models with validation.
    * `02_test.sh`: perform inference using the fine-tuned models.
  * the source code we used to:
    * `run.py`: the main code for training/validating/testing.
    * `utils.py`: the utils functions like metrics and preprocessing.
    * `model.py`: the DL model structure of a SE tool.
    
[Here](https://drive.google.com/file/d/19he5gO4kKaO8a2c74mcY-WqNXxTCyLn4/view?usp=share_link) we stored the the datasets you need in order to replicate our experiments:

* `all_datasets.zip` contains all the processed and split datasets we used: 
  * api_seq_data
    * codebert_data: the format of the datasets for CodeBERT 
    * codet5_data: the format of the datasets for CodeT5
    * mularec_data: the format of the datasets for MulaRec
  * code_review_data
    * codebert_data: the format of the datasets for CodeBERT 
    * codet5_data: the format of the datasets for CodeT5
    * t5_review_data: the format of the datasets for T5_Review
  * vulnerability_data
    * codebert_data: the format of the datasets for CodeBERT 
    * codet5_data: the format of the datasets for CodeT5
    * treevul_data: the format of the datasets for TreeVul

* [Here](https://drive.google.com/file/d/1JYNbkhDoJgDr9_-jHWIkCBt7B6WEFt0Z/view?usp=share_link) store the prediction files. `./generated_predictions.zip` contains the predictions of each models used in analysis.

* `requirments.txt` contains the dependencies needed.


## Install dependencies

 Please install them first.
```
conda create -n lt4code python=3.8 
conda activate lt4code
pip install -r  requirments.txt
```


## Use LTAnalyzer to analyze the exstence and degree of long-tailedness in SE datasets
To run RQ1, you need to make sure that `generated_predictions/` is at the root path of this project. Then,
```
cd  RQ1_and_LTAnalyzer
python LTAnalyzer.py
```
In the python script `LTAnalyzer.py` set up the task name (Line 242) and the corresponding dataset (Line 243) For example:
```
task='api'
data_dir='../all_data/api_seq_data/mularec_data/'
```
The python script `LTAnalyzer.py` will compute the Gini coeffient as well as the visuaization of data distribution in RQ1.


## Use predictions to analyze the impacts of long-tailed distribution on DL models
To compute the results in RQ2 and RQ3, you need to first put `generated_predictions/` at the root path of this project. Then go into the folder:
```
cd RQ2_and_RQ3/
```
Then, we need to choose the file of the task of interest to run. For example:
```
python RQ2_and_3_API.py
```
In the python script `RQ2_and_3_API.py`, it compute the results for different models by reading the predictions in different paths (Line 260).  
For example, 
```
metrics_by_ratios2 = process_results_for_plot("../generated_predictions/api_rec/Results/CodeT5/CE/test_last.gold", \
                                             "../generated_predictions/api_rec/Results/CodeT5/CE/test_last.output", vocab_tokens, freq_vocab, 'codet5')
```
where the path "../generated_predictions/api_rec/Results/CodeT5/CE/" refers to the results of CodeT5 in standard Cross Entropy.

For other tasks, we can run:
```
python RQ2_and_3_Revision.py  
python RQ2_and_3_Vulnerability.py
``` 
Similarly, just change the paths to the predictions and can see the model performances of different models in corresponding tasks.


## Finetune CodeBERT to the tail data (RQ4)
To run the code, ensure that `all_data/` is in the root path of this project. For API sequence dataset,

Training:
```
cd RQ4/Tail-detection/code/
bash 01_train_api.sh  
``` 

Testing:
``` 
bash 02_test_api.sh  
``` 

For other tasks,
```
cd RQ4/Tail-detection/code/
bash 01_train_review.sh  
bash 02_test_review.sh 
bash 01_train_treevul.sh
bash 02_test_treevul.sh
``` 



## Train models from sractch (API Sequence Recommendation)

### CodeBERT
To run the CodeBERT for this task, ensure that `all_data/` is in the root path of this project. 

Training:
```
unizp API_seq_rec.zip
cd API_seq_rec/CodeBERT/CE
bash 01_train.sh 
```

Testing:
```
bash 02_test.sh
```

### CodeT5
To run the CodeT5 for this task, ensure that `all_data/` is in the root path of this project. 

Training:
```
cd API_seq_rec/CodeT5/CE
bash 01_train.sh 
```

Testing:
```
bash 02_test.sh
```

### MulaRec
Please refers to its [replication packages](https://github.com/soarsmu/MulaRec) for details 

Training:
```
cd API_seq_rec/MulaRec/CE
01_train.sh
```

Testing:
```
02_test.sh
```
Note: if want run the Focal Loss for this task. You can visit the corresponding folder for the model such as `API_seq_rec/CodeBERT/FL` and use the training and test scripts as above.


## Train models from sractch (Code Revision Recommendation)

### CodeBERT
To run the CodeBERT for this task, ensure that `all_data/` is in the root path of this project. 

Training:
```
unzip Code_Revision_rec.zip
cd Code_Revision_rec/CodeBERT/CE
bash 01_train.sh 
```

Testing:
```
bash 02_test.sh
```

### CodeT5
To run the CodeT5 for this task, ensure that `all_data/` is in the root path of this project. 

Training:
```
cd Code_Revision_rec/CodeT5/CE
bash 01_train.sh 
```

Testing:
```
bash 02_test.sh
```
Note: if want run the Focal Loss for this task. You can visit the corresponding folder for the model such as `Code_Revision_rec/CodeBERT/FL` and use the training and test scripts as above.

### T5_Review
 
To train the T5_Review models,  the authors used the *Google Colab* service. To replicate the training you will need a **Google Colab** pro account and a **Google Cloud Storage** (GCS) account. Once you have you GCS account you need to set up a new bucket. Please, follow the [guide](https://cloud.google.com/storage/docs/quickstart-console) provided by Google.
Then download the materials of T5_Review [Here](https://zenodo.org/record/5387856#.YTDrPZ4zZyo).

In your GCS bucket upload the content of the archive `automating_code_review.zip` (ontained from the link above). It stored datasets and code. 

Once everything is set you can:
 * Fine-tune a T5_Review model (with or without pre-training) on the downstream task (_code&comment-to-code_ tasks) using their datasets, following the `FineTuning.ipynb` notebook;

Please refers to their [replication packages](https://github.com/RosaliaTufano/code_review_automation) for more details.



## Train models from sractch (Vulnerability Type Prediction)

### TreeVul

TreeVul requires many specific environmental requirments:

1. OS: Ubuntu GPU: NVIDIA GTX 3090.
2. Language: Python (v3.8)
3. CUDA: 11.2
4. Python packages:
   * [PyTorch 1.8.1+cu11](https://pytorch.org/)
   * [AllenNLP 2.4.0](https://allennlp.org/)
   * [Transformers 4.5.1](https://huggingface.co/)

To get more comprehensive understanding or TreeVul, please refers their [replication packages](https://figshare.com/articles/online_resource/TreeVul_-_Replication_Package/19727050) for more details.

To run TreeVul,

Training:
```
unzip Vulnerability_Type_Pred.zip
cd Vulnerability_Type_Pred/TreeVul/CE/TreeVul_code/
CUBLAS_WORKSPACE_CONFIG=:16:8 allennlp train TreeVul/config_treevul.json -s TreeVul/out_treevul/ --include-package TreeVul
```

Testing:
```
python predict.py
```

### CodeBERT
The same requirments as TreeVul.

Training:
```
cd  Vulnerability_Type_Pred/CodeBERT/CE/
CUBLAS_WORKSPACE_CONFIG=:16:8 allennlp train Baseline/config_baseline_codebert.json -s Baseline/out_codebert/ --include-package Baseline
```

Testing:
```
python predict_codebert.py
```

### CodeT5
The same requirments as TreeVul.

Training:
```
cd  Vulnerability_Type_Pred/CodeT5/CE/
CUBLAS_WORKSPACE_CONFIG=:16:8  allennlp train Baseline/config_baseline_codet5.json, -s, Baseline/out_baseline_codet5/, --include-package", Baseline
```

Testing:
```
python 02_predict.py
```

Note: if want run the Focal Loss or LRT for this task. You can visit the corresponding path such as `Vulnerability_Type_Pred/TreeVul/FL/` and use the training and test scripts there.



