# Crisis Consolidated Dataset



## Datasets
* Download dataset:

Before trying to start running any script, please download the dataset first. More detail of this dataset can be found here: https://crisisnlp.qcri.org/CrisisConsolidated.html and the associated published papers.

* Download the dataset (https://crisisnlp.qcri.org/data/CrisisConsolidated/CrisisConsolidated_v1.0.tar.gz)

Assuming that your current working directory is YOUR_PATH/multimodal_social_media
```
tar -xvf CrisisConsolidated_v1.0.tar.gz
mv CrisisConsolidated_v1.0/data_image $PWD/
```

**Different Versions of the dataset**
* data/all_events_en
* data/individual_event_en
* data/event_aware_en
* data/class_label_mapped
* data/initial_filtering
* data/data_split_all_lang



## Experiments

### Setting up environments to Run CNN based experiments


For CNN based experiments we used python 2.7

#### Create a virtual environment
```
python -m venv crisis_cnn_env python=2.7
```
#### Activate your virtual environment
```
source $PATH_TO_ENV/crisis_cnn_env/bin/activate
```

#### Install dependencies
```
pip install -r requirements.txt
```

### Setting up environments to Run BERT based experiments

```
conda env create -f environment_crisis_bert_env.yml

```

### word2vec model:
Download the word2vec model and place it under your home or current working directory, (https://crisisnlp.qcri.org/data/lrec2016/crisisNLP_word2vec_model_v1.2.zip)

### CNN
* You need to modify the word2vec model path in ```bin/text_cnn_pipeline_unimodal.py``` script.

```
CUDA_VISIBLE_DEVICES=1 python bin/text_cnn_pipeline_unimodal.py -i data/all_events_en/crisis_consolidated_informativeness_filtered_lang_en_train.tsv \
-v data/all_events_en/crisis_consolidated_informativeness_filtered_lang_en_dev.tsv -t data/all_events_en/crisis_consolidated_informativeness_filtered_lang_en_test.tsv \
--log_file checkpoint_log/informativeness_cnn.txt --w2v_checkpoint w2v_models/data_w2v_info_cnn.model -m models/informativeness_cnn.model -l labeled/informativeness_labeled_cnn.tsv \
-o results/informativeness_results_cnn.txt >&log/text_info_cnn.txt &

CUDA_VISIBLE_DEVICES=0 python bin/text_cnn_pipeline_unimodal.py -i data/all_events_en/crisis_consolidated_humanitarian_filtered_lang_en_train.tsv \
-v data/all_events_en/crisis_consolidated_humanitarian_filtered_lang_en_dev.tsv -t data/all_events_en/crisis_consolidated_humanitarian_filtered_lang_en_test.tsv \
--log_file checkpoint_log/humanitarian_cnn.txt --w2v_checkpoint w2v_models/data_w2v_hum_cnn.model -m models/humanitarian_cnn.model -l labeled/humanitarian_labeled_cnn.tsv \
-o results/humanitarian_results_cnn.txt >&log/text_hum_cnn.txt &

```

### Run experiments with CrisisLex dataset

```
CUDA_VISIBLE_DEVICES=1 python bin/text_cnn_pipeline_unimodal.py -i data/individual_event_en/crisislex_informativeness_filtered_lang_en_train.tsv \
-v data/individual_event_en/crisislex_informativeness_filtered_lang_en_dev.tsv -t data/individual_event_en/crisislex_informativeness_filtered_lang_en_test.tsv \
--log_file checkpoint_log/crisislex_informativeness_cnn.txt --w2v_checkpoint w2v_models/data_w2v_crisislex_informativeness_cnn.model -m models/crisislex_informativeness_cnn.model -l labeled/crisislex_informativeness_labeled_cnn.tsv \
-o results/crisislex_informativeness_results_cnn.txt >&log/text_crisislex_informativeness_cnn.txt &

CUDA_VISIBLE_DEVICES=0 python bin/text_cnn_pipeline_unimodal.py -i data/individual_event_en/crisislex_humanitarian_filtered_lang_en_train.tsv \
-v data/individual_event_en/crisislex_humanitarian_filtered_lang_en_dev.tsv -t data/individual_event_en/crisislex_humanitarian_filtered_lang_en_test.tsv \
--log_file checkpoint_log/crisislex_humanitarian_cnn.txt --w2v_checkpoint w2v_models/data_w2v_crisislex_humanitarian_cnn.model -m models/crisislex_humanitarian_cnn.model -l labeled/crisislex_humanitarian_labeled_cnn.tsv \
-o results/crisislex_humanitarian_results_cnn.txt >&log/text_crisislex_humanitarian_cnn.txt &

```

### Run experiments with CrisisNLP dataset
```
CUDA_VISIBLE_DEVICES=1 python bin/text_cnn_pipeline_unimodal.py -i data/individual_event_en/crisisnlp_informativeness_filtered_lang_en_train.tsv \
-v data/individual_event_en/crisisnlp_informativeness_filtered_lang_en_dev.tsv -t data/individual_event_en/crisisnlp_informativeness_filtered_lang_en_test.tsv \
--log_file checkpoint_log/crisisnlp_informativeness_cnn.txt --w2v_checkpoint w2v_models/data_w2v_crisisnlp_informativeness_cnn.model -m models/crisisnlp_informativeness_cnn.model -l labeled/crisisnlp_informativeness_labeled_cnn.tsv \
-o results/crisisnlp_informativeness_results_cnn.txt >&log/text_crisisnlp_informativeness_cnn.txt &

CUDA_VISIBLE_DEVICES=0 python bin/text_cnn_pipeline_unimodal.py -i data/individual_event_en/crisisnlp_humanitarian_filtered_lang_en_train.tsv \
-v data/individual_event_en/crisisnlp_humanitarian_filtered_lang_en_dev.tsv -t data/individual_event_en/crisisnlp_humanitarian_filtered_lang_en_test.tsv \
--log_file checkpoint_log/crisisnlp_humanitarian_cnn.txt --w2v_checkpoint w2v_models/data_w2v_crisisnlp_humanitarian_cnn.model -m models/crisisnlp_humanitarian_cnn.model -l labeled/crisisnlp_humanitarian_labeled_cnn.tsv \
-o results/crisisnlp_humanitarian_results_cnn.txt >&log/text_crisisnlp_humanitarian_cnn.txt &
```

### Cross model evaluation

#### Informativeness task

```
CUDA_VISIBLE_DEVICES=1 python bin/text_cnn_classifier.py -c models/crisis_consolidated_informativeness_filtered_lang_en_train_text/informativeness_cnn.config \
-d data/individual_event_en/crisisnlp_informativeness_filtered_lang_en_test.tsv -l labeled/crisisnlp_informativeness_filtered_lang_en_test_cnn_model_full_data.tsv -o results/crisisnlp_informativeness_test_results_cnn_model_full_data.txt

CUDA_VISIBLE_DEVICES=1 python bin/text_cnn_classifier.py -c models/crisis_consolidated_informativeness_filtered_lang_en_train_text/informativeness_cnn.config \
-d data/individual_event_en/crisislex_informativeness_filtered_lang_en_test.tsv -l labeled/crisislex_informativeness_filtered_lang_en_test_cnn_model_full_data.tsv -o results/crisislex_informativeness_test_results_cnn_model_full_data.txt

CUDA_VISIBLE_DEVICES=1 python bin/text_cnn_classifier.py -c models/crisisnlp_informativeness_filtered_lang_en_train_text/crisisnlp_informativeness_cnn.config \
-d data/all_events_en/crisis_consolidated_informativeness_filtered_lang_en_test.tsv -l labeled/crisis_consolidated_informativeness_filtered_lang_en_test_cnn_model_crisisnlp_data.tsv -o results/crisis_consolidated_informativeness_test_results_cnn_model_crisisnlp_data.txt

CUDA_VISIBLE_DEVICES=1 python bin/text_cnn_classifier.py -c models/crisislex_informativeness_filtered_lang_en_train_text/crisislex_informativeness_cnn.config \
-d data/all_events_en/crisis_consolidated_informativeness_filtered_lang_en_test.tsv -l labeled/crisis_consolidated_informativeness_filtered_lang_en_test_cnn_model_crisislex_data.tsv -o results/crisis_consolidated_informativeness_test_results_cnn_model_crisislex_data.txt
```

####  Humanitarian task

```
CUDA_VISIBLE_DEVICES=1 python bin/text_cnn_classifier.py -c models/crisis_consolidated_humanitarian_filtered_lang_en_train_text/humanitarian_cnn.config \
-d data/individual_event_en/crisisnlp_humanitarian_filtered_lang_en_test.tsv -l labeled/crisisnlp_humanitarian_filtered_lang_en_test_cnn_model_full_data.tsv -o results/crisisnlp_humanitarian_test_results_cnn_model_full_data.txt

CUDA_VISIBLE_DEVICES=1 python bin/text_cnn_classifier.py -c models/crisis_consolidated_humanitarian_filtered_lang_en_train_text/humanitarian_cnn.config \
-d data/individual_event_en/crisislex_humanitarian_filtered_lang_en_test.tsv -l labeled/crisislex_humanitarian_filtered_lang_en_test_cnn_model_full_data.tsv -o results/crisislex_humanitarian_test_results_cnn_model_full_data.txt

CUDA_VISIBLE_DEVICES=1 python bin/text_cnn_classifier.py -c models/crisislex_humanitarian_filtered_lang_en_train_text/crisislex_humanitarian_cnn.config \
-d data/all_events_en/crisis_consolidated_humanitarian_filtered_lang_en_test.tsv -l labeled/crisis_consolidated_humanitarian_filtered_lang_en_test_cnn_model_crisislex_data.tsv -o results/crisis_consolidated_humanitarian_test_results_cnn_model_crisislex_data.txt

CUDA_VISIBLE_DEVICES=1 python bin/text_cnn_classifier.py -c models/crisisnlp_humanitarian_filtered_lang_en_train_text/crisisnlp_humanitarian_cnn.config \
-d data/all_events_en/crisis_consolidated_humanitarian_filtered_lang_en_test.tsv -l labeled/crisis_consolidated_humanitarian_filtered_lang_en_test_cnn_model_crisisnlp_data.tsv -o results/crisis_consolidated_humanitarian_test_results_cnn_model_crisisnlp_data.txt
```


## BERT model experiments
```
nohup bash bin/bert_multiclass.sh info data/all_events_en/crisis_consolidated_informativeness_filtered_lang_en_train.tsv data/all_events_en/crisis_consolidated_informativeness_filtered_lang_en_dev.tsv data/all_events_en/crisis_consolidated_informativeness_filtered_lang_en_test.tsv info >&log/bert_info.txt &
nohup bash bin/bert_multiclass.sh hum data/all_events_en/crisis_consolidated_humanitarian_filtered_lang_en_train.tsv data/all_events_en/crisis_consolidated_humanitarian_filtered_lang_en_dev.tsv data/all_events_en/crisis_consolidated_humanitarian_filtered_lang_en_test.tsv hum >&log/bert_hum.txt &

```

## Event aware experiments

### CNN

```
CUDA_VISIBLE_DEVICES=1 python bin/text_cnn_pipeline_unimodal.py -i data/event_aware_en/crisis_consolidated_informativeness_filtered_lang_en_w_event_info_train.tsv \
-v data/event_aware_en/crisis_consolidated_informativeness_filtered_lang_en_w_event_info_dev.tsv -t data/event_aware_en/crisis_consolidated_informativeness_filtered_lang_en_w_event_info_test.tsv \
--log_file checkpoint_log/event-aware_informativeness_cnn.txt --w2v_checkpoint w2v_models/data_w2v_event-aware_info_cnn.model -m models/event-aware_informativeness_cnn.model -l labeled/event-aware_informativeness_labeled_cnn.tsv \
-o results/event-aware_informativeness_results_cnn.txt >&log/event-aware_text_info_cnn.txt &

CUDA_VISIBLE_DEVICES=0 python bin/text_cnn_pipeline_unimodal.py -i data/event_aware_en/crisis_consolidated_humanitarian_filtered_lang_en_w_event_info_train.tsv \
-v data/event_aware_en/crisis_consolidated_humanitarian_filtered_lang_en_w_event_info_dev.tsv -t data/event_aware_en/crisis_consolidated_humanitarian_filtered_lang_en_w_event_info_test.tsv \
--log_file checkpoint_log/event-aware-humanitarian_cnn.txt --w2v_checkpoint w2v_models/data_w2v_event-aware_cnn.model -m models/event-aware_humanitarian_cnn.model -l labeled/event-aware_humanitarian_labeled_cnn.tsv \
-o results/event-aware_humanitarian_results_cnn.txt >&log/event-aware_text_hum_cnn.txt &

```

### BERT
```
nohup bash bin/bert_multiclass.sh info-event-aware data/event_aware_en/crisis_consolidated_informativeness_filtered_lang_en_w_event_info_train.tsv data/event_aware_en/crisis_consolidated_informativeness_filtered_lang_en_w_event_info_dev.tsv data/event_aware_en/crisis_consolidated_informativeness_filtered_lang_en_w_event_info_test.tsv info-event-aware >&log/bert_info_event-aware.txt &

nohup bash bin/bert_multiclass.sh hum-event-aware data/event_aware_en/crisis_consolidated_humanitarian_filtered_lang_en_w_event_info_train.tsv data/event_aware_en/crisis_consolidated_humanitarian_filtered_lang_en_w_event_info_dev.tsv data/event_aware_en/crisis_consolidated_humanitarian_filtered_lang_en_w_event_info_test.tsv hum-event-aware >&log/bert_hum_event-aware.txt &

```

## Please cite the following papera if you are using the data:

* *Firoj Alam, Hassan Sajjad, Muhammad Imran and Ferda Ofli, "Standardizing and Benchmarking Crisis-related Social Media Datasets for Humanitarian Information Processing", arxiv. 2020.*

```bib
@inproceedings{crisisConsolidated2020,
  Author = {Firoj Alam and Hassan Sajjad and Muhammad Imran and Ferda Ofli},
  Title = {Standardizing and Benchmarking Crisis-related Social Media Datasets for Humanitarian Information Processing},
  Year = {2020}
}
```

**and the follwing associated papers**

* *Muhammad Imran, Prasenjit Mitra, Carlos Castillo. Twitter as a Lifeline: Human-annotated Twitter Corpora for NLP of Crisis-related Messages. In Proceedings of the 10th Language Resources and Evaluation Conference (LREC), 2016, Slovenia.*
* *A. Olteanu, S. Vieweg, C. Castillo. 2015. What to Expect When the Unexpected Happens: Social Media Communications Across Crises. In Proceedings of the ACM 2015 Conference on Computer Supported Cooperative Work and Social Computing (CSCW '15). ACM, Vancouver, BC, Canada.*
* *A. Olteanu, C. Castillo, F. Diaz, S. Vieweg. 2014. CrisisLex: A Lexicon for Collecting and Filtering Microblogged Communications in Crises. In Proceedings of the AAAI Conference on Weblogs and Social Media (ICWSM'14). AAAI Press, Ann Arbor, MI, USA.*
* *Muhammad Imran, Shady Elbassuoni, Carlos Castillo, Fernando Diaz and Patrick Meier. Practical Extraction of Disaster-Relevant Information from Social Media. In Social Web for Disaster Management (SWDM'13) - Co-located with WWW, May 2013, Rio de Janeiro, Brazil.*
* *Muhammad Imran, Shady Elbassuoni, Carlos Castillo, Fernando Diaz and Patrick Meier. Extracting Information Nuggets from Disaster-Related Messages in Social Media. In Proceedings of the 10th International Conference on Information Systems for Crisis Response and Management (ISCRAM), May 2013, Baden-Baden, Germany.*


```bib
@inproceedings{imran2016lrec,
  author = {Muhammad Imran and Prasenjit Mitra and Carlos Castillo},
  title = {Twitter as a Lifeline: Human-annotated Twitter Corpora for NLP of Crisis-related Messages},
  booktitle = {Proc. of the LREC, 2016},
  year = {2016},
  month = {5},
  publisher = {ELRA},
  address = {Paris, France},
  isbn = {978-2-9517408-9-1},
  language = {english}
 }
 @inproceedings{olteanu2015expect,
  title={What to expect when the unexpected happens: Social media communications across crises},
  author={Olteanu, Alexandra and Vieweg, Sarah and Castillo, Carlos},
  booktitle={Proc. of the 18th ACM Conference on Computer Supported Cooperative Work \& Social Computing},
  pages={994--1009},
  year={2015},
  organization={ACM}
}
@inproceedings{olteanu2014crisislex,
  title={CrisisLex: A Lexicon for Collecting and Filtering Microblogged Communications in Crises.},
  author={Olteanu, Alexandra and Castillo, Carlos and Diaz, Fernando and Vieweg, Sarah},
  booktitle = "Proc. of the 8th ICWSM, 2014",
  publisher = "AAAI press",
  year={2014}
}
@inproceedings{imran2013practical,
  title={Practical extraction of disaster-relevant information from social media},
  author={Imran, Muhammad and Elbassuoni, Shady and Castillo, Carlos and Diaz, Fernando and Meier, Patrick},
  booktitle={Proc. of the 22nd WWW},
  pages={1021--1024},
  year={2013},
  organization={ACM}
}
@inproceedings{imran2013extracting,
  title={Extracting information nuggets from disaster-related messages in social media},
  author={Imran, Muhammad and Elbassuoni, Shady Mamoon and Castillo, Carlos and Diaz, Fernando and Meier, Patrick},
  booktitle={Proc. of the 12th ISCRAM},
  year={2013}
}
```
