# PL-Marker++

This repository contains source code for PL-Marker++, used in our paper _A Novel Corpus of Annotated Medical Imaging Reports and Information Extraction Results Using BERT-based Language Models_ (Accepted @ LREC-COLING2024).

Example input, output data are not included (will be added upon IRB approval).
PL-Marker++, which is the augmented version of PL-Marker, provides the classification of subtypes for extracted entities.

Original PL-Marker implementation can be found at https://github.com/thunlp/PL-Marker

## Step 1. Download repo and required models
 - Download current directory
 - Unzip transformers.zip
 - Download all models from https://drive.google.com/drive/u/0/folders/1eyaqjrMNUJLxAIHxiYrqX4cCapxgZjPj and put in the same directory

## Step 2. Create virtual enviroments

 - Create 2 seperate Conda environments (both using python=3.8.18) using **mspert_req.txt** (mspert) and **plmarker_req.txt** (plmarker).
 conda create -n mspert_test python=3.8.18
 conda activate mspert_test
 pip install -r ./mspert_req.txt

 conda create -n plmarker_test python=3.8.18
 conda activate plmarker_test
 pip install -r ./plmarker_req.txt
 pip install --editable ./transformers

## Step 3. Put radiology reports in "sample_data" folder

 - Input radiology reports should be located in ./sample_data using .txt file format
 - sample.txt is randomly selected from [mtsamples](https://mtsamples.com/site/pages/sample.asp?Type=95-Radiology&Sample=1403-CT%20Abdomen%20&%20Pelvis%20-%202) radiology report (open-source radiology reports)

## Step 4. Run shell script

 - bash ./run_plmarker.sh
   -> This shell script includes entity extraction, subtype extraction and relation extraction.
 - Final output file with entity, subtype and relation information is "./example_input_ent_pred_test_normalized_with_RE.json"
    - Output with only entity extraction can be found in "./incidentaloma_models/PL-Marker-incidentaloma-bertbase-45/example_input_ent_pred_test.json"
    - Output with entity+subtype extraction can be found in "./example_input_ent_pred_test_normalized.json"
 - All predictions are performed in sentence-level.

## Contact

- Namu Park (npark95@uw.edu)
