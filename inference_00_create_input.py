# import pandas as pd
# import numpy as np
# import pickle
import json
import os
# import tqdm
import spacy

from transformers import BertTokenizerFast, BertConfig, BertForTokenClassification

# pkl file containing pandas dataframe (columns: STUDY_ID, AccessionNumber, ModalityTypeCode, ReportTextDeid)
# df = pd.read_pickle('./DB_deid_text_0809.pkl')

input_txt_paths = sorted([x for x in os.listdir('./sample_data')])
input_txts = []

for path in input_txt_paths:
    with open('./sample_data/{}'.format(path), 'r') as f:
        txt = f.read()
        input_txts.append(txt)

# preprocess
# df.drop_duplicates(subset=['AccessionNumber'], inplace=True)
# df = df[~df['ReportTextDeid'].isnull()]
# df['ReportTextDeid'] = df['ReportTextDeid'].apply(lambda x: x.strip())

# tokenizer
spacy_model='en_core_web_sm'
tokenizer = spacy.load(spacy_model)

BASE_MODEL_NAME = "bert-base-uncased"
tokenizer2 = BertTokenizerFast.from_pretrained(BASE_MODEL_NAME, max_length=512, truncation = True)

# Truncate if the number of tokens exceed the max token length for BERT models
def truncate(sent, max_length=512):
    new_sent = sent

    while True:
        if len(tokenizer2.tokenize(str(new_sent)))>max_length:
            new_sent = new_sent[:-1]
        else:
            break
    return new_sent




with open('./incidentaloma/example_input.json', 'w', encoding="utf-8") as f:

    for idx in range(len(input_txts)):
        # input txt file format: 1024_CT.txt (AccesNum_Modality.txt)
        acces_num = input_txt_paths[idx].split('_')[0]
        modality = input_txt_paths[idx].split('_')[1].replace('.txt','')
        text = input_txts[idx]

        text_dict = {}
        text_dict['id'] = '{}_{}'.format(acces_num, modality)

        tokens = []
        offsets = []
        ner = []
        relations = []
        sent_index = []

        doc = tokenizer(text)

        for idx, sent in enumerate(doc.sents):
            # sent = rm_ws(sent)

            new_sent = truncate(sent)

            if new_sent!=sent:
                print(acces_num+'_'+modality, str(sent))

            tok = [t.text for t in new_sent]
            os = [(t.idx, t.idx + len(t.text)) for t in new_sent]

            tokens.append(tok)
            offsets.append(os)
            sent_index.append(idx)
            ner.append([])
            relations.append([])

        text_dict['sentences'] = tokens
        text_dict['ner'] = ner
        text_dict['relations'] = relations
        text_dict['offsets'] = offsets
        text_dict['sent_index'] = sent_index
        text_dict['text'] = text

        json.dump(text_dict,f)
        f.write('\n')
