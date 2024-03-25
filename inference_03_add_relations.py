import numpy as np
import json
import os
import timeit
import torch
import tqdm
import sys

import ast

from typing import List, Optional, Tuple, Union

from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)

from transformers import BertTokenizer, BertForTokenClassification, BertForSequenceClassification
import torch
from tqdm import tqdm_notebook

import numpy as np
import json
import pickle

from transformers import BertTokenizer, BertForTokenClassification
import torch
from torch import nn

import collections
import json
import numpy as np
import os
import random
import re
import shutil
import torch
import torch.nn as nn

from collections import Counter, defaultdict
from datasets import load_dataset, ClassLabel
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.optim import AdamW

from transformers import (
    BertTokenizerFast, BertModel, BertForPreTraining, BertConfig, BertPreTrainedModel,
    DataCollatorWithPadding,
    get_scheduler
)
from transformers.modeling_outputs import SequenceClassifierOutput

from transformers import BertTokenizerFast, BertConfig, BertForTokenClassification

class BertForRelationExtraction(BertPreTrainedModel):
  def __init__(self, config, num_labels):
    super(BertForRelationExtraction, self).__init__(config)
    self.num_labels = num_labels
    # body
    self.bert = BertModel(config)
    # head
    self.dropout = nn.Dropout(config.hidden_dropout_prob)
    self.layer_norm = nn.LayerNorm(config.hidden_size * 2)
    self.linear = nn.Linear(config.hidden_size * 2, self.num_labels)
    self.init_weights()

  def forward(self, input_ids, token_type_ids, attention_mask,
              span_idxs, labels=None):
    outputs = (
        self.bert(input_ids, token_type_ids=token_type_ids,
                  attention_mask=attention_mask,
                  output_hidden_states=False)
            .last_hidden_state)

    sub_maxpool, obj_maxpool = [], []
    for bid in range(outputs.size(0)):
      # span includes entity markers, maxpool across span
      sub_span = torch.max(outputs[bid, span_idxs[bid, 0]:span_idxs[bid, 1]+1, :],
                           dim=0, keepdim=True).values
      obj_span = torch.max(outputs[bid, span_idxs[bid, 2]:span_idxs[bid, 3]+1, :],
                           dim=0, keepdim=True).values
      sub_maxpool.append(sub_span)
      obj_maxpool.append(obj_span)

    sub_emb = torch.cat(sub_maxpool, dim=0)
    obj_emb = torch.cat(obj_maxpool, dim=0)
    rel_input = torch.cat((sub_emb, obj_emb), dim=-1)

    rel_input = self.layer_norm(rel_input)
    rel_input = self.dropout(rel_input)
    logits = self.linear(rel_input)

    if labels is not None:
      loss_fn = nn.CrossEntropyLoss()
      loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
      return SequenceClassifierOutput(loss, logits)
    else:
      return SequenceClassifierOutput(None, logits)

cuda = sys.argv[1]
target_file = sys.argv[2]

device = "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu"
# load Relation Extraction Model
model = torch.load('./RE_model.pt', map_location="cuda:{}".format(cuda))
model.to(device)

label2id = {'Anatomy': 0,
 'Anatomy_Child': 1,
 'Anatomy_Parent': 2,
 'Assertion': 3,
 'Characteristic': 4,
 'Count': 5,
 'Indication': 6,
 'Indication_Type': 7,
 'Lesion': 8,
 'Medical_Problem': 9,
 'Size': 10,
 'Size_Trend': 11}

id2label = {v:k for v,k in enumerate(label2id)}

valid_relations = ['Anatomy',
 'Anatomy_Child',
 'Anatomy_Parent',
 'Assertion',
 'Characteristic',
 'Count',
 'Indication',
 'Indication_Type',
 'Lesion',
 'Medical_Problem',
 'Size',
 'Size_Trend']

BASE_MODEL_NAME = "bert-base-cased"
tokenizer = BertTokenizerFast.from_pretrained(BASE_MODEL_NAME)
# tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
vocab_size_orig = len(tokenizer.vocab)

marker_tokens = []
# entity_types = ["LOC", "PER", "ORG"]
entity_types = valid_relations
for entity_type in entity_types:
    marker_tokens.append("<S:{:s}>".format(entity_type))
    marker_tokens.append("</S:{:s}>".format(entity_type))
    marker_tokens.append("<O:{:s}>".format(entity_type))
    marker_tokens.append("</O:{:s}>".format(entity_type))

tokenizer.add_tokens(marker_tokens)
vocab_size_new = len(tokenizer.vocab)

print("original vocab size:", vocab_size_orig)
print("new vocab size:", vocab_size_new)

valid_relations = sorted(list(valid_relations))
rel_tags = ClassLabel(names=valid_relations)
label2id = {'has':0, 'null':1}
id2label = {0:'has', 1:'null'}


# label2id, id2label

def encode_data(examples):
    tokenized_inputs = tokenizer(examples["tokens"],
                               is_split_into_words=True,
                               truncation=True, max_length=400)
#     print(tokenized_inputs)
    span_idxs = []
    for input_id in tokenized_inputs.input_ids:
        tokens = tokenizer.convert_ids_to_tokens(input_id)

#         try:
        span_idxs.append([
      [idx for idx, token in enumerate(tokens) if token.startswith("<S:")][0],
      [idx for idx, token in enumerate(tokens) if token.startswith("</S:")][0],
      [idx for idx, token in enumerate(tokens) if token.startswith("<O:")][0],
      [idx for idx, token in enumerate(tokens) if token.startswith("</O:")][0]
                        ])
        tokenized_inputs["span_idxs"] = span_idxs
        tokenized_inputs["labels"] = [label2id[label] for label in examples["label"]]

#         except:
#             print(tokens)
#             raise Exception
    return tokenized_inputs


model.to(device)


# used json from inference_02_add_subtypes.py as input
pred_data = json.load(open('{}_ent_pred_test_normalized.json'.format(target_file), "r"))

candidate_relation_list = []

new_dict_list = []


start_time = timeit.default_timer()

with open('{}_ent_pred_test_normalized_with_RE.json'.format(target_file), 'w') as f:

    pbar = tqdm.tqdm(total=len(pred_data))

    for sentence in tqdm_notebook(pred_data):
        sent_dic = {}
    #         if sentence['relations']!=[]:
        original_tokens = sentence['tokens']
        entities = sentence['entities']
        relation_list = []

        if len(entities)>=2:

            sent_dic['sentText'] = ' '.join(original_tokens)
            sent_dic['articleID'] = sentence['id']

            trigger_entity = [x for x in entities if x['type'] in ['Medical_Problem', 'Lesion', 'Indication']]
            non_trigger_entity = [x for x in entities if x['type'] not in ['Medical_Problem', 'Lesion', 'Indication']]

            for trigger in trigger_entity:
                trigger_offset = (trigger['start'], trigger['end'])
                trigger_type = trigger['type']
                for attribute in non_trigger_entity:
                    candidate_dict = {}
                    tokens_with_marker = list([x for x in original_tokens])
                    attribute_type = attribute['type']
                    attribute_offset = (attribute['start'], attribute['end'])

                    tokens_with_marker.insert(attribute_offset[0], '<O:{}>'.format(attribute_type))
                    tokens_with_marker.insert(attribute_offset[1]+1, '</O:{}>'.format(attribute_type))

                    # if trigger comes after the attribute
                    if trigger_offset[0]>=attribute_offset[1]:
                        tokens_with_marker.insert(trigger_offset[0]+2, '<S:{}>'.format(trigger_type))
                        tokens_with_marker.insert(trigger_offset[1]+3, '</S:{}>'.format(trigger_type))

                    else:
                        # if trigger comes before the attribute, no impact on index
                        tokens_with_marker.insert(trigger_offset[0], '<S:{}>'.format(trigger_type))
                        tokens_with_marker.insert(trigger_offset[1]+1, '</S:{}>'.format(trigger_type))


                    candidate_dict['tokens'] = [tokens_with_marker]
                    candidate_dict['label'] = ['has']


                    try:
                        batch = {k: torch.tensor(v).to(device) for k, v in encode_data(candidate_dict).items()}
                    except:
                        pass

                    with torch.no_grad():
                        outputs = model(**batch)
                        predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()


                    # if the relation is 'has', add it to the relations list
                    if id2label[predictions[0]] =='has':
                        temp_relation_dict = {'type':'relation'}

                        for idx, entity in enumerate(entities):
                            if entity['type']==trigger_type and entity['start']==trigger_offset[0] and entity['end']==trigger_offset[1]:
                                head_idx =idx
                                temp_relation_dict['head'] = head_idx
                            if entity['type']==attribute_type and entity['start']==attribute_offset[0] and entity['end']==attribute_offset[1]:
                                tail_idx = idx
                                temp_relation_dict['tail'] = tail_idx


                        relation_list.append(temp_relation_dict)




            new_sentence = sentence.copy()
            new_sentence['relations']= relation_list
        else:

            new_sentence = sentence.copy()
            new_sentence['relations']= []

        new_dict_list.append(new_sentence)
        pbar.update(1)
    pbar.close()
    f.write(json.dumps(new_dict_list))
