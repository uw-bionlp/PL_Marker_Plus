import numpy as np
import json
import os
import torch
import ast
import sys
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
import tqdm
from tqdm import tqdm_notebook
from transformers import BertTokenizer, BertForTokenClassification, BertForSequenceClassification
import torch
import timeit
from transformers import BertTokenizerFast, BertConfig, BertForTokenClassification

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

class BertForSequenceClassification2(BertForSequenceClassification):
    def __init__(self, config, num_labels_list=[]):
        super().__init__(config)
        self.num_labels_list = num_labels_list
        self.config = config

#         self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.ModuleDict()
        self.classifier['Assertion'] = nn.Linear(config.hidden_size, self.num_labels_list[0], bias=True)
        self.classifier['Anatomy_Parent'] = nn.Linear(config.hidden_size, self.num_labels_list[1], bias=True)
        self.classifier['Anatomy_Child'] = nn.Linear(config.hidden_size, self.num_labels_list[2], bias=True)
        self.classifier['Indication_Type'] = nn.Linear(config.hidden_size, self.num_labels_list[3], bias=True)
        self.classifier['Size'] = nn.Linear(config.hidden_size, self.num_labels_list[4], bias=True)
        self.classifier['Size_Trend'] = nn.Linear(config.hidden_size, self.num_labels_list[5], bias=True)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict= False,
        target_entity = None
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)

        logits_Assertion = self.classifier['Assertion'](pooled_output)
        logits_Anatomy_Parent = self.classifier['Anatomy_Parent'](pooled_output)
        logits_Anatomy_Child = self.classifier['Anatomy_Child'](pooled_output)
        logits_Indication_Type = self.classifier['Indication_Type'](pooled_output)
        logits_Size = self.classifier['Size'](pooled_output)
        logits_Size_Trend = self.classifier['Size_Trend'](pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if len(self.num_labels_list) == 1:
                    self.config.problem_type = "regression"
#                 elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
#                     self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
                    print('multi_label')

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":

                loss_fct = CrossEntropyLoss()
                Assertion_loss = loss_fct(logits_Assertion, labels[:,0])
                Indication_Type_loss = loss_fct(logits_Indication_Type, labels[:,3])
                Anatomy_Parent_loss = loss_fct(logits_Anatomy_Parent, labels[:,1])
                Anatomy_Child_loss = loss_fct(logits_Anatomy_Child, labels[:,2])
                Size_loss = loss_fct(logits_Size,labels[:,4])
                Size_Trend_loss = loss_fct(logits_Size_Trend,labels[:,5])

                loss = Assertion_loss+Indication_Type_loss+Anatomy_Parent_loss+Anatomy_Child_loss+\
                    Size_loss+Size_Trend_loss


#                 loss = loss_fct(logits, labels)
        if not return_dict:
#             print('loss:',loss)
            output = (logits_Assertion, logits_Anatomy_Parent,logits_Anatomy_Child,
                     logits_Indication_Type, logits_Size, logits_Size_Trend) + outputs[2:]
#             return ((loss,) + output) if loss is not None else output
            return ((loss,) + output)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

cuda = sys.argv[1]
target_file = sys.argv[2]


device = "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu"
# load subtype classification model
model = torch.load('./Subtype_model.pt')
model.to(device)

# Tokenize using BERT
transformer = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = BertTokenizerFast.from_pretrained(transformer)

labels_to_ids_Assertion = {'None': 0, 'present': 1, 'absent': 2, 'possible': 3}
ids_to_labels_Assertion = {y:x for x,y in labels_to_ids_Assertion.items()}

labels_to_ids_Anatomy_Parent = {'Abdomen': 0, 'None': 1, 'Hepato-Biliary': 2, 'Urinary': 3, 'Digestive': 4, 'Body_Regions': 5, 'Cardiovascular': 6, 'Respiratory': 7, 'Neurological': 8, 'Thoracic': 9, 'Musculo-Skeletal': 10, 'F_Reproductive_Obstetric': 11, 'Head_Neck': 12, 'Lymphatic': 13, 'M_Reproductive': 14, 'Skin': 15, 'Miscellaneous': 16}
ids_to_labels_Anatomy_Parent = {y:x for x,y in labels_to_ids_Anatomy_Parent.items()}

labels_to_ids_Anatomy_Child = {'Undetermined': 0, 'None': 1, 'Liver': 2, 'Spleen': 3, 'Pancreas': 4, 'Adrenal_Gland': 5, 'Kidney': 6, 'Intestine': 7, 'Pelvis': 8, 'Arterial': 9, 'Lung': 10, 'Large_Intestine': 11, 'Ureter': 12, 'Retroperitoneal': 13, 'Urinary_Bladder': 14, 'Spine_Unspecified': 15, 'Spine_Thoracic': 16, 'Pituitary': 17, 'Extraaxial': 18, 'Cerebrospinal_Fluid_Pathway': 19, 'Skeletal_and_or_Smooth_Muscle': 20, 'Spine_Lumbar': 21, 'Spine_Cord': 22, 'Cerebrovascular_System': 23, 'Brain': 24, 'Bone_and_or_Joint': 25, 'Female_Genital_Structure': 26, 'Neck': 27, 'Mediastinal': 28, 'Upper_Limb': 29, 'Bile_Duct': 30, 'Peritoneal_Sac': 31, 'Abdominal_Wall': 32, 'Gallblader': 33, 'Pleural_Membrane': 34, 'Nasal_Sinus': 35, 'Mesentery': 36, 'Ovary': 37, 'Adnexal': 38, 'Coronary_Artery': 39, 'Esophagus': 40, 'Stomach': 41, 'Small_Intestine': 42, 'Heart': 43, 'Lower_Limb': 44, 'Prostate': 45, 'Thyroid': 46, 'Pericardial_Sac': 47, 'Tracheobronchial': 48, 'Pulmonary_Artery': 49, 'Skin_and_or_Mucous_Membrane': 50, 'Nerve': 51, 'Spine_Sacral': 52, 'Missing_Anatomy': 53, 'Venous': 54, 'Breast': 55, 'Pharynx': 56, 'Ear': 57, 'Spine_Cervical': 58, 'Mouth': 59, 'Biomedical_Device': 60, 'Entire_Body': 61, 'Uterus': 62, 'Eye': 63, 'Connective_Tissue': 64, 'Laryngeal': 65, 'Subcutaneous': 66, 'Testis': 67}
ids_to_labels_Anatomy_Child = {y:x for x,y in labels_to_ids_Anatomy_Child.items()}

labels_to_ids_Indication_Type = {'None': 0, 'symptom': 1, 'nonneoplastic_dx': 2, 'neoplastic_dx': 3, 'trauma': 4}
ids_to_labels_Indication_Type = {y:x for x,y in labels_to_ids_Indication_Type.items()}

labels_to_ids_Size = {'None': 0, 'current': 1, 'past': 2}
ids_to_labels_Size = {y:x for x,y in labels_to_ids_Size.items()}

labels_to_ids_Size_Trend  = {'None': 0, 'increasing': 1, 'new': 2, 'no_change': 3, 'decreasing': 4, 'disappear': 5}
ids_to_labels_Size_Trend = {y:x for x,y in labels_to_ids_Size_Trend.items()}



with open('./incidentaloma_models/PL-Marker-incidentaloma-bertbase-45/{}_ent_pred_test.json'.format(target_file),'r') as f:
    pred = f.readlines()
    entity = [eval(x) for x in pred]

with open('./incidentaloma/{}.json'.format(target_file),'r') as f:
    valid = f.readlines()
    valid = [eval(x) for x in valid]

new_valid = []
for x in valid:
    id_ = x['id']
    for idx, sentence in enumerate(x['sentences']):
        temp_dict ={}
        temp_dict['id'] = id_
        temp_dict['sent_index'] = idx
        temp_dict['tokens'] = sentence
        temp_dict['offsets'] = x['offsets'][idx]
        new_valid.append(temp_dict)

doc_list = []
for x in tqdm_notebook(valid):
#     if x['id'] not in doc_list:
    doc_list.append(x['id'])

a = []
for x in doc_list[:100]:
    if x not in a:
        a.append(x)

def f7(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

new_doc_list = f7(doc_list)
doc_list = new_doc_list

new_json = []


previous_doc_idx = new_valid[0]['id']
token_idx = 0

check=0

pbar = tqdm.tqdm(total=len(new_valid))

for sentence in new_valid:

    if sentence['id'] not in doc_list:
        break

    idx = sentence['sent_index']
    doc_idx = doc_list.index(sentence['id'])


    if previous_doc_idx!=sentence['id']:
        token_idx=0


    pred_entity = [x for x in entity if x['id']==sentence['id']][0]['predicted_ner'][idx]



    new_entity_list = []
    for ner in pred_entity:
        new_dict = {}
        new_dict['type'] = ner[2]
        new_dict['start'] = ner[0]-token_idx
        new_dict['end'] = ner[1]+1-token_idx
        new_entity_list.append(new_dict)



    pred_relation_list = []
    new_relation_list = []


    tokens = sentence['tokens']
    new_subtype_list = []
    for new_entity in new_entity_list:
        temp_subtype_dict ={}
        temp_subtype_dict['type'] = {'Assertion': 'None', 'Anatomy_Parent':'None', 'Anatomy_Child':'None',
                                    'Indication_Type':'None', 'Size':'None', 'Size_Trend':'None'}
        temp_subtype_dict['start'] = new_entity['start']
        temp_subtype_dict['end'] = new_entity['end']

        if new_entity['type'] in ['Medical_Problem', 'Lesion']:
            temp_type = new_entity['type']
            temp_tokens = tokens.copy()
            temp_tokens.insert(new_entity['start'], '<{}>'.format(temp_type))
            temp_tokens.insert(new_entity['end']+1, '<{}>'.format(temp_type))
            temp_sentence = ' '.join(temp_tokens)
#             print('temp sentence:', temp_sentence)
            inputs = tokenizer(temp_sentence, return_tensors="pt", max_length=512, truncation=True).to(device)
            with torch.no_grad():
                logits = model(**inputs)
                pred = logits[1].argmax().item()
                pred_label = ids_to_labels_Assertion[pred]
            temp_subtype_dict['type']['Assertion'] = pred_label



        if new_entity['type'] in ['Indication']:
            temp_type = new_entity['type']
            temp_tokens = tokens.copy()
            temp_tokens.insert(new_entity['start'], '<{}>'.format(temp_type))
            temp_tokens.insert(new_entity['end']+1, '<{}>'.format(temp_type))
            temp_sentence = ' '.join(temp_tokens)
#             print('temp sentence:', temp_sentence)
            inputs = tokenizer(temp_sentence, return_tensors="pt", max_length=512, truncation=True).to(device)
            with torch.no_grad():
                logits = model(**inputs)

                pred_assertion = logits[1].argmax().item()
                pred_assertion_label = ids_to_labels_Assertion[pred_assertion]

                pred_indication_type = logits[4].argmax().item()
                pred_indication_type_label = ids_to_labels_Indication_Type[pred_indication_type]

            temp_subtype_dict['type']['Assertion'] = pred_assertion_label
            temp_subtype_dict['type']['Indication_Type'] = pred_indication_type_label


        if new_entity['type'] in ['Anatomy']:
            temp_type = new_entity['type']
            temp_tokens = tokens.copy()
            temp_tokens.insert(new_entity['start'], '<{}>'.format(temp_type))
            temp_tokens.insert(new_entity['end']+1, '<{}>'.format(temp_type))
            temp_sentence = ' '.join(temp_tokens)
#             print('temp sentence:', temp_sentence)
            inputs = tokenizer(temp_sentence, return_tensors="pt", max_length=512, truncation=True).to(device)
            with torch.no_grad():
                logits = model(**inputs)

                pred_anatomy_parent = logits[2].argmax().item()
                pred_anatomy_parent_label = ids_to_labels_Anatomy_Parent[pred_anatomy_parent]

                pred_anatomy_child = logits[3].argmax().item()
                pred_anatomy_child_label = ids_to_labels_Anatomy_Child[pred_anatomy_child]

#             print(pred_anatomy_parent_label)
#             print(pred_anatomy_child_label)
            temp_subtype_dict['type']['Anatomy_Parent'] = pred_anatomy_parent_label
            temp_subtype_dict['type']['Anatomy_Child'] = pred_anatomy_child_label


        if new_entity['type'] in ['Size']:
            temp_type = new_entity['type']
            temp_tokens = tokens.copy()
            temp_tokens.insert(new_entity['start'], '<{}>'.format(temp_type))
            temp_tokens.insert(new_entity['end']+1, '<{}>'.format(temp_type))
            temp_sentence = ' '.join(temp_tokens)
#             print('temp sentence:', temp_sentence)
            inputs = tokenizer(temp_sentence, return_tensors="pt", max_length=512, truncation=True).to(device)
            with torch.no_grad():
                logits = model(**inputs)
                pred = logits[5].argmax().item()
                pred_label = ids_to_labels_Size[pred]
            temp_subtype_dict['type']['Size'] = pred_label


        if new_entity['type'] in ['Size_Trend']:
            temp_type = new_entity['type']
            temp_tokens = tokens.copy()
            temp_tokens.insert(new_entity['start'], '<{}>'.format(temp_type))
            temp_tokens.insert(new_entity['end']+1, '<{}>'.format(temp_type))
            temp_sentence = ' '.join(temp_tokens)
#             print('temp sentence:', temp_sentence)
            inputs = tokenizer(temp_sentence, return_tensors="pt", max_length=512, truncation=True).to(device)
            with torch.no_grad():
                logits = model(**inputs)
                pred = logits[6].argmax().item()
                pred_label = ids_to_labels_Size_Trend[pred]
            temp_subtype_dict['type']['Size_Trend'] = pred_label


        new_subtype_list.append(temp_subtype_dict)


#     print()
#     print('new_subtype:', new_subtype_list)

    new_sentence = {}
    new_sentence['id'] = sentence['id']
#         new_sentence['doc_text'] = sentence['doc_text']
    new_sentence['sent_index'] = sentence['sent_index']
    new_sentence['tokens'] = sentence['tokens']
    new_sentence['offsets'] = sentence['offsets']
    new_sentence['entities'] = new_entity_list
    new_sentence['subtypes'] = new_subtype_list
    new_sentence['relations'] = new_relation_list


    token_idx+=len(sentence['tokens'])
    previous_doc_idx = sentence['id']

    new_json.append(new_sentence)

    pbar.update(1)
pbar.close()
#     break

with open('{}_ent_pred_test_normalized.json'.format(target_file), 'w') as f:
    json.dump(new_json, f)
