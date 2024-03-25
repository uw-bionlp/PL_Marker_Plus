#!/usr/bin/env bash -l
eval "$(conda shell.bash hook)"
echo "before calling source: $PATH "
echo ""
## `source activate` is deprecated

conda activate plmarker_test
echo "Calling plmarker_test: $PATH"
echo ""
echo "Creating PL-Marker format input"
python3 ./inference_00_create_input.py
echo "Input Creation Done"
echo ""
bash ./inference_01_pred_entities.sh
echo "Entity Extraction Done"
conda deactivate


conda activate mspert_test
echo "Calling mspert_test: $PATH"
bash ./inference_02_add_subtypes.sh
echo "Subtype Extraction Done"
echo ""
bash ./inference_03_add_relations.sh
echo "Realtion Extraction Done"
echo ""
conda deactivate
