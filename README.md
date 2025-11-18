# Annotating-Food-in-Context

This repository provides the resources used to study how humans and large language models identify food-related entities and their uses in historical texts. It includes the annotation guidelines, the selected corpus, human and model-generated annotations, the prompting code, and the scripts used for evaluation.

## Contents

### `guidelines.pdf`
The final version of the annotation guidelines.  

### `annotation_final/`
Human annotations in WebAnno/INCEpTION format.  
Each folder corresponds to a document in the corpus and contains the complete annotation structure.

### `prompt_txt.py`
The prompt used to instruct LLMs during automatic annotation, designed to follow the same principles described in the guidelines.

### Conversion Scripts
Two scripts are provided to convert annotations into BIO format:
- **WebAnno → BIO** for human annotations  
- **XML → BIO** for LLM-generated annotations, which are produced in inline XML format to improve stability and accuracy

### `experiments/`
Contains:
- the selected corpus (*texts to annotate*)
- the subset of human annotations
- the subset of LLM annotations used for experiments
- a script to check alignment between human and model annotations
- the full evaluation pipeline in `evaluation_all`

The paper is currently under review.

