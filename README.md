# LG-VQA

The repository contains the code for the paper [Language Guided Visual Question Answering: Elevate Your Multimodal Language Model Using Knowledge-Enriched Prompts](https://arxiv.org/abs/2310.20159) published at Findings of EMNLP 2023.

# Usage

Download the images for the respective datasets and update the image paths in the `data/*/*.json` files.

The code for guidance genenration can be found in [guidance](https://github.com/declare-lab/LG-VQA/tree/main/guidance). We have pre-computed the guidances and uploaded it in the `data` folder.

The VQA models can be trained using the `train.py` script. Some examples commands are shown in the `run.sh` file.
