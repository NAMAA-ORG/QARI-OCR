# QARI-OCR

This repository contains code for training and evaluating an Arabic OCR model based on vision-language architecture designed for Arabic text extraction and understanding.

## Overview

Our Arabic OCR model is designed to extract and understand Arabic text from images. This repository includes:

- Data preprocessing utilities
- Model evaluation scripts
- Training notebook
- Instructions for data preparation

## Models

We evaluate a vision-language model specifically fine-tuned for Arabic OCR tasks. The model architecture is based on a transformer model with vision capabilities.

## Dataset

The model was trained and evaluated on a custom dataset containing Arabic text images with corresponding text annotations. The dataset is provided in parquet format for anonymity.

## Getting Started

### Installation

1. Clone this repository:
```bash
git clone https://github.com/anonymous/arabic-ocr.git
cd arabic-ocr
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare the dataset:
```bash
bash prepare_dataset.sh
```

### Data Preparation

1. Place your dataset files in the `raw_data` directory
2. Run the data preparation script to convert your data to parquet format:
```bash
python prepare_dataset.py
```

### Evaluation

To evaluate the model's performance:

```bash
python eval.py
```

This will evaluate the model on the test split of the dataset and output metrics including BLEU, WER (Word Error Rate), and CER (Character Error Rate).

## Training

The training code is provided in a Jupyter notebook format (`train.ipynb`). The notebook includes:
- Data preparation
- Model configuration
- Training loop
- Evaluation during training

## Usage

```python
from transformers import AutoModelForCausalLM, AutoProcessor
from utils import vlm_output
from PIL import Image

# Load model and processor
model_path = "models/ocr_model"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_path)

# Load image
image_path = "path/to/image.jpg"
image = Image.open(image_path)

# Extract text from image
extracted_text = vlm_output(image, processor, model, max_tokens=2000)
print(extracted_text)
```

## Metrics

The evaluation script reports the following metrics:
- BLEU: Measures the similarity between the generated text and the reference text
- WER (Word Error Rate): Measures the word-level error rate
- CER (Character Error Rate): Measures the character-level error rate



