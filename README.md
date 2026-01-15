# QARI-OCR

This repository contains code for training and evaluating QARI OCR model based on vision-language architecture designed for Arabic text extraction and understanding.

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
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info



BASE_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
ADAPTER_MODEL = "NAMAA-Space/Qari-OCR-0.2.2.1-VL-2B-Instruct" 

base_model = Qwen2VLForConditionalGeneration.from_pretrained(
      BASE_MODEL,
      torch_dtype="auto",
      device_map="auto"
)

model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL)

processor = AutoProcessor.from_pretrained(BASE_MODEL)

IMAGE_PATH = "path/image.jpg"

prompt = "Below is the image of one page of a document, as well as some raw textual content that was previously extracted for it. Just return the plain text representation of this document as if you were reading it naturally. Do not hallucinate."


max_tokens = 2000

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": f"file://{IMAGE_PATH}"},
            {"type": "text", "text": prompt},
        ],
    }
  ]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
      text=[text],
      images=image_inputs,
      videos=video_inputs,
      padding=True,
      return_tensors="pt",
).to("cuda")

generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)
generated_ids_trimmed = [
      out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
      generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]

print("OCR Output:\n", output_text)
```

## Metrics

The evaluation script reports the following metrics:
- BLEU: Measures the similarity between the generated text and the reference text
- WER (Word Error Rate): Measures the word-level error rate
- CER (Character Error Rate): Measures the character-level error rate



