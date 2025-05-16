import numpy as np
import os
import pandas as pd
from utils import url_to_image, extract_text_from_html, vlm_output
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
import torch
import evaluate
import json
import unicodedata
import re

def normalize_text(text):
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

# Set random seed for reproducibility
np.random.seed(42)

# Load evaluation metrics
bleu_metric = evaluate.load("bleu")
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

# Load dataset from local parquet files
def load_local_dataset(data_dir="data"):
    test_path = os.path.join(data_dir, "test.parquet")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test dataset not found at {test_path}")
    
    test_df = pd.read_parquet(test_path)
    print(f"Dataset size: {len(test_df)}")
    print("Sample row:")
    print(test_df.iloc[0])
    
    # Take only the first 200 samples for testing
    return test_df.head(200)

dataset = load_local_dataset()

# Load model and processor
model_path = "models/ocr_model"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_path)

# Set maximum token length for generation
max_tokens = 4000

# Initialize metrics tracking
model_bleus = []
model_wers = []
model_cers = []

# Evaluate each sample
for idx, row in dataset.iterrows():
    # Assuming 'image_path' or 'image_url' is in the dataset
    image_path = row.get('image_path')
    image_url = row.get('image_url')
    
    if image_path and os.path.exists(image_path):
        image = Image.open(image_path)
    elif image_url:
        image = url_to_image({'url': image_url})
    else:
        print(f"Skipping sample {idx}: No image found")
        continue
    
    if not image:
        print(f"Skipping sample {idx}: Invalid image")
        continue
    
    # Process image with the model
    model_output = extract_text_from_html(vlm_output(image, processor=processor, model=model, max_tokens=max_tokens))
    
    # Prepare ground truth
    ground_truth = row.get('text', '')
    if isinstance(ground_truth, str):
        ground_truth = " ".join(extract_text_from_html(ground_truth).split("\n"))
        ground_truth = normalize_text(ground_truth)
        model_output = normalize_text(model_output)
    else:
        print(f"Skipping sample {idx}: No ground truth text")
        continue
    
    try:
        # Compute metrics
        model_bleu = bleu_metric.compute(predictions=[model_output], references=[[ground_truth]])['bleu']
        model_wer = wer_metric.compute(predictions=[model_output], references=[ground_truth])
        model_cer = cer_metric.compute(predictions=[model_output], references=[ground_truth])
        
        # Store metrics
        model_bleus.append(model_bleu)
        model_wers.append(model_wer)
        model_cers.append(model_cer)
        
        print(f"Sample {idx}: Model BLEU = {model_bleu:.3f}, WER = {model_wer:.3f}, CER = {model_cer:.3f}")
    except Exception as e:
        print(f"Error computing metrics for sample {idx}: {e}")
        continue

# Calculate average metrics
avg_model_bleu = sum(model_bleus) / len(model_bleus) if model_bleus else 0
avg_model_wer = sum(model_wers) / len(model_wers) if model_wers else 0
avg_model_cer = sum(model_cers) / len(model_cers) if model_cers else 0

# Print summary results
print(f"Average Model BLEU: {avg_model_bleu:.3f}")
print(f"Average Model WER: {avg_model_wer:.3f}")
print(f"Average Model CER: {avg_model_cer:.3f}")

# Save metrics to file
metrics = {
    "model_bleu": model_bleus,
    "model_wer": model_wers,
    "model_cer": model_cers,
    "avg_model_bleu": avg_model_bleu,
    "avg_model_wer": avg_model_wer,
    "avg_model_cer": avg_model_cer
}

with open("evaluation_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Metrics saved to evaluation_metrics.json")
