import numpy as np
from datasets import load_dataset
from utils import url_to_image, extract_text_from_html, vlm_output
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
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

# Load dataset from Hugging Face
dataset = load_dataset("NAMAA-Space/qari-0.3-markdown-mixed-dataset-small")
dataset = dataset['test'].select(range(200))
print(f"Dataset size: {len(dataset)}")
print("Sample row:")
print(dataset[0])

# Load model and processor
model_name = "NAMAA-Space/Qari-OCR-0.3-SNAPSHOT-VL-2B-Instruct-merged"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_name)

# Set maximum token length for generation
max_tokens = 4000

# Initialize metrics tracking
qari_bleus = []
qari_wers = []
qari_cers = []

# Evaluate each sample
for el in range(len(dataset)):
    sample = dataset[el]
    image = url_to_image({'url': sample['image']})
    
    if not image:
        print(f"Skipping sample {el}: No image")
        continue
    
    # Process image with the model
    qari_output = extract_text_from_html(vlm_output(image, processor=processor, model=model, max_tokens=max_tokens))
    
    # Prepare ground truth
    ground_truth = " ".join(extract_text_from_html(sample['text']).split("\n"))
    ground_truth = normalize_text(ground_truth)
    qari_output = normalize_text(qari_output)
    
    try:
        # Compute metrics
        qari_bleu = bleu_metric.compute(predictions=[qari_output], references=[[ground_truth]])['bleu']
        qari_wer = wer_metric.compute(predictions=[qari_output], references=[ground_truth])
        qari_cer = cer_metric.compute(predictions=[qari_output], references=[ground_truth])
        
        # Store metrics
        qari_bleus.append(qari_bleu)
        qari_wers.append(qari_wer)
        qari_cers.append(qari_cer)
        
        print(f"Sample {el}: Qari BLEU = {qari_bleu:.3f}, WER = {qari_wer:.3f}, CER = {qari_cer:.3f}")
    except Exception as e:
        print(f"Error computing metrics for sample {el}: {e}")
        continue

# Calculate average metrics
avg_qari_bleu = sum(qari_bleus) / len(qari_bleus) if qari_bleus else 0
avg_qari_wer = sum(qari_wers) / len(qari_wers) if qari_wers else 0
avg_qari_cer = sum(qari_cers) / len(qari_cers) if qari_cers else 0

# Print summary results
print(f"Average Qari BLEU: {avg_qari_bleu:.3f}")
print(f"Average Qari WER: {avg_qari_wer:.3f}")
print(f"Average Qari CER: {avg_qari_cer:.3f}")

# Save metrics to file
metrics = {
    "qari_bleu": qari_bleus,
    "qari_wer": qari_wers,
    "qari_cer": qari_cers,
    "avg_qari_bleu": avg_qari_bleu,
    "avg_qari_wer": avg_qari_wer,
    "avg_qari_cer": avg_qari_cer
}

with open("evaluation_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Metrics saved to evaluation_metrics.json")
