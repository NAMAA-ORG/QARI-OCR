from bs4 import BeautifulSoup
import re
import evaluate
from PIL import Image
import requests
from io import BytesIO
from qwen_vl_utils import process_vision_info

# Initialize metrics
cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")

def extract_text_from_html(html_content):
    """Extract plain text from HTML while preserving structure."""
    if not html_content or not isinstance(html_content, str):
        return ""
    try:
        soup = BeautifulSoup(html_content, "html.parser")
        text = soup.get_text(separator=" ").strip()
        return re.sub(r"\s+", " ", text).strip()
    except Exception as e:
        print(f"Error extracting text from HTML: {e}")
        return ""

def compute_metrics(example, reference_col="reviewedContent", prediction_col="ocrOutput"):
    """
    Compute WER and CER between reference and prediction columns.

    Args:
        example (dict): A single example from the dataset.
        reference_col (str): Name of the reference column (default: "reviewedContent").
        prediction_col (str): Name of the prediction column (default: "ocrOutput").

    Returns:
        dict: Dictionary with 'wer' and 'cer' keys, containing metric values or None for invalid inputs.
    """
    result = {"wer": None, "cer": None}

    reference_raw = example.get(reference_col, "")
    prediction_raw = example.get(prediction_col, "")

    if not isinstance(reference_raw, str):
        reference_raw = str(reference_raw) if reference_raw is not None else ""
    if not isinstance(prediction_raw, str):
        prediction_raw = str(prediction_raw) if prediction_raw is not None else ""

    prediction_raw = " ".join(prediction_raw.split("\n")).strip()

    reference = extract_text_from_html(reference_raw)
    prediction = extract_text_from_html(prediction_raw)

    if not reference or not prediction:
        return result

    try:
        wer = wer_metric.compute(references=[reference], predictions=[prediction])
        result["wer"] = wer
    except Exception as e:
        print(f"Error computing WER: {e}")

    try:
        cer = cer_metric.compute(predictions=[prediction], references=[reference])
        result["cer"] = cer
    except Exception as e:
        print(f"Error computing CER: {e}")

    return result

def is_good_row(example, wer_threshold=0.5, cer_threshold=0.3):
    """
    Check if a row is of good quality based on WER and CER.

    Args:
        example (dict): A single example from the dataset.
        wer_threshold (float): Maximum acceptable WER (default: 0.5).
        cer_threshold (float): Maximum acceptable CER (default: 0.3).

    Returns:
        bool: True if the row is good, False otherwise.
    """
    wer = example.get("wer", float("inf"))
    cer = example.get("cer", float("inf"))
    if wer is None or cer is None or wer == float("inf") or cer == float("inf"):
        return False
    if wer > wer_threshold or cer > cer_threshold:
        return False
    return True

def url_to_image(data, save_path=None):
    """
    Convert a URL from a dictionary to an image.

    Args:
        data (dict): Dictionary containing the URL (e.g., {'url': 'https://...'}).
        save_path (str, optional): Path to save the image. If None, image is not saved.

    Returns:
        PIL.Image: Image object if successful, None if failed.
    """
    try:
        url = data.get('url')
        if not url:
            raise ValueError("No 'url' key found in the input dictionary.")
        response = requests.get(url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        if save_path:
            image.save(save_path)
            print(f"Image saved to {save_path}")
        return image
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image: {e}")
        return None
    except ValueError as e:
        print(f"Error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None
    

def vlm_output(image, processor, model, max_tokens=2000):
    """
    Process an image with a vision language model to extract text.
    
    Args:
        image (PIL.Image): Input image.
        processor: The model processor.
        model: The vision language model.
        max_tokens (int): Maximum number of tokens to generate.
        
    Returns:
        str: Extracted text from the image.
    """
    prompt = "Extract the Arabic text from the following image"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    inputs = inputs.to("cuda")
    
    generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    vlm_output = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return vlm_output
