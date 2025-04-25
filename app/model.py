from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from config.logs_config import *
from app.preprocessing import preprocess


tokenizer = AutoTokenizer.from_pretrained(settings.model_path)
model = AutoModelForSequenceClassification.from_pretrained(settings.model_path)
model.eval()

label_map = {
    0: "negative",
    1: "neutral",
    2: "positive"
}


def predict(text: str) -> dict:
    try:
        logger_interface.debug(f"Starting BERT prediction for text: {text[:100]}...")

        if not text or len(text.strip()) < 5:
            return {"error": "Text too short or empty"}
        if len(text) > 5000:
            return {"error": "Text too long"}

        text = preprocess(text)

        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).squeeze().tolist()
            pred_index = torch.argmax(logits, dim=1).item()

        return {
            "sentiment": label_map[pred_index],
            "confidence": round(probs[pred_index], 4)
        }

    except Exception as e:
        logger_interface.error(f"Error during BERT prediction: {e}", exc_info=True)
        return {"error": "Internal model error"}
