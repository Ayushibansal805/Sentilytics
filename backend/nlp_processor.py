# import re

# from transformers import pipeline

# # Load sentiment model once so repeated calls stay fast.
# sentiment_model = pipeline(
#     "sentiment-analysis",
#     model="distilbert-base-uncased-finetuned-sst-2-english",
# )

# industry_keywords = {
#     "E-commerce": ["delivery", "refund", "order", "shipping", "product"],
#     "Banking": ["bank", "transaction", "payment", "account", "loan"],
#     "Travel": ["flight", "hotel", "booking", "trip", "ticket"],
#     "Food": ["restaurant", "food", "taste", "service", "menu"],
#     "Technology": ["app", "software", "bug", "crash", "update"],
# }


# def clean_text(text: str) -> str:
#     text = text.lower()
#     text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
#     return text.strip()


# def analyze_sentiment(text: str):
#     cleaned_text = clean_text(text)
#     result = sentiment_model(cleaned_text)[0]
#     label = result["label"]
#     score = round(result["score"], 3)
#     return label, score


# def detect_industry(text: str):
#     lowered = text.lower()
#     scores = {}

#     for industry, keywords in industry_keywords.items():
#         score = sum(word in lowered for word in keywords)
#         if score > 0:
#             scores[industry] = score

#     if not scores:
#         return ["General"]

#     sorted_industries = sorted(scores, key=scores.get, reverse=True)
#     return sorted_industries[:3]

import re
from functools import lru_cache

from transformers import pipeline


# Load the model only when it is first needed.
# This avoids loading the model immediately when FastAPI starts.
@lru_cache(maxsize=1)
def get_sentiment_model():
    return pipeline(
        task="sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1,  # Force CPU; no GPU required on Render
    )


industry_keywords = {
    "E-commerce": ["delivery", "refund", "order", "shipping", "product"],
    "Banking": ["bank", "transaction", "payment", "account", "loan"],
    "Travel": ["flight", "hotel", "booking", "trip", "ticket"],
    "Food": ["restaurant", "food", "taste", "service", "menu"],
    "Technology": ["app", "software", "bug", "crash", "update"],
}


def clean_text(text: str) -> str:
    """Normalize text before sentiment analysis."""
    if not isinstance(text, str):
        raise TypeError("Input text must be a string.")

    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.strip()


def analyze_sentiment(text: str):
    """Return the sentiment label and confidence score."""
    cleaned_text = clean_text(text)

    if not cleaned_text:
        return "NEUTRAL", 0.0

    model = get_sentiment_model()
    result = model(cleaned_text, truncation=True)[0]

    label = result["label"]
    score = round(float(result["score"]), 3)

    return label, score


def detect_industry(text: str):
    """Identify likely industries based on matching keywords."""
    if not isinstance(text, str):
        raise TypeError("Input text must be a string.")

    lowered = text.lower()
    scores = {}

    for industry, keywords in industry_keywords.items():
        score = sum(
            1 for keyword in keywords
            if re.search(rf"\b{re.escape(keyword)}\b", lowered)
        )

        if score > 0:
            scores[industry] = score

    if not scores:
        return ["General"]

    sorted_industries = sorted(
        scores,
        key=lambda industry: scores[industry],
        reverse=True,
    )

    return sorted_industries[:3]
