"""OCR Service - extracts structured receipt data from images using GPT-4o vision
via the emergentintegrations library."""
import os
import json
import uuid
import re
import logging
from openai import OpenAI
import base64
logger = logging.getLogger(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CATEGORIES = ["Transport", "Medical", "Education", "Business", "Other  "]

SYSTEM_PROMPT = f"""You are a receipt OCR engine for South African taxpayers. The user uploads a single receipt/invoice image. Extract the data and return STRICT JSON only (no markdown, no commentary).

Schema:
{{
  "amount": <number, the total amount paid including VAT - never null. If unclear, use 0>,
  "currency": "ZAR" or detected currency code,
  "date": "YYYY-MM-DD" (transaction date; if unclear use today's date),
  "vendor": "<short store/vendor name, max 80 chars>",
  "category": one of {CATEGORIES},
  "confidence": <float 0-1 indicating extraction confidence>,
  "notes": "<optional short text, max 120 chars>"
}}

Category rules:
- "Transport": Uber, Bolt, fuel, petrol, parking, tolls
- "Medical": pharmacy, doctor, dispensary, hospital, Clicks, Dis-Chem prescriptions
- "Education": books, tuition, courses, stationery
- "Business": office supplies, software, professional services
- "Other": groceries, restaurants, entertainment, anything else

Return ONLY the JSON object."""


def _strip_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


async def extract_receipt(image_path: str, mime_type: str):
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extract receipt data and return JSON only."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_data}"
                        }
                    }
                ]
            }
        ],
        temperature=0
    )

    raw = response.choices[0].message.content

    try:
        return json.loads(raw)
    except Exception:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise RuntimeError(f"Invalid JSON returned: {raw}")
