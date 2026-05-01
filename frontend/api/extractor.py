"""PDF text extraction + Gemini AI structured parsing for JudgeAI."""

import json
import os
import re

import io
from pdfminer.high_level import extract_text as miner_extract_text
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

GEMINI_PROMPT = """You are a legal document analyst. Extract the following \
from this Karnataka High Court judgment:
1. case_number (e.g. WP 12345/2024)
2. date (format: YYYY-MM-DD)
3. petitioner (company or person name)
4. respondent (government department name)
5. directions (list of specific court orders/directions)
6. deadline (compliance deadline, format: YYYY-MM-DD, infer from context if not explicit)
7. department (which govt department must comply)

Return ONLY valid JSON. No explanation. Format:
{
  "case_number": "",
  "date": "",
  "petitioner": "",
  "respondent": "",
  "directions": [],
  "deadline": "",
  "department": "",
  "confidence": {
    "case_number": "high/medium/low",
    "date": "high/medium/low",
    "directions": "high/medium/low",
    "deadline": "high/medium/low"
  }
}"""


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Use pdfminer.six to pull all text from a PDF byte stream.

    Raises ValueError if no extractable text is found (scanned PDF).
    """
    try:
        # Create a file-like object from bytes
        fp = io.BytesIO(pdf_bytes)
        text = miner_extract_text(fp)
        cleaned = text.strip()
        
        if not cleaned:
            raise ValueError("scanned PDF - manual entry needed")
        return cleaned
    except Exception as e:
        if isinstance(e, ValueError):
            raise e
        raise RuntimeError(f"PDF extraction error: {str(e)}")


def extract_with_gemini(text: str) -> dict:
    """Send extracted PDF text to Gemini 1.5 Pro and return structured JSON."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in environment")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = f"{GEMINI_PROMPT}\n\n--- DOCUMENT TEXT ---\n{text}"
    response = model.generate_content(prompt)

    raw = response.text.strip()

    # Strip markdown code fences if Gemini wraps the JSON
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Return raw text wrapped in an error structure so the caller
        # can still surface something useful.
        parsed = {
            "error": "Failed to parse Gemini response as JSON",
            "raw_response": raw,
        }

    return parsed
