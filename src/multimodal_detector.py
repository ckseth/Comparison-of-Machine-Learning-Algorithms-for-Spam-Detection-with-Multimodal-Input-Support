"""
Multimodal Spam Detection Pipeline
Implementation supporting Text, PDF documents, and Image OCR inputs.
Paper: 'Comparison of Machine Learning Algorithms for Spam Detection with Multimodal Input Support'
APCIT 2024 / IEEE
"""

import os
import re
import argparse
from typing import Dict, Any, Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier


class MultimodalSpamDetector:
    """Multimodal Spam Detector supporting Text, PDF documents, and Images (OCR)."""

    def __init__(self, dataset_path: Optional[str] = None):
        self.vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2, stop_words="english")
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        self.is_trained = False

        if dataset_path and os.path.exists(dataset_path):
            self.train(dataset_path)

    def train(self, dataset_path: str):
        """Train the classifier using the SMS Spam Collection dataset."""
        print(f"Training MultimodalSpamDetector with dataset: {dataset_path}...")
        df = pd.read_csv(dataset_path, sep="\t", header=None, names=["label", "message"], encoding="utf-8")
        df = df.dropna(subset=["label", "message"]).drop_duplicates()
        df = df[df["label"].isin(["ham", "spam"])].reset_index(drop=True)

        X = self.vectorizer.fit_transform(df["message"])
        y = df["label"]
        self.classifier.fit(X, y)
        self.is_trained = True
        print("[OK] MultimodalSpamDetector successfully trained.")

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract plain text from a PDF document."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        extracted_text = []
        try:
            import pypdf
            reader = pypdf.PdfReader(pdf_path)
            for page in reader.pages:
                txt = page.extract_text()
                if txt:
                    extracted_text.append(txt)
        except ImportError:
            try:
                import PyPDF2
                reader = PyPDF2.PdfReader(pdf_path)
                for page in reader.pages:
                    txt = page.extract_text()
                    if txt:
                        extracted_text.append(txt)
            except ImportError:
                print("Warning: Neither 'pypdf' nor 'PyPDF2' is installed. Reading binary text fallback.")
                with open(pdf_path, "rb") as f:
                    content = f.read().decode("latin1", errors="ignore")
                    extracted_text.append(content[:1000])

        return "\n".join(extracted_text).strip()

    def extract_text_from_image(self, image_path: str) -> str:
        """Extract text from an image using Optical Character Recognition (OCR)."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        try:
            import pytesseract
            from PIL import Image
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)
            return text.strip()
        except ImportError:
            print("Note: 'pytesseract' is not installed. To enable OCR on images, install pytesseract and Tesseract-OCR engine.")
            return ""

    def analyze_spam_indicators(self, text: str) -> Dict[str, Any]:
        """Analyze text for heuristic spam indicators and summarization."""
        # Find URLs
        urls = re.findall(r"https?://\S+|www\.\S+", text)
        # Find Phone numbers / short codes
        phones = re.findall(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b|\b\d{5,6}\b", text)
        # Check high-risk spam keywords
        spam_keywords = ["free", "win", "winner", "prize", "cash", "urgent", "claim", "congratulations", "offer", "discount"]
        found_keywords = [kw for kw in spam_keywords if re.search(r"\b" + kw + r"\b", text, re.IGNORECASE)]

        summary = text[:200] + ("..." if len(text) > 200 else "")

        return {
            "summary": summary,
            "char_count": len(text),
            "word_count": len(text.split()),
            "urls_detected": urls,
            "phone_numbers_detected": phones,
            "suspicious_keywords": found_keywords
        }

    def predict(self, text: str) -> Dict[str, Any]:
        """Predict whether the input text is Spam or Ham."""
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet. Call train() or provide dataset_path during init.")

        if not text.strip():
            return {
                "prediction": "Unknown",
                "confidence": 0.0,
                "note": "Input text was empty or could not be extracted."
            }

        vec = self.vectorizer.transform([text])
        prediction = self.classifier.predict(vec)[0]
        proba = self.classifier.predict_proba(vec)[0]
        confidence = max(proba)

        indicators = self.analyze_spam_indicators(text)

        return {
            "prediction": prediction.upper(),
            "confidence": round(float(confidence), 4),
            "details": indicators
        }

    def process_multimodal_input(self, input_path_or_text: str, modality: str = "text") -> Dict[str, Any]:
        """
        Process any input according to its modality ('text', 'pdf', or 'image').
        """
        modality = modality.lower()
        if modality == "pdf":
            raw_text = self.extract_text_from_pdf(input_path_or_text)
        elif modality == "image":
            raw_text = self.extract_text_from_image(input_path_or_text)
        elif modality == "text":
            if os.path.exists(input_path_or_text):
                with open(input_path_or_text, "r", encoding="utf-8", errors="ignore") as f:
                    raw_text = f.read()
            else:
                raw_text = input_path_or_text
        else:
            raise ValueError(f"Unsupported modality: {modality}. Choose 'text', 'pdf', or 'image'.")

        result = self.predict(raw_text)
        result["modality"] = modality
        result["input"] = input_path_or_text
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal Spam Detector CLI")
    parser.add_argument("--dataset", type=str, default="SMSSpamCollection", help="Path to SMSSpamCollection")
    parser.add_argument("--input", type=str, default="Free entry in 2 a wkly comp to win FA Cup final tickets! Text FA to 87121", help="Input text or file path")
    parser.add_argument("--modality", type=str, default="text", choices=["text", "pdf", "image"], help="Input modality type")
    args = parser.parse_args()

    detector = MultimodalSpamDetector(dataset_path=args.dataset)
    res = detector.process_multimodal_input(args.input, modality=args.modality)
    print("\n" + "=" * 40)
    print("MULTIMODAL SPAM DETECTION RESULT")
    print("=" * 40)
    print(f"Modality:   {res['modality']}")
    print(f"Prediction: {res['prediction']} (Confidence: {res.get('confidence', 0.0):.2%})")
    print(f"Details:    {res.get('details', {})}")
