from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import pipeline
import torch

app = FastAPI(
    title="AI Summarizer API",
    version="1.0.0"
)

# -----------------------
# DEVICE CONFIG
# -----------------------
device = -1  # FORCE CPU (IMPORTANT for Render free tier)

# -----------------------
# LOAD MODEL (LIGHTWEIGHT)
# -----------------------
summarizer = pipeline(
    task="summarization",
    model="sshleifer/distilbart-xsum-12-3",  # lighter than cnn version
    device=device
)

# -----------------------
# REQUEST SCHEMA
# -----------------------
class SummaryRequest(BaseModel):
    text: str
    max_words: int = Field(default=100, ge=20, le=500)

# -----------------------
# RESPONSE SCHEMA
# -----------------------
class SummaryResponse(BaseModel):
    success: bool
    original_word_count: int
    summary_word_count: int
    summary: str

# -----------------------
# HELPERS
# -----------------------
def chunk_text(text: str, chunk_size: int = 400):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i + chunk_size])


def truncate_words(text: str, max_words: int):
    return " ".join(text.split()[:max_words])

# -----------------------
# ROUTES
# -----------------------
@app.get("/")
def root():
    return {"message": "AI Summarizer API Running"}

# -----------------------
# MAIN SUMMARIZER
# -----------------------
@app.post("/summarize", response_model=SummaryResponse)
def summarize(request: SummaryRequest):

    text = request.text.strip()

    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    summaries = []

    try:
        # PASS 1: chunk summarization
        for chunk in chunk_text(text):

            result = summarizer(
                chunk,
                max_length=min(request.max_words * 2, 256),
                min_length=max(request.max_words, 30),
                do_sample=False,
                truncation=True
            )

            summaries.append(result[0]["summary_text"])

        combined = " ".join(summaries)

        # PASS 2: refine if multiple chunks
        if len(summaries) > 1:
            result = summarizer(
                combined,
                max_length=min(request.max_words * 2, 256),
                min_length=max(request.max_words, 30),
                do_sample=False,
                truncation=True
            )
            combined = result[0]["summary_text"]

        final = truncate_words(combined, request.max_words)

        return SummaryResponse(
            success=True,
            original_word_count=len(text.split()),
            summary_word_count=len(final.split()),
            summary=final
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))