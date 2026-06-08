from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import pipeline
import torch

app = FastAPI(
    title="AI Summarizer API",
    description="Text Summarization using DistilBART",
    version="1.0.0"
)

# -----------------------
# MODEL LOAD (IMPORTANT)
# -----------------------
device = 0 if torch.cuda.is_available() else -1

summarizer = pipeline(
    task="summarization",
    model="sshleifer/distilbart-cnn-12-6",
    device=device
)

# -----------------------
# REQUEST MODEL
# -----------------------
class SummaryRequest(BaseModel):
    text: str
    max_words: int = Field(
        default=100,
        ge=20,
        le=1000,
        description="Maximum words in final summary"
    )

# -----------------------
# RESPONSE MODEL
# -----------------------
class SummaryResponse(BaseModel):
    success: bool
    original_word_count: int
    summary_word_count: int
    summary: str

# -----------------------
# UTILITIES
# -----------------------
def chunk_text(text: str, chunk_size: int = 800):
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
# SUMMARIZER ENDPOINT
# -----------------------
@app.post("/summarize", response_model=SummaryResponse)
def summarize(request: SummaryRequest):

    if not request.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty"
        )

    summaries = []

    # Safe generation limits
    max_length = min(int(request.max_words * 2), 512)
    min_length = max(int(max_length * 0.4), 30)

    try:

        # FIRST PASS: chunk summarization
        for chunk in chunk_text(request.text):

            result = summarizer(
                chunk,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                truncation=True
            )

            summaries.append(result[0]["summary_text"])

        combined_summary = " ".join(summaries)

        # SECOND PASS (if multiple chunks)
        if len(summaries) > 1:
            result = summarizer(
                combined_summary,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                truncation=True
            )

            combined_summary = result[0]["summary_text"]

        final_summary = truncate_words(
            combined_summary,
            request.max_words
        )

        return SummaryResponse(
            success=True,
            original_word_count=len(request.text.split()),
            summary_word_count=len(final_summary.split()),
            summary=final_summary
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )