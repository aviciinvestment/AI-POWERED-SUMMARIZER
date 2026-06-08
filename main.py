from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import pipeline
import torch

app = FastAPI(
    title="AI Summarizer API",
    description="Text Summarization using BART",
    version="1.0.0"
)

# Load model once
device = 0 if torch.cuda.is_available() else -1

summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
    device=device
)


class SummaryRequest(BaseModel):
    text: str
    max_words: int = Field(
        default=100,
        ge=20,
        le=1000,
        description="Maximum words in final summary"
    )


class SummaryResponse(BaseModel):
    success: bool
    original_word_count: int
    summary_word_count: int
    summary: str


def chunk_text(text: str, chunk_size: int = 800):
    words = text.split()

    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i + chunk_size])


def truncate_words(text: str, max_words: int):
    words = text.split()
    return " ".join(words[:max_words])


@app.get("/")
def root():
    return {
        "message": "AI Summarizer API Running"
    }


@app.post(
    "/summarize",
    response_model=SummaryResponse
)
def summarize(request: SummaryRequest):

    if not request.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty"
        )

    summaries = []

    # Approximate token conversion
    max_length = min(int(request.max_words * 2), 512)
    min_length = max(int(max_length * 0.5), 30)

    early_stopping=True

    try:

        for chunk in chunk_text(request.text):

            result = summarizer(
                chunk,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )

            summaries.append(
                result[0]["summary_text"]
            )

        combined_summary = " ".join(summaries)

        # Second-pass summarization
        if len(summaries) > 1:

            result = summarizer(
                combined_summary,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )

            combined_summary = result[0]["summary_text"]

        final_summary = truncate_words(
            combined_summary,
            request.max_words
        )

        return SummaryResponse(
            success=True,
            original_word_count=len(
                request.text.split()
            ),
            summary_word_count=len(
                final_summary.split()
            ),
            summary=final_summary
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )