from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


SentimentLabel = Literal["positive", "neutral", "negative"]
PipelineStatus = Literal["pending", "active", "completed", "skipped", "error"]
SourceType = Literal["audio", "text"]


class HighlightRange(BaseModel):
    product: str
    text: str
    start: int
    end: int
    label: str | None = None


class ProductSentiment(BaseModel):
    name: str
    entityType: str = "ASPECT"
    sentiment: SentimentLabel
    score: float
    confidence: float
    mentions: int
    context: str
    contexts: list[str] = Field(default_factory=list)
    highlights: list[HighlightRange] = Field(default_factory=list)


class SentimentCounts(BaseModel):
    positive: int = 0
    negative: int = 0
    neutral: int = 0


class SentimentSummary(BaseModel):
    positive: int = 0
    negative: int = 0
    neutral: int = 0
    counts: SentimentCounts = Field(default_factory=SentimentCounts)
    dominant: str = "neutral"
    averageScore: float = 0.0
    totalProducts: int = 0


class ConversionScore(BaseModel):
    probability: float
    label: str
    confidence: float
    features: dict[str, int | float | str]
    model: str


class PipelineStage(BaseModel):
    id: str
    title: str
    status: PipelineStatus
    detail: str


class AnalysisMetadata(BaseModel):
    sourceType: SourceType
    sourceName: str
    language: str | None = None
    processingMs: int
    extractionProvider: str | None = None
    transcriptionConfidence: float | None = None
    whisperModel: str | None = None
    wordCount: int = 0
    sentenceCount: int = 0
    extractionQuality: dict[str, int | float | str] = Field(default_factory=dict)
    createdAt: str


class AnalysisResponse(BaseModel):
    transcript: str
    normalizedText: str
    products: list[ProductSentiment] = Field(default_factory=list)
    highlights: list[HighlightRange] = Field(default_factory=list)
    summary: SentimentSummary = Field(default_factory=SentimentSummary)
    conversionScore: ConversionScore | None = None
    pipeline: list[PipelineStage] = Field(default_factory=list)
    metadata: AnalysisMetadata
