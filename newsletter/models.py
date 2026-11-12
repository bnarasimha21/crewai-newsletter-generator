"""Structured research and brief models shared by the crew, templates and evals."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class Item(BaseModel):
    """One researched or written newsletter item."""

    title: str = Field(description="Short headline for the item")
    summary: str = Field(description="One to three sentence summary")
    source_url: str = Field(description="URL of the source this item came from")
    source_name: str = Field(description="Human readable name of the source/publication")
    published: date | None = Field(
        default=None, description="Publication date if known, else null"
    )


class ResearchItems(BaseModel):
    """Raw output of the research task: candidate items before the writer runs."""

    topic: str
    items: list[Item] = Field(default_factory=list)


class Brief(BaseModel):
    """Final structured newsletter content, ready for template rendering."""

    topic: str
    generated_on: date
    intro: str
    items: list[Item] = Field(default_factory=list)
    outro: str
