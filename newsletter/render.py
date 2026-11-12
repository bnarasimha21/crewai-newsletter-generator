"""Render HTML to PDF with WeasyPrint under a deny-all resource policy.

Confirmed on weasyprint==70.0 by reading weasyprint/urls.py: the
`url_fetcher` argument to `weasyprint.HTML(string=..., url_fetcher=...)` is
just a callable `url_fetcher(url) -> response`, where `response` can be a
plain dict with keys like `string`/`filename`/`file_obj` (see
`URLFetcher.fetch`'s return value, a `URLFetcherResponse`) or an exception
can be raised instead. WeasyPrint's `fetch()` helper (also in urls.py)
catches everything except subclasses of `FatalURLFetchingError` and turns
it into a logged warning plus a resource that fails to load, which is
exactly what we want for a blocked scheme: the page renders anyway, minus
the blocked resource, rather than crashing the whole PDF.

Deny-all policy: only `data:` URLs (inline, no network/filesystem access)
are allowed through, by delegating to WeasyPrint's own default
`URLFetcher`. `http://`, `https://` and `file://` are refused before any
socket or filesystem call is made.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from weasyprint import HTML
from weasyprint.urls import URLFetcher

from newsletter.errors import RenderFailed

logger = logging.getLogger("newsletter.render")


def new_deny_all_url_fetcher() -> URLFetcher:
    """Build a fresh deny-all url_fetcher for one render.

    Confirmed by reading weasyprint/urls.py: WeasyPrint's internal
    `fetch()` helper requires the url_fetcher to be an object with a
    `_fail_on_errors` attribute (it reads `url_fetcher._fail_on_errors`
    directly), not a plain function - so a bare callable raises
    AttributeError instead of being treated as a refusal. `URLFetcher`
    itself already supports this via its `allowed_protocols` constructor
    argument (checked in `URLFetcher.fetch` before any socket or file is
    opened): any scheme not in the set raises `ValueError` there, which
    WeasyPrint's `fetch()` catches and turns into a logged warning plus a
    resource that fails to load, never a crash.

    A fresh instance per render, not a shared module-level singleton:
    `URLFetcher.open()` stashes the pending `Request` on `self._request`
    before calling `fetch()`, so one instance reused across concurrent
    renders (see issue 1's parallel-`generate()` requirement) could hand
    one render's pending request to another's `fetch()` call.
    """
    return URLFetcher(allowed_protocols={"data"})


@dataclass
class RenderReport:
    """What happened during one html_to_pdf call."""

    pdf_path: Path
    page_count: int
    byte_size: int
    warnings: list[str] = field(default_factory=list)


class _WarningCapture(logging.Handler):
    """Collects WeasyPrint's own log warnings (e.g. a blocked resource)."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record.getMessage())


def html_to_pdf(html: str, pdf_path: str | Path) -> RenderReport:
    """Render `html` to a PDF at `pdf_path` using WeasyPrint.

    Never touches the network or filesystem for linked resources: the
    deny-all url_fetcher above is always used. Bundle any CSS/fonts the
    template needs as `data:` URIs or inline `<style>` rather than linked
    files, since linked `file://`/relative-path resources are refused too.

    Raises:
        RenderFailed: wrapping whatever WeasyPrint raised, with the
            original exception message preserved on `.original_message`.
    """
    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    capture = _WarningCapture()
    weasy_logger = logging.getLogger("weasyprint")
    weasy_logger.addHandler(capture)
    try:
        document = HTML(string=html, url_fetcher=new_deny_all_url_fetcher()).render()
        document.write_pdf(target=str(pdf_path))
    except Exception as exc:  # noqa: BLE001 - re-raised as a typed error below
        raise RenderFailed(
            f"WeasyPrint failed to render the newsletter PDF: {exc}",
            original_message=str(exc),
        ) from exc
    finally:
        weasy_logger.removeHandler(capture)

    if not pdf_path.exists():
        raise RenderFailed("WeasyPrint reported success but no PDF file was written.")

    byte_size = pdf_path.stat().st_size
    return RenderReport(
        pdf_path=pdf_path,
        page_count=len(document.pages),
        byte_size=byte_size,
        warnings=capture.records,
    )
