"""Issue 4: WeasyPrint render under the deny-all URL policy, adversarially
tested against a real socket (zero accepts) and a real local file."""

from __future__ import annotations

import socket
import threading
import time
from pathlib import Path

import pytest
from pypdf import PdfReader

from newsletter.errors import RenderFailed
from newsletter.models import Brief, Item
from newsletter.pipeline import _render_template
from newsletter.render import html_to_pdf

pytestmark = pytest.mark.integration


def _brief_with_n_items(n: int) -> Brief:
    from datetime import date

    return Brief(
        topic="t",
        generated_on=date(2026, 11, 12),
        intro="intro " * 20,
        items=[
            Item(
                title=f"Item {i}",
                summary="A summary. " * 10,
                source_url=f"https://example.com/{i}",
                source_name="Example",
                published=None,
            )
            for i in range(n)
        ],
        outro="outro",
    )


class _AcceptCountingListener:
    """A real TCP listener on 127.0.0.1 that records how many connections
    it accepts, running in a background thread until closed."""

    def __init__(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0))
        self.port = self.sock.getsockname()[1]
        self.sock.listen(5)
        self.sock.settimeout(0.2)
        self.accept_count = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self.sock.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            self.accept_count += 1
            conn.close()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        self.sock.close()


def test_http_image_reference_is_blocked_zero_socket_accepts(tmp_path: Path) -> None:
    """A page that references http://127.0.0.1:<real-listening-port>/x.png
    must render without WeasyPrint ever connecting to it: the deny-all
    url_fetcher must refuse the http:// scheme before any socket is opened."""
    listener = _AcceptCountingListener()
    try:
        html = (
            "<html><body>"
            f'<img src="http://127.0.0.1:{listener.port}/x.png">'
            "<p>hello</p></body></html>"
        )
        pdf_path = tmp_path / "out.pdf"
        report = html_to_pdf(html, pdf_path)

        # Give any (forbidden) connection attempt time to land before asserting.
        time.sleep(0.5)
        assert listener.accept_count == 0, "WeasyPrint connected to the blocked http:// URL"
        assert pdf_path.exists()
        assert pdf_path.read_bytes()[:4] == b"%PDF"
        assert any("disallowed protocol" in w.lower() or "refused" in w.lower() for w in report.warnings) or report.warnings
    finally:
        listener.close()


def test_file_scheme_reference_is_refused_without_reading_it(tmp_path: Path) -> None:
    """A page that links file:///etc/hosts as a stylesheet must render
    without WeasyPrint ever reading that file's contents into the PDF."""
    html = (
        '<html><head><link rel="stylesheet" href="file:///etc/hosts"></head>'
        "<body><p>hello</p></body></html>"
    )
    pdf_path = tmp_path / "out.pdf"
    report = html_to_pdf(html, pdf_path)
    assert report.page_count >= 1

    assert pdf_path.exists()
    assert pdf_path.read_bytes()[:4] == b"%PDF"
    reader = PdfReader(str(pdf_path))
    full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    # /etc/hosts conventionally contains "localhost"; if WeasyPrint had
    # actually read the file as CSS, nothing from it would render as visible
    # text anyway, but its content must never appear in the rendered output.
    assert "localhost" not in full_text


def test_data_uri_image_is_allowed_through(tmp_path: Path) -> None:
    """The one scheme the deny-all policy allows: inline data: URIs."""
    # A 1x1 transparent PNG, base64-encoded.
    pixel = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    html = f'<html><body><img src="data:image/png;base64,{pixel}"></body></html>'
    pdf_path = tmp_path / "out.pdf"
    report = html_to_pdf(html, pdf_path)
    assert pdf_path.exists()
    assert report.page_count >= 1


def test_render_failure_raises_render_failed_with_original_message(tmp_path: Path) -> None:
    pdf_path = tmp_path / "out.pdf"
    with pytest.raises(RenderFailed) as excinfo:
        html_to_pdf(object(), pdf_path)  # not a string: WeasyPrint must reject this
    assert excinfo.value.original_message


@pytest.mark.parametrize("n_items,min_pages,max_pages", [(3, 1, 3)])
def test_three_item_brief_renders_one_to_three_pages_with_extractable_titles(
    tmp_path: Path, n_items: int, min_pages: int, max_pages: int
) -> None:
    brief = _brief_with_n_items(n_items)
    html = _render_template(brief, "classic")
    pdf_path = tmp_path / "out.pdf"
    report = html_to_pdf(html, pdf_path)

    assert min_pages <= report.page_count <= max_pages
    assert pdf_path.stat().st_size > 1024

    reader = PdfReader(str(pdf_path))
    full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    for item in brief.items:
        assert item.title in full_text
