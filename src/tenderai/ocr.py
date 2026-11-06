"""Gated OCR for empty-text PDFs (Hypothesis H9).

Runs only on PDFs whose embedded text layer is empty, using macOS Apple Vision
(`VNRecognizeTextRequest`, languages `ru-RU` + `en-US`, `.accurate`) with SHA-256
disk caching so repeated runs do not re-render pages.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

_SWIFT_OCR_SCRIPT = """\
import AppKit
import PDFKit
import Vision

let input = URL(fileURLWithPath: CommandLine.arguments[1])
guard let document = PDFDocument(url: input) else {
    fputs("cannot open pdf\\n", stderr)
    exit(2)
}

let scale: CGFloat = 3.0
for pageIndex in 0..<document.pageCount {
    guard let page = document.page(at: pageIndex) else { continue }
    let box = page.bounds(for: .mediaBox)
    let size = NSSize(width: box.width * scale, height: box.height * scale)
    let image = NSImage(size: size)
    image.lockFocus()
    NSColor.white.set()
    NSRect(origin: .zero, size: size).fill()
    if let context = NSGraphicsContext.current?.cgContext {
        context.scaleBy(x: scale, y: scale)
        page.draw(with: .mediaBox, to: context)
    }
    image.unlockFocus()
    guard let cg = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else { continue }
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.recognitionLanguages = ["ru-RU", "en-US"]
    request.usesLanguageCorrection = true
    try VNImageRequestHandler(cgImage: cg).perform([request])
    var lines: [String] = []
    for observation in request.results ?? [] {
        if let best = observation.topCandidates(1).first {
            lines.append(best.string)
        }
    }
    if !lines.isEmpty {
        print("[page \\(pageIndex + 1)]")
        for line in lines {
            print(line)
        }
        print("")
    }
}
"""


def ocr_pdf(
    path: Path,
    sha256: str,
    cache_root: Path | None = None,
) -> tuple[str, int, str]:
    """Return ``(text, duration_ms, cache_state)`` for an empty-text PDF."""
    root = cache_root or Path(".cache")
    cache_file = root / "ocr" / f"{sha256}.json"
    if cache_file.exists():
        try:
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
            return str(payload.get("text", "")), 0, "hit"
        except (OSError, json.JSONDecodeError):
            pass

    if shutil.which("swift") is None:
        return "", 0, "unavailable"

    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "ocr_vision.swift"
        script.write_text(_SWIFT_OCR_SCRIPT, encoding="utf-8")
        completed = subprocess.run(
            ["swift", "-module-cache-path", "/tmp/swift_module_cache", str(script), str(path)],
            check=False,
            capture_output=True,
            timeout=120,
        )
    duration_ms = int((time.perf_counter() - started) * 1000)
    if completed.returncode != 0:
        return "", duration_ms, "error"

    text = completed.stdout.decode("utf-8", errors="replace").strip()
    if text:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(
            json.dumps({"sha256": sha256, "engine": "apple-vision", "text": text}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return text, duration_ms, "miss"
