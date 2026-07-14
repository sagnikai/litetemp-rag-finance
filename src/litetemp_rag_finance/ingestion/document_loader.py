from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import pandas as pd


class DocumentLoader:
    SUPPORTED_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".html", ".xml"}

    def __init__(self, raw_path: str | Path = "data/raw"):
        self.raw_path = Path(raw_path)

    def list_sources(self) -> list[dict[str, Any]]:
        sources = []
        for path in self.raw_path.iterdir():
            if path.is_dir():
                for file in path.rglob("*"):
                    if file.suffix in self.SUPPORTED_EXTENSIONS:
                        sources.append({
                            "path": str(file),
                            "source_id": path.name,
                            "extension": file.suffix,
                            "size": file.stat().st_size,
                        })
        return sources

    def load_file(self, path: str | Path) -> str:
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".txt":
            return path.read_text(encoding="utf-8")
        elif suffix == ".md":
            return path.read_text(encoding="utf-8")
        elif suffix == ".csv":
            df = pd.read_csv(path)
            return df.to_string()
        elif suffix == ".json":
            df = pd.read_json(path)
            return df.to_string()
        elif suffix == ".html":
            text = path.read_text(encoding="utf-8")
            return self._extract_text_from_html(text)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    def load_all(self) -> Generator[tuple[str, str, str], None, None]:
        for source in self.list_sources():
            text = self.load_file(source["path"])
            yield source["source_id"], source["path"], text

    def _extract_text_from_html(self, html: str) -> str:
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "nav", "footer"]):
                tag.decompose()
            return soup.get_text(separator="\n", strip=True)
        except ImportError:
            import re
            clean = re.sub(r"<[^>]+>", " ", html)
            clean = re.sub(r"\s+", " ", clean)
            return clean.strip()
