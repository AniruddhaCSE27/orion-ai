from pathlib import Path

from backend.services.vector_store import VectorStore

UPLOADS_DIR = Path(__file__).resolve().parents[1] / "data" / "uploads"


def _extract_pdf_text(file_path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF support requires the 'pypdf' package.") from exc

    reader = PdfReader(str(file_path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def _extract_txt_text(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8", errors="ignore")


def extract_text(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf_text(file_path)
    if suffix == ".txt":
        return _extract_txt_text(file_path)
    raise ValueError(f"Unsupported file type: {suffix}")


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200):
    clean_text = " ".join((text or "").split())
    if not clean_text:
        return []

    chunks = []
    start = 0
    while start < len(clean_text):
        end = min(len(clean_text), start + chunk_size)
        chunks.append(clean_text[start:end])
        if end >= len(clean_text):
            break
        start = max(0, end - overlap)
    return chunks


def build_chunk_documents(file_path: Path):
    text = extract_text(file_path)
    chunks = chunk_text(text)
    documents = []

    for chunk_id, chunk in enumerate(chunks, start=1):
        documents.append(
            {
                "source_type": "document",
                "source_filename": file_path.name,
                "chunk_id": chunk_id,
                "title": file_path.name,
                "url": "",
                "content": chunk,
                "text": chunk,
            }
        )

    return documents


def save_uploaded_file(filename: str, content: bytes) -> Path:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    destination = UPLOADS_DIR / filename
    destination.write_bytes(content)
    return destination


def index_document(file_path: Path):
    documents = build_chunk_documents(file_path)
    store = VectorStore(namespace="documents")
    store.add_documents(documents)

    return {
        "filename": file_path.name,
        "chunks_indexed": len(documents),
        "source_type": "document",
    }
