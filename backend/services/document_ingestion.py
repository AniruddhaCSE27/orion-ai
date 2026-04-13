from pathlib import Path

from backend.services.vector_store import VectorStore

UPLOADS_DIR = Path(__file__).resolve().parents[1] / "data" / "uploads"


def _extract_pdf_text(file_path: Path):
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF support requires the 'pypdf' package.") from exc

    reader = PdfReader(str(file_path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")

    text = "\n".join(pages).strip()
    page_count = len(reader.pages)
    if not text:
        print(f"WARNING: No readable text extracted from PDF: {file_path.name}")
    print(f"PDF INGESTION: filename={file_path.name}, pages={page_count}, characters={len(text)}")
    return {
        "text": text,
        "page_count": page_count,
        "character_count": len(text),
    }


def _extract_txt_text(file_path: Path):
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    print(f"TXT INGESTION: filename={file_path.name}, characters={len(text)}")
    return {
        "text": text,
        "page_count": 1,
        "character_count": len(text),
    }


def extract_text(file_path: Path):
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
    extraction = extract_text(file_path)
    text = extraction.get("text", "")
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

    print(f"CHUNKING: filename={file_path.name}, chunk_count={len(documents)}")
    return {
        "documents": documents,
        "page_count": extraction.get("page_count", 0),
        "character_count": extraction.get("character_count", 0),
        "chunks_created": len(documents),
    }


def save_uploaded_file(filename: str, content: bytes) -> Path:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    destination = UPLOADS_DIR / filename
    destination.write_bytes(content)
    return destination


def index_document(file_path: Path):
    chunk_result = build_chunk_documents(file_path)
    documents = chunk_result["documents"]
    store = VectorStore(namespace="documents")
    before_count = store.count_documents()
    store.add_documents(documents)
    after_count = store.count_documents()
    print(
        "INDEXING:",
        f"filename={file_path.name},",
        f"extracted_text_length={chunk_result['character_count']},",
        f"chunk_count={chunk_result['chunks_created']},",
        f"indexed_document_count={after_count}",
    )

    return {
        "filename": file_path.name,
        "pages_read": chunk_result["page_count"],
        "characters_extracted": chunk_result["character_count"],
        "chunks_indexed": chunk_result["chunks_created"],
        "indexed_document_count": after_count,
        "new_chunks_added": max(0, after_count - before_count),
        "source_type": "document",
    }
