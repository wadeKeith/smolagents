import hashlib
import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Iterable, Mapping

import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer


class CompanyRAGStore:
    """
    Lightweight wrapper around a persistent Chroma vector database that stores
    documents grouped by company name, while also keeping raw snapshots on disk.
    """

    def __init__(
        self,
        persist_directory: str = "rag_vector_db",
        corpus_directory: str = "rag_corpus",
        playbook_directory: str = "rag_playbooks",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 600,
        chunk_overlap: int = 120,
        max_raw_files: int = 120,
    ) -> None:
        self.persist_directory = Path(persist_directory)
        self.corpus_directory = Path(corpus_directory)
        self.playbook_directory = Path(playbook_directory)
        self.playbook_archive_directory = self.playbook_directory / "archive"
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.corpus_directory.mkdir(parents=True, exist_ok=True)
        self.playbook_directory.mkdir(parents=True, exist_ok=True)
        self.playbook_archive_directory.mkdir(parents=True, exist_ok=True)
        self.max_raw_files = max_raw_files

        self._client = chromadb.PersistentClient(path=str(self.persist_directory))
        self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model)
        self._collections: dict[str, chromadb.api.models.Collection.Collection] = {}
        self._text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            AutoTokenizer.from_pretrained("thenlper/gte-small"),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
            strip_whitespace=True,
            separators=["\n\n", "\n", ".", "。", "！", "?", " ", ""],
        )

    # ------------------------------------------------------------------ #
    # Collection helpers
    # ------------------------------------------------------------------ #
    def _slugify(self, value: str) -> str:
        normalized = unicodedata.normalize("NFKD", value)
        ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
        ascii_slug = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_text).strip("-").lower()
        if ascii_slug:
            return ascii_slug[:128]

        digest = hashlib.sha256(unicodedata.normalize("NFC", value).encode("utf-8")).hexdigest()[:32]
        return f"h-{digest}"

    def _collection_name(self, company_name: str) -> str:
        return f"company_{self._slugify(company_name)}"

    def _get_collection(self, company_name: str):
        collection_name = self._collection_name(company_name)
        if collection_name not in self._collections:
            self._collections[collection_name] = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"company_name": company_name},
                embedding_function=self._embedding_fn,
            )
        return self._collections[collection_name]

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def add_documents(
        self,
        company_name: str,
        contents: Iterable[str],
        metadata: Mapping[str, str] | None = None,
    ) -> int:
        """
        Add raw documents for a company. Returns the number of vector chunks stored.
        """
        metadata = metadata or {}
        timestamp = time.strftime("%Y%m%d%H%M%S")
        collection = self._get_collection(company_name)
        slug = self._slugify(company_name)

        total_chunks = 0
        for index, text in enumerate(contents):
            if not text or not text.strip():
                continue
            normalized_text = text.strip()
            doc_hash = hashlib.md5(normalized_text.encode("utf-8")).hexdigest()
            raw_path = (
                self.corpus_directory / slug / f"{timestamp}_{index}_{doc_hash[:8]}.json"
            )
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_payload = {
                "company_name": company_name,
                "content": normalized_text,
                "metadata": dict(metadata),
                "stored_at": timestamp,
            }
            raw_path.write_text(json.dumps(raw_payload, ensure_ascii=False, indent=2), encoding="utf-8")

            chunks = self._text_splitter.split_text(normalized_text)
            if not chunks:
                continue

            chunk_ids = [f"{slug}_{doc_hash}_{i}" for i in range(len(chunks))]
            chunk_metadatas = [
                dict(metadata) | {"chunk_index": i, "doc_hash": doc_hash, "raw_path": str(raw_path)}
                for i in range(len(chunks))
            ]
            # Upsert so reruns refresh the vector store without duplication errors.
            collection.upsert(ids=chunk_ids, documents=chunks, metadatas=chunk_metadatas)
            total_chunks += len(chunks)

            self._prune_raw_docs(slug)

        return total_chunks

    def query(self, company_name: str, query: str, top_k: int = 5) -> list[dict[str, str]]:
        """
        Retrieve top matching chunks for a company.
        Returns a list of dictionaries containing chunk content and metadata.
        """
        if top_k <= 0:
            return []
        collection = self._get_collection(company_name)
        if collection.count() == 0:
            return []
        results = collection.query(query_texts=[query], n_results=top_k)
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        return [
            {
                "content": doc,
                "metadata": meta or {},
            }
            for doc, meta in zip(documents, metadatas)
        ]

    def get_playbook(self, company_name: str) -> str:
        path = self._playbook_path(company_name)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return ""

    def update_playbook(
        self,
        *,
        company_name: str,
        location_hint: str,
        source: str,
        category: str,
        curated_entry: str,
        curator,
    ) -> None:
        entry = (curated_entry or "").strip()
        if not entry:
            return

        path = self._playbook_path(company_name)
        existing = self.get_playbook(company_name)
        if existing:
            self._archive_playbook(path, existing)
        updated = curator.update_playbook(
            company_name=company_name,
            location_hint=location_hint,
            source=source,
            category=category,
            existing_playbook=existing,
            new_entry=entry,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(updated, encoding="utf-8")

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _playbook_path(self, company_name: str) -> Path:
        return self.playbook_directory / f"{self._slugify(company_name)}.md"

    def _prune_raw_docs(self, slug: str) -> None:
        directory = self.corpus_directory / slug
        if not directory.exists():
            return
        files = sorted(directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for obsolete in files[self.max_raw_files :]:
            try:
                obsolete.unlink()
            except OSError:
                continue

    def _archive_playbook(self, path: Path, content: str) -> None:
        slug = path.stem
        archive_dir = self.playbook_archive_directory / slug
        archive_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d%H%M%S")
        archive_path = archive_dir / f"{timestamp}.md"
        archive_path.write_text(content, encoding="utf-8")
