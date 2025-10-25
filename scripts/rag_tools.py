from __future__ import annotations

from typing import Any

from smolagents import Tool

from scripts.company_rag_store import CompanyRAGStore


class CompanyRAGRetrieveTool(Tool):
    name = "company_rag_retrieve"
    description = (
        "读取本地为企业维护的 Playbook，并附带若干原始向量片段，帮助快速回顾核心事实。"
        "在联网搜索前先调用此工具，可以减少重复抓取。"
    )
    inputs = {
        "company_name": {
            "type": "string",
            "description": "企业名称，用于定位对应的本地向量库。",
        },
        "query": {
            "type": "string",
            "description": "可选的检索关键词，若省略则自动根据 Playbook 内容匹配。",
            "nullable": True,
        },
        "top_k": {
            "type": "integer",
            "description": "附带返回的原始片段数量，默认 3。",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, rag_store: CompanyRAGStore, company_name_resolver=None):
        super().__init__()
        self._store = rag_store
        self._company_name_resolver = company_name_resolver

    def forward(self, company_name: str, query: str | None = None, top_k: int | None = None) -> str:
        target_company = (company_name or "").strip()
        if not target_company and callable(self._company_name_resolver):
            target_company = self._company_name_resolver() or ""
        if not target_company:
            return "company_name 不能为空。"
        playbook = self._store.get_playbook(target_company).strip()
        sections: list[str] = []
        if playbook:
            sections.append("## Playbook\n" + playbook)
        query_text = (query or "").strip()
        if not query_text:
            query_text = playbook[:400] or target_company
        raw_chunks = self._store.query(target_company, query_text, top_k=top_k or 3)
        if raw_chunks:
            formatted = []
            for idx, chunk in enumerate(raw_chunks, start=1):
                meta = chunk.get("metadata", {})
                formatted.append(
                    f"[片段 {idx}] 来源: {meta.get('source', meta.get('raw_path', '未知'))}\n{chunk.get('content', '').strip()}"
                )
            sections.append("## 原始参考片段\n" + "\n\n".join(formatted))
        if not sections:
            return "Playbook 为空，建议先执行搜索或查看官网资料后再试。"
        return "\n\n---\n\n".join(sections)


class CompanyRAGIngestTool(Tool):
    name = "company_rag_ingest"
    description = (
        "将新获取的网页或新闻内容写入本地企业向量数据库，"
        "以便后续无需重复抓取即可快速检索。"
    )
    inputs = {
        "company_name": {
            "type": "string",
            "description": "企业名称，用于确定写入的向量库目录。",
        },
        "content": {
            "type": "string",
            "description": "要写入的文本内容，建议是一段较完整的网页正文或新闻摘要。",
        },
        "source": {
            "type": "string",
            "description": "信息来源（如 URL、媒体名称），便于后续追溯。",
            "nullable": True,
        },
        "category": {
            "type": "string",
            "description": "内容类别（如官网公告、主流媒体、地方媒体、舆情等），用于元数据标注。",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(
        self,
        rag_store: CompanyRAGStore,
        company_name_resolver=None,
        location_hint_resolver=None,
        curator=None,
    ):
        super().__init__()
        self._store = rag_store
        self._company_name_resolver = company_name_resolver
        self._location_hint_resolver = location_hint_resolver
        self._curator = curator

    def forward(self, company_name: str, content: str, source: str | None = None, category: str | None = None) -> str:
        target_company = (company_name or "").strip()
        if not target_company and callable(self._company_name_resolver):
            target_company = self._company_name_resolver() or ""
        if not target_company:
            return "company_name 不能为空。"
        if not content.strip():
            return "content 不能为空。"
        location_hint = ""
        if callable(self._location_hint_resolver):
            location_hint = self._location_hint_resolver() or ""
        query_hint = content[:400]
        existing_chunks = self._store.query(target_company, query_hint, top_k=3)
        existing_context = "\n\n".join(chunk["content"] for chunk in existing_chunks if chunk.get("content"))
        curated_entry = content.strip()
        if self._curator is not None:
            curated_entry = (
                self._curator.curate(
                    company_name=target_company,
                    location_hint=location_hint or "目标地区",
                    source=source or "manual_ingest",
                    category=category or "manual",
                    new_text=content,
                    existing_context=existing_context,
                )
                or ""
            )
        curated_entry = curated_entry.strip()
        if not curated_entry:
            return "未能生成有效的整理内容，已跳过入库。"
        metadata: dict[str, Any] = {}
        if source:
            metadata["source"] = source
        if category:
            metadata["category"] = category
        stored_chunks = self._store.add_documents(
            company_name=target_company,
            contents=[curated_entry],
            metadata=metadata,
        )
        self._store.update_playbook(
            company_name=target_company,
            location_hint=location_hint or "目标地区",
            source=source or "manual_ingest",
            category=category or "manual",
            curated_entry=curated_entry,
            curator=self._curator,
        )
        return f"已入库，并生成 {stored_chunks} 个向量分片，Playbook 已更新。"
