from __future__ import annotations

from typing import Any

from smolagents import Tool

from scripts.company_rag_store import CompanyRAGStore


class CompanyRAGRetrieveTool(Tool):
    name = "company_rag_retrieve"
    description = (
        "读取本地为企业维护的 Playbook，总结近期整理的关键事实、风险与机会。"
        "在联网搜索前先调用此工具，可以减少重复抓取。"
    )
    inputs = {
        "company_name": {
            "type": "string",
            "description": "企业名称，用于定位对应的本地向量库。",
        },
    }
    output_type = "string"

    def __init__(self, rag_store: CompanyRAGStore, company_name_resolver=None):
        super().__init__()
        self._store = rag_store
        self._company_name_resolver = company_name_resolver

    def forward(self, company_name: str) -> str:
        target_company = (company_name or "").strip()
        if not target_company and callable(self._company_name_resolver):
            target_company = self._company_name_resolver() or ""
        if not target_company:
            return "company_name 不能为空。"
        playbook = self._store.get_playbook(target_company)
        if playbook.strip():
            return playbook
        return "Playbook 为空，建议先执行搜索或查看官网资料后再试。"


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
        metadata: dict[str, Any] = {}
        if source:
            metadata["source"] = source
        if category:
            metadata["category"] = category
        stored_chunks = self._store.add_documents(
            company_name=target_company,
            contents=[content],
            metadata=metadata,
        )
        location_hint = ""
        if callable(self._location_hint_resolver):
            location_hint = self._location_hint_resolver() or ""
        if self._curator is not None:
            self._store.update_playbook(
                company_name=target_company,
                location_hint=location_hint or "",
                source=source or "manual_ingest",
                category=category or "manual",
                curated_entry=content,
                curator=self._curator,
            )
        return f"已入库，并生成 {stored_chunks} 个向量分片，Playbook 已更新。"
