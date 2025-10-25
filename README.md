## 企业背调智能体框架（中文说明）

> 一个以 Playbook 为核心、支持多模型协同、具备自动整理与监控能力的企业背调应用。

---

### 1. 架构概览

```
┌─────────┐    ┌────────────┐    ┌─────────────┐
│Manager  │───▶│Search Agent│───▶│浏览工具/RAG │
│(CodeAgent)│  └────────────┘    └─────────────┘
│  └Critic──▶ 使用不同模型协作           │
└─────────┘                           │
                                       ▼
                              ┌────────────────┐
                              │Playbook + RAG  │
                              │(Chroma + 本地档)│
                              └────────────────┘
```

- **Manager (默认 gpt-4o)**：负责解析背景调查模版、规划任务、调度子智能体并整合最终报告。
- **Search agent (默认 Fireworks Qwen2.5-72B)**：执行联网检索/浏览任务，自动将新发现交给整理器。
- **Critic agent (默认 Claude 3 Haiku)**：审核阶段性产出，强调 Playbook 覆盖度与质量。
- **RAG Curator (默认 gpt-4o-mini)**：对网页正文/搜索摘要做去噪、去重，写入向量库和 Playbook。
- **Playbook & RAG**：`rag_vector_db/` 保存向量化摘要，`rag_playbooks/` 保存最新版 Playbook，`rag_playbooks/archive/` 保存历史版本，`rag_corpus/` 存放原始快照。

---

### 2. 快速开始

1. **依赖安装**
   ```bash
   pip install -e .
   ```
2. **配置环境变量（示例）**
   ```bash
   export MANAGER_MODEL_ID=gpt-4o
   export MANAGER_API_KEY=sk-xxx
   export SEARCH_MODEL_ID=fireworks_ai/qwen-2.5-72b-instruct
   export SEARCH_API_KEY=fw-xxx
   export CRITIC_MODEL_ID=anthropic/claude-3-haiku-20240307
   export CRITIC_API_KEY=fw-yyy
   export CURATOR_MODEL_ID=gpt-4o-mini
   export CURATOR_API_KEY=sk-zzz
   export SERPAPI_API_KEY=<可选：联网搜索>
   ```
   > 若未设置对应 env，系统会自动回退到 CLI 指定的 `--model-type/--model-id`。

3. **命令行运行**
   ```bash
   python run.py --company-name "示例公司" --company-site "上海"
   ```

4. **Gradio 前端**
   ```bash
   python app.py
   ```
   每次提交都会基于当前表单创建全新 agent，确保不会继承上一家公司的记忆。

---

### 3. 模型/成本监控

- 所有整理调用（RAG Curator）都会写入 `metrics/curation_log.jsonl`，记录时间、公司、来源、输入/输出字符数。  
- 查看最近 24 小时的整理成本：
  ```bash
  python scripts/curation_monitor.py --window-hours 24
  ```
- Google Search 若缺乏 SERP API key，会自动切换到 `ddgs`（DuckDuckGo 搜索）确保流程不中断。

---

### 4. Playbook/归档维护

- **实时写入**：任何网页访问、搜索、文件阅读都会在整理后写入 Playbook。
- **自动归档**：每次更新会将旧版本保存到 `rag_playbooks/archive/<slug>/timestamp.md`。
- **管理脚本**
  ```bash
  # 列出当前所有 Playbook
  python scripts/playbook_manager.py list

  # 查看公司 Playbook / 指定归档版本
  python scripts/playbook_manager.py show --company "示例公司"
  python scripts/playbook_manager.py show --company "示例公司" --version 20250101120000

  # 清理归档
  python scripts/playbook_manager.py prune --company "示例公司" --keep 5
  python scripts/playbook_manager.py prune-all --keep 10
  ```
- **RAG 查询**：`company_rag_retrieve` 现在会返回 “Playbook + 最近原始片段”，当 Playbook 还未覆盖某主题时，agent 仍可回退到原始摘要。

---

### 5. 目录与关键脚本

| 路径 | 功能 |
| --- | --- |
| `scripts/agent_factory.py` | 统一创建多模型智能体：manager/search/critic/curator、RAG 更新、缓存工具等。 |
| `scripts/rag_curator.py` | 背调资料整理器。 |
| `scripts/company_rag_store.py` | Chroma 存储 + Playbook/归档管理。 |
| `scripts/rag_tools.py` | Playbook 检索/手动入库工具。 |
| `scripts/curation_monitor.py` | 整理成本监控脚本。 |
| `scripts/playbook_manager.py` | Playbook 浏览与清理工具。 |
| `metrics/curation_log.jsonl` | 整理调用日志。 |
| `rag_playbooks/` | 最新 Playbook；`archive/` 中存历史版本。 |

---

### 6. 任务流程（与模板）

1. CLI/Gradio 生成公司模板 (`run.py` 的 `company_template`)。
2. Manager 拆解任务、调度 search agent，并强制在每个阶段后由 critic 审核。
3. search agent **先** 调用 `company_rag_retrieve` 读取 Playbook，再决定是否联网。
4. 任意网页/搜索内容 → `curate_for_rag` → RAG + Playbook + 监控日志。
5. critic 检查 Playbook 覆盖度，必要时要求补查。
6. manager 在所有高优缺口被解释/补齐后才输出最终报告。

---

### 7. 常见问题

- **为何 Playbook 内容会“变少”？**  
  现在在更新前会自动将旧版本归档，你可以通过 `scripts/playbook_manager.py show --version` 找回历史版本。

- **如何限制整理成本？**  
  通过 `scripts/curation_monitor.py` 观察调用频率，必要时可以在 `.env` 中设置不同整理模型（如更小的 Qwen/通义）或调整 `curate_for_rag` 长度阈值。

- **SERP API Key 不可用时会怎样？**  
  搜索工具会自动切换到 DuckDuckGo (`ddgs`)；若你需要精准 Google 结果，请在 `.env` 中设置 `SERPAPI_API_KEY`。

---

### 8. 贡献/自定义

- 可通过修改 `scripts/agent_factory.py` 中的 `_build_remote_model` 或 `.env`，自定义任意 agent 的模型/端点。
- 若需要集成新的 RAG 或监控后端，可在 `scripts/company_rag_store.py` 与 `scripts/curation_monitor.py` 扩展。
- 欢迎提交 PR/Issue，或在 README 中列出的脚本基础上继续构建。

---

祝你调查顺利，记得善用 Playbook 与监控脚本！ ✨
