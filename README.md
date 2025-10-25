## 企业尽调智能体（中文说明）

> 一个聚焦“企业背景调查 / 背调 Playbook”的多智能体系统：自动搜索 + RAG 精炼 + Playbook 管理 + 成本监控。

---

### 1. 架构概览

```
┌─────────────┐        ┌─────────────┐
│ Manager      │───────▶│ Critic      │
│ (gpt-4o)     │        │ (Claude Haiku)│
└──────┬──────┘        └──────┬──────┘
       │                      │
       ▼                      │
┌─────────────┐               │
│ SearchAgent │───────────────┘
│ (Qwen2.5-72B)│  调用浏览器、搜索、RAG 工具
└──────┬──────┘
       ▼
┌───────────────────────┐
│ RAG + Playbook        │ ◀─ 自动整理器 (gpt-4o-mini)
│ ・Chroma 向量库        │    记录每次入库的成本
│ ・Playbook + 归档      │
└───────────────────────┘
```

- **Manager**：解析 `company_template`，规划任务、汇总最终报告。
- **SearchAgent**：先读 Playbook，再按需联网；所有抓取内容都会进入整理管线。
- **Critic**：检查覆盖度、可信度、Playbook 更新情况。
- **RAG Curator**：将网页/搜索结果压缩成 200 词内要点，写入向量库 & Playbook，自动记录成本。
- **Playbook**：`rag_playbooks/` 下保存当前版本；`rag_playbooks/archive/` 保存历史快照（可浏览/清理）。

---

### 2. 快速开始

1. 安装依赖
   ```bash
   pip install -e .
   ```
2. 配置模型（可按需调整，缺省时会回退到 CLI 指定模型）
   ```bash
   export MANAGER_MODEL_ID=gpt-4o
   export MANAGER_API_KEY=...

   export SEARCH_MODEL_ID=fireworks_ai/qwen-2.5-72b-instruct
   export FIREWORKS_API_KEY=...

   export CRITIC_MODEL_ID=anthropic/claude-3-haiku-20240307
   export CRITIC_API_KEY=...

   export CURATOR_MODEL_ID=gpt-4o-mini
   export CURATOR_API_KEY=...

   export SERPAPI_API_KEY=<可选，用于 Google 搜索>
   ```
   若未设置 `SERPAPI_API_KEY`，系统会自动切换到 `ddgs`（DuckDuckGo）检索。

3. 运行命令行
   ```bash
   python run.py --company-name "某某科技" --company-site "上海" --time-window-months 24
   ```

4. Gradio 前端
   ```bash
   python app.py
   ```
   每次提交都会创建全新 agent，避免不同公司之间的状态污染。

---

### 3. 关键目录

| 路径 | 功能 |
| --- | --- |
| `scripts/agent_factory.py` | 统一创建多模型 agent（manager/search/critic/curator）并自动缓存实例。 |
| `scripts/company_rag_store.py` | 向量库 + 原始快照 + Playbook + 归档管理。 |
| `scripts/rag_curator.py` | 整理器（对内容去重/压缩），并配合 `metrics/curation_log.jsonl` 记录成本。 |
| `scripts/rag_tools.py` | `company_rag_retrieve`（Playbook + 原始片段），`company_rag_ingest`（手动整理入库）。 |
| `scripts/playbook_manager.py` | Playbook 浏览/清理脚本（见下方）。 |
| `scripts/curation_monitor.py` | 整理流程的频率/字符量监控。 |
| `rag_playbooks/` | 最新 Playbook；`archive/` 保存历史版本。 |
| `rag_vector_db/` | Chroma 向量库数据。 |
| `rag_corpus/` | 原始抓取快照（自动裁剪保留最近 N 个）。 |

---

### 4. Playbook 浏览与清理

```bash
# 列出所有 Playbook
python scripts/playbook_manager.py list

# 查看某家公司当前/历史版本
python scripts/playbook_manager.py show --company "示例公司"
python scripts/playbook_manager.py show --company "示例公司" --version 20250101120000

# 清理单个公司的归档
python scripts/playbook_manager.py prune --company "示例公司" --keep 5

# 批量清理所有公司
python scripts/playbook_manager.py prune-all --keep 10
```

- 修改 `keep` 可以控制归档数量。
- 最新 Playbook 写在 `rag_playbooks/<slug>.md`，归档存放在 `rag_playbooks/archive/<slug>/timestamp.md`。

---

### 5. 整理成本监控

整理器每次运行都会记录输入/输出字符数，写入 `metrics/curation_log.jsonl`。可用脚本查看：

```bash
# 查看过去 24 小时
python scripts/curation_monitor.py --window-hours 24

# 结果示例
{
  "window_seconds": 86400,
  "events": 12,
  "input_chars": 56000,
  "output_chars": 8400,
  "per_company": {
    "示例公司": {"events": 8, "input_chars": 43000, "output_chars": 6200},
    "...": {...}
  }
}
```

根据统计可判断整理是否过于频繁，必要时可更换更廉价模型或调整 `curate_for_rag` 的长度阈值。

---

### 6. 工作流程

1. `run.py` 根据 CLI/输入生成 `company_template` 背调提示。
2. manager 使用 `gpt-4o`，按模板拆解任务，调度 search agent、critic。
3. search agent **首先** 调用 `company_rag_retrieve` 阅读 Playbook；如需补充，再调用浏览器/Search 工具。
4. 任何新抓取内容都会经过整理器 → 写入向量库、Playbook，并记入成本日志。
5. critic 在每个阶段检查 Playbook 覆盖度并给出整改建议。
6. manager 仅在高优缺口得到解释/补充时才输出最终报告。

---

### 7. 常见问题

- **Playbook 为什么会“变少”？**  
  现在更新前会先归档旧版本，你可以用 `scripts/playbook_manager.py show --version ...` 找回历史版本。

- **没有设置某个模型的 API key 会怎样？**  
  `_build_remote_model` 会回退到 CLI 模型或 manager 模型；若连它们都没有，就会抛错，需要补充环境变量。

- **SERP API key 不可用会导致失败吗？**  
  不会，`CachingGoogleSearchTool` 会自动用 DuckDuckGo (`ddgs`) 兜底，不过结果可能不如 Google 精准。

- **如何限制整理成本？**  
  通过 `scripts/curation_monitor.py` 观察调用频率，必要时更换 `CURATOR_MODEL_ID` 为更小的模型，或者增大 `curate_for_rag` 的最小长度，减少短文本整理。

---

### 8. 贡献与扩展

- 所有关键逻辑都集中在 `scripts/`，可按需更换模型、工具或 Playbook 策略。
- 欢迎根据自身场景扩展更多 Playbook 管理脚本或监控指标。
- 提交 PR 之前建议运行 `python -m compileall run.py scripts/*.py`，确保语法无误。

祝调查顺利，善用 Playbook 和监控工具！ ✨

---

### 9. 微信小程序接入（WeChat Mini Program）

- 新增 `wechat_miniapp.py`（FastAPI）作为微信小程序的后端，复用 agent pipeline 并内置计费策略。
- 依赖：`fastapi`、`uvicorn[standard]`，已经写入 `requirements.txt`。
- 环境变量：
  - `WECHAT_APPID` / `WECHAT_SECRET`：调用 `jscode2session` 所需。
  - `WECHAT_SESSION_URL`（可选）：自定义微信接口地址。
  - `WECHAT_TEST_MODE=true`：本地调试时跳过微信 API，按 code 生成伪 openid。
- 启动：
  ```bash
  uvicorn wechat_miniapp:app --host 0.0.0.0 --port 8080
  ```
- 关键接口：
  | Method | Path | 说明 |
  | --- | --- | --- |
  | `GET` | `/wechat/pricing` | 返回套餐与附加项，用于前端渲染价格卡片。 |
  | `POST` | `/wechat/login` | 代理 `wx.login` code → openid。测试模式下本地生成。 |
  | `POST` | `/wechat/orders` | 创建背调订单，校验套餐/附加项并异步触发 agent。 |
  | `GET` | `/wechat/orders/{order_id}` | 轮询订单状态，返回 prompt/answer/error。 |
- 计费策略：
  - 标准版 ¥99：24 个月公开信息 + Playbook 快照。
  - 深度版 ¥199：多轮主题分析 + 归档与 RAG 导出。
  - 专业版 ¥299：定制检索关键词、批量导出、API 回调。
  - 附加项：Playbook 归档导出 +¥20；自定义时效（12/36 个月）+¥30；人工复核 +¥100。

微信小程序端只需按上述接口提交公司信息、套餐及附加项选择，即可获取实时价格与订单状态。
