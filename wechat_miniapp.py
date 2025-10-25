"""WeChat mini-program backend for the background-check agent system.

This FastAPI service exposes a small surface area tailored for mini-programs:

1. `/wechat/pricing`      – transparent plan catalog consumed by UI.
2. `/wechat/login`        – thin proxy for `jscode2session` (with local test fallback).
3. `/wechat/orders`       – create a background-check order, computes fees, runs agent.
4. `/wechat/orders/{id}`  – poll order status, retrieve prompt & completion.

Pricing follows the brief:

Single run ¥25 – 24 个月公开数据扫描 + Playbook 快照。

Run locally:
    uvicorn wechat_miniapp:app --host 0.0.0.0 --port 8080

Environment variables:
    WECHAT_APPID, WECHAT_SECRET        – required to call `jscode2session` in prod.
    WECHAT_SESSION_URL (optional)      – override API URL, defaults to the official one.
    WECHAT_TEST_MODE ("true")         – skip upstream call, derive fake openid for dev.
    WECHAT_SEARCH_MAX_STEPS           – override agent search steps (default 30/40/60).
"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, List

import requests
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

from run import build_company_prompt, build_company_variables
from scripts.agent_factory import create_agent


# ── Pricing configuration ---------------------------------------------------

PRICING_PLANS: Dict[str, Dict[str, Any]] = {
    "standard": {
        "name": "单次尽调",
        "price_cny": 25,
        "tagline": "固定 ¥25，一次完成 24 个月尽调",
        "features": [
            "24 个月公开渠道扫描",
            "公司概况 / 合规风险 / 舆情雷达",
            "Playbook 实时快照",
        ],
    }
}


# ── Data models -------------------------------------------------------------


class CompanyPayload(BaseModel):
    company_name: str = Field(..., min_length=2, max_length=120, description="企业名称")
    jurisdiction_hint: str | None = Field(
        default="CN", max_length=8, description="司法管辖区域，如 CN/US/EU"
    )
    time_window_months: int | None = Field(
        default=24, ge=6, le=120, description="时间窗口（月）"
    )
    report_language: str | None = Field(
        default="中文", min_length=2, max_length=20, description="输出语言"
    )
    company_site: str | None = Field(
        default=None, max_length=120, description="重点地区/城市"
    )


class OrderRequest(BaseModel):
    plan_id: str = Field(..., description="唯一支持的方案：standard")
    company: CompanyPayload
    wechat_code: str | None = Field(
        default=None, description="wx.login() code；若已有 openid 可为空"
    )
    openid: str | None = Field(
        default=None, description="前端缓存的 openid，可跳过 code 交换"
    )


class LoginRequest(BaseModel):
    code: str = Field(..., min_length=4, description="wx.login() 返回的临时 code")


class PricingItem(BaseModel):
    id: str
    name: str
    price_cny: int
    description: str | None = None
    features: List[str] | None = None
    tagline: str | None = None


class PricingCatalog(BaseModel):
    plans: List[PricingItem]


class PricingBreakdownItem(BaseModel):
    kind: str
    label: str
    amount_cny: int


class OrderSummary(BaseModel):
    order_id: str
    status: str
    total_cny: int
    breakdown: List[PricingBreakdownItem]
    plan_id: str


class OrderDetail(OrderSummary):
    company: CompanyPayload
    prompt: str | None = None
    answer: str | None = None
    error: str | None = None
    updated_at: float


@dataclass
class OrderRecord:
    summary: OrderSummary
    company_payload: CompanyPayload
    prompt: str | None = None
    answer: str | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    @property
    def status(self) -> str:
        return self.summary.status


order_store: Dict[str, OrderRecord] = {}
order_lock = Lock()


# ── Helper utilities --------------------------------------------------------


def _model_dump(model: BaseModel) -> dict[str, Any]:
    """Compatibility helper for Pydantic v1/v2."""

    dump = getattr(model, "model_dump", None)
    if callable(dump):
        return dump()
    return model.dict()


def _ensure_plan(plan_id: str) -> Dict[str, Any]:
    plan = PRICING_PLANS.get(plan_id)
    if not plan:
        raise HTTPException(status_code=400, detail=f"未知方案：{plan_id}")
    return plan


def _compute_pricing(plan_id: str) -> tuple[int, List[PricingBreakdownItem]]:
    plan = _ensure_plan(plan_id)
    breakdown = [
        PricingBreakdownItem(kind="plan", label=plan["name"], amount_cny=plan["price_cny"])
    ]
    return plan["price_cny"], breakdown


def _fetch_wechat_openid(code: str) -> str:
    if os.getenv("WECHAT_TEST_MODE", "").lower() == "true":
        return f"test_openid_{code[:8]}"

    appid = os.getenv("WECHAT_APPID")
    secret = os.getenv("WECHAT_SECRET")
    if not appid or not secret:
        raise HTTPException(status_code=500, detail="后端未配置 WECHAT_APPID/WECHAT_SECRET")

    url = os.getenv("WECHAT_SESSION_URL", "https://api.weixin.qq.com/sns/jscode2session")
    resp = requests.get(
        url,
        params={
            "appid": appid,
            "secret": secret,
            "js_code": code,
            "grant_type": "authorization_code",
        },
        timeout=10,
    )
    data = resp.json()
    if "errcode" in data and data["errcode"] != 0:
        raise HTTPException(status_code=502, detail=f"微信返回错误：{data}")
    openid = data.get("openid")
    if not openid:
        raise HTTPException(status_code=502, detail="未从微信返回中获取 openid")
    return openid


def _agent_config_for_plan(plan_id: str) -> dict[str, int]:
    if plan_id == "pro":
        return {"search_max_steps": 60, "critic_max_steps": 60, "manage_max_steps": 120}
    if plan_id == "deep":
        return {"search_max_steps": 45, "critic_max_steps": 45, "manage_max_steps": 90}
    return {"search_max_steps": 30, "critic_max_steps": 30, "manage_max_steps": 60}


def _start_agent_run(order_id: str, company_payload: CompanyPayload, plan_id: str) -> None:
    config = _agent_config_for_plan(plan_id)
    variables = build_company_variables(**_model_dump(company_payload))
    prompt = build_company_prompt(variables)
    try:
        agent = create_agent(company_context=variables, **config)
        answer = agent.run(prompt)
    except Exception as exc:  # pragma: no cover - network/model failures handled at runtime
        with order_lock:
            record = order_store[order_id]
            record.error = str(exc)
            record.summary.status = "error"
            record.updated_at = time.time()
        return

    with order_lock:
        record = order_store[order_id]
        record.prompt = prompt
        record.answer = answer
        record.summary.status = "completed"
        record.summary.total_cny = record.summary.total_cny  # no-op, clarity
        record.updated_at = time.time()


# ── FastAPI application -----------------------------------------------------


app = FastAPI(title="WeChat Mini Program Backend", version="0.1.0")


@app.get("/wechat/pricing", response_model=PricingCatalog)
def get_pricing() -> PricingCatalog:
    plans = [
        PricingItem(id=plan_id, name=meta["name"], price_cny=meta["price_cny"], tagline=meta["tagline"], features=meta["features"])
        for plan_id, meta in PRICING_PLANS.items()
    ]
    return PricingCatalog(plans=plans)


@app.post("/wechat/login")
def exchange_code(payload: LoginRequest) -> Dict[str, str]:
    openid = _fetch_wechat_openid(payload.code)
    return {"openid": openid}


@app.post("/wechat/orders", response_model=OrderSummary)
def create_order(request: OrderRequest, background_tasks: BackgroundTasks) -> OrderSummary:
    _ensure_plan(request.plan_id)
    company_payload = CompanyPayload(**_model_dump(request.company))

    total, breakdown = _compute_pricing(request.plan_id)

    openid = request.openid
    if not openid and request.wechat_code:
        openid = _fetch_wechat_openid(request.wechat_code)

    if not openid:
        raise HTTPException(status_code=400, detail="缺少 openid 或微信 code")

    order_id = uuid.uuid4().hex
    summary = OrderSummary(
        order_id=order_id,
        status="pending",
        total_cny=total,
        breakdown=breakdown,
        plan_id=request.plan_id,
    )

    record = OrderRecord(summary=summary, company_payload=company_payload)
    with order_lock:
        order_store[order_id] = record

    background_tasks.add_task(_async_agent_wrapper, order_id, company_payload, request.plan_id)

    return summary


def _async_agent_wrapper(order_id: str, company_payload: CompanyPayload, plan_id: str) -> None:
    with order_lock:
        record = order_store[order_id]
        record.summary.status = "processing"
        record.updated_at = time.time()
    _start_agent_run(order_id, company_payload, plan_id)


@app.get("/wechat/orders/{order_id}", response_model=OrderDetail)
def get_order(order_id: str) -> OrderDetail:
    record = order_store.get(order_id)
    if not record:
        raise HTTPException(status_code=404, detail="订单不存在")

    detail = OrderDetail(
        order_id=record.summary.order_id,
        status=record.summary.status,
        total_cny=record.summary.total_cny,
        breakdown=record.summary.breakdown,
        plan_id=record.summary.plan_id,
        company=record.company_payload,
        prompt=record.prompt,
        answer=record.answer,
        error=record.error,
        updated_at=record.updated_at,
    )
    return detail


__all__ = ["app", "PRICING_PLANS"]
