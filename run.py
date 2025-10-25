import argparse
import os
import threading

from dotenv import load_dotenv
from huggingface_hub import login
from scripts.agent_factory import create_agent
from jinja2 import StrictUndefined, Template
from typing import TYPE_CHECKING, Any, Literal, Type, TypeAlias, TypedDict, Union

def populate_template(template: str, variables: dict[str, Any]) -> str:
    compiled_template = Template(template, undefined=StrictUndefined)
    try:
        return compiled_template.render(**variables)
    except Exception as e:
        raise Exception(f"Error during jinja template rendering: {type(e).__name__}: {e}")

company_template = """
请以公开渠道与主流媒体为核心，为“{{company_name}}”撰写企业背景调查报告（司法管辖提示：{{jurisdiction_hint | default('CN')}}，重点关注最近 {{time_window_months | default(24)}} 个月内的动态，并补充更早但仍影响判断的事件），使用 {{report_language | default('中文')}} 输出。

交付结构
一、任务概述与信息范围
- 说明本次调研目标、重点问题、覆盖地域与排除项，列出关注的核心法人实体、品牌或业务条线。

二、信息来源总览
- 汇总已检索与计划检索的公开渠道，至少涵盖：公司官网与公告、当地政府及监管数据库（工商、证监、交易所、税务、环保、安全生产等）、主流与行业媒体、地方新闻站点、权威商业资讯平台、法院及执行公告、知识产权/商标专利库、信用评级或商业数据服务、社交媒体与口碑平台、消费者投诉网站。为每条来源记录：渠道名称、访问链接或检索方式、最新抓取时间、可获取内容类型（新闻/公告/数据等）、访问限制（公开/付费/需登陆）。

三、企业概况速览
- 概述企业历史沿革、核心业务与产品组合、主营市场、管理层及股权结构要点，引用官方或权威公开资料。若存在同名公司或关联主体，需标注差异。

四、近期战略重点
- 结合官网动态、官方公告、主流媒体与地方媒体报道，总结企业在 {{time_window_months | default(24)}} 个月内的主要动作（如产线扩张、市场进入、合作伙伴、技术发布、投融资等），并指出报道来源、发布时间及其对业务的潜在影响。

五、合规与监管动向
- 汇总监管公告、政府公示、司法或行政处罚、环保/安全检查、税务信息等公开记录；说明事件时间、主管机构、处理状态与对企业运营或外部合作的影响。

六、企业舆情雷达
- 采集主流媒体、地方新闻门户、行业评论、社交/视频平台（微博、微信公号、抖音、B 站、知乎、Glassdoor 等）以及消费者投诉渠道对企业的讨论热度、情绪倾向、核心议题与关键词演变；区分官方声明、第三方报道与匿名信息，标注发布时间、传播渠道、互动量或传播级别（如阅读量/转发/点赞）。
- 对重大舆情事件提供时间轴，说明事件触发点、企业回应、外部反馈与当前状态；如存在谣言或未经证实的消息，要指出核实状态与建议的监测动作。

七、风险与机会评估
- 基于上述信息，对企业近期的核心风险点（如经营、合规、声誉、财务）与亮点机会逐项分析；说明来源支持、影响程度（【高】/【中】/【低】）以及可能的后续观察指标。

八、信息缺口与后续建议
- 指出尚未获取但对判断关键的资料，说明原因（如付费、权限限制、需访谈），并建议下一步验证路径或需要关注的渠道。

附录：资料清单
- 按引用顺序列出所有来源（名称、URL/访问路径、发布日期或检索日期、来源类型、可信度评估）。如引用二手渠道或推断，请明确标注并说明依据。
"""

DEFAULT_COMPANY_VARIABLES: dict[str, Any] = {
    "company_name": "华为技术有限公司",
    "jurisdiction_hint": "CN",
    "time_window_months": 64,
    "report_language": "中文",
    "company_site": "深圳市",
}


def build_company_variables(
    company_name: str | None = None,
    jurisdiction_hint: str | None = None,
    time_window_months: int | None = None,
    report_language: str | None = None,
    company_site: str | None = None,
) -> dict[str, Any]:
    variables = DEFAULT_COMPANY_VARIABLES.copy()
    updates = {
        "company_name": company_name,
        "jurisdiction_hint": jurisdiction_hint,
        "time_window_months": time_window_months,
        "report_language": report_language,
        "company_site": company_site,
    }
    for key, value in updates.items():
        if value is not None:
            variables[key] = value
    return variables


def build_company_prompt(variables: dict[str, Any]) -> str:
    return populate_template(company_template, variables=variables)


load_dotenv(override=True)
login(os.getenv("HF_TOKEN"))

append_answer_lock = threading.Lock()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--company-name",
        type=str,
        help="Company name to inject into the default background check template.",
        default=None,
    )
    parser.add_argument(
        "--jurisdiction-hint",
        type=str,
        help="Jurisdiction hint (e.g. CN, US) for the template.",
        default="CN",
    )
    parser.add_argument(
        "--time-window-months",
        type=int,
        help="How many months of company history to review in the template.",
        default=64,
    )
    parser.add_argument(
        "--report-language",
        type=str,
        help="Language to request for the generated report.",
        default="Chinese",
    )
    parser.add_argument(
        "--company-site",
        type=str,
        help="City or region to emphasise when looking up company information.",
        default=None,
    )
    parser.add_argument(
        "--search_max_steps",
        type=int,
        default=50,
        help="The maximum number of steps the agent can take",
    )
    parser.add_argument(
        "--critic_max_steps",
        type=int,
        default=50,
        help="The maximum number of steps the agent can take",
    )
    parser.add_argument(
        "--manage_max_steps",
        type=int,
        default=100,
        help="The maximum number of steps the agent can take",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    company_context = build_company_variables(
        company_name=args.company_name,
        jurisdiction_hint=args.jurisdiction_hint,
        time_window_months=args.time_window_months,
        report_language=args.report_language,
        company_site=args.company_site,
    )

    agent = create_agent(
        search_max_steps=args.search_max_steps,
        critic_max_steps=args.critic_max_steps,
        manage_max_steps=args.manage_max_steps,
        company_context=company_context,
    )

    task_prompt = build_company_prompt(company_context)
    answer = agent.run(task_prompt)

    print(f"Got this answer: {answer}")


if __name__ == "__main__":
    main()
