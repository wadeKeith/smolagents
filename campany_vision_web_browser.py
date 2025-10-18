import argparse
from io import BytesIO
from time import sleep

import helium
import importlib
import yaml
import PIL.Image
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

from smolagents import CodeAgent, WebSearchTool, tool
from smolagents.agents import ActionStep
from smolagents.cli import load_model
from jinja2 import StrictUndefined, Template
from typing import TYPE_CHECKING, Any, Literal, Type, TypeAlias, TypedDict, Union

def populate_template(template: str, variables: dict[str, Any]) -> str:
    compiled_template = Template(template, undefined=StrictUndefined)
    try:
        return compiled_template.render(**variables)
    except Exception as e:
        raise Exception(f"Error during jinja template rendering: {type(e).__name__}: {e}")

helium_template = """
# Helium Navigation Protocol
- Drive every web interaction with Helium; do not rely on raw HTTP scraping.
- The `web_search` tool is optional for quick SERP previews, but every piece of evidence must ultimately be visited with Helium.
- A Chrome session with `from helium import *` is already active; issue automation instructions via code.

## Core Motions
<code>
go_to("https://www.google.com")
write("ACME Corp business registry")
press(ENTER)
</code>
- After submitting a query pause if needed: `import time; time.sleep(2)`.
- Use `click("Visible text")` or `click(Link("Anchor text"))` to open results; check the new screenshot before acting again.
- Scroll with `scroll_down(num_pixels=900)` / `scroll_up(num_pixels=700)` until the relevant field is captured.
- Dismiss overlays with `close_popups()` instead of trying to target the close icon.
- `search_item_ctrl_f("keyword")` jumps to rendered text; use it before manual scrolling loops.
- `go_back()` returns to the previous SERP when a lead is unproductive.

## Evidence Discipline
- Screenshots and console prints are the audit trail. Log evidence immediately after revealing it:
```
from datetime import datetime, timezone, timedelta
tokyo_now = datetime.now(timezone(timedelta(hours=9)))
print("EVIDENCE | url=", get_driver().current_url, "| section=Registry filing")
print("FIELD | paid_in_capital=CNY 10,000,000 | source=", get_driver().current_url, "| captured_at=", tokyo_now.date().isoformat())
```
- When information is missing or blocked:
<code>
print("SOURCE_INACCESSIBLE | url=", get_driver().current_url, "| reason=login_or_captcha")
</code>
- Avoid duplicative logging; once a fact is captured move on to the next gap.

## Research Priorities
- Use search engines (Google, Bing, 百度) to land on: official registries, regulator announcements, court/case systems, sanction/watchlist databases, credit agencies, and well-regarded media.
- Typical query structures: `"[company] 统一社会信用代码"`, `"[company] administrative penalty"`, `"[company] site:sec.gov"`, `"[company] litigation"`, `"[company] layoffs reddit"`, `"[company] glassdoor reviews"`.
- Prefer primary sources for hard facts (legal representative, registration ID, registered & paid-in capital). Use third-party aggregators only for leads and corroborate them.
- For social sentiment, gather quotes from employment communities (Glassdoor, Indeed, Blind, Reddit, 脉脉, 知乎, 微博) and summarise themes; anonymize personal identifiers.

## Conduct
- Do not log in, bypass paywalls, or interact with CAPTCHAs.
- Respect page performance; if a site is slow, wait or pursue an alternative official source.
- Keep navigation deliberate: close irrelevant tabs, avoid endless scrolling, and stop once required coverage is achieved.

Continue iterating Thought → Code → Observation until you have enough verified evidence, then compile the JSON + Markdown deliverables and call `final_answer(...)`.
"""


'''
## Inputs
- company_name: "{{company_name}}"
- jurisdiction_hint (optional): "{{jurisdiction_hint}}"   # e.g., "CN", "SG", "US"; may be empty or unknown
- time_window_months (optional): {{time_window_months | default(24)}}  # default 24 months for risk/sentiment recency
- report_language (optional): "{{report_language | default('English')}}"
'''

campany_template = """
You are an investigative code agent building a job-seeker oriented company background dossier on "{{company_name}}".

### Inputs
- company_name: "{{company_name}}"
- jurisdiction_hint: "{{jurisdiction_hint | default('unknown')}}"
- time_window_months: {{time_window_months | default(24)}}
- report_language: "{{report_language | default('English')}}"

### Objectives
1. Resolve the legal entity/entities behind the brand, distinguishing parents, subsidiaries, and homonyms.
2. Collect the corporate profile and capital structure, explicitly confirming registered capital and **paid-in capital** with authoritative sources.
3. Document ownership, management, shareholders, beneficial owners, subsidiaries, and financing activity relevant to employment prospects.
4. Surface legal/regulatory risks (litigation, penalties, sanctions, enforcement, ESG/data/privacy cases) with emphasis on the last {{time_window_months | default(24)}} months.
5. Summarize social sentiment from job/community platforms, highlight themes, sentiment balance, and anonymized representative quotes.
6. Produce transparent deliverables (`report_json`, `report_markdown`) for job seekers, including evidence trail, gaps, and recommended follow-ups.

### Research Checklist
- Entity resolution: enumerate candidates, note jurisdictions, registration IDs, and rationale for the primary selection.
- Basic profile: legal/registered name, legal representative, registration number, establishment date, company type, industry, registered & operating addresses, website/domain, employee counts, short business description.
- Capital & finance: registered capital, paid-in/contributed capital, currency, effective date, verification status, historical changes, recent financing rounds.
- Ownership & people: shareholders with stakes, beneficial owners/ultimate controllers (with confidence levels), key executives/directors, subsidiaries/joint ventures.
- Risk coverage: litigations/arbitrations, judgments, liens, administrative penalties, regulatory actions, watchlists/sanctions, major negative media, product/ESG/data incidents, business abnormalities. Capture dates, authorities, case IDs, monetary amounts, outcomes, and source URLs.
- Social sentiment: identify platforms, topics, sentiment split, anonymized quotes (2-6) with links, credibility notes, and confidence.
- Metadata: record `captured_at` dates (YYYY-MM-DD, Asia/Tokyo), `status` fields (`verified`, `not_found`, `inaccessible`, `pending_review`), and evidence logs for every material claim.

### Output Contract
Return a single `final_answer({"report_json": ..., "report_markdown": ...})` once coverage is complete.
- `report_json` must be JSON-serializable with keys:
  - `query`: include `input_name`, `jurisdiction_hint`, `time_window_months`, `report_language`, `generated_at`, `timezone`.
  - `entity_resolution`: `candidates` (list of dicts with name, jurisdiction, registration_id, relationship, confidence, notes, source_url) and `selected` (dict with legal_name, jurisdiction, registration_id, confidence, rationale, source_url).
  - `basic_profile`: dictionaries for core facts; each dictionary should contain `value`, `status`, `source_url`, `captured_at`, and optional `notes`.
  - `capital_structure`: registered capital, paid-in capital, currency, last_verified, notes (same dictionary pattern).
  - `ownership_and_people`: lists for shareholders, beneficial owners, key people, subsidiaries/investments with stake/role, status, source_url, captured_at.
  - `legal_and_compliance`: lists per risk category with fields (type, title, date, authority, amount, status, description, source_url, captured_at).
  - `social_reputation`: sentiment summary, topic clusters, representative quotes (each quote with text, platform, sentiment, source_url, captured_at, anonymized attribution), bias_notes, confidence.
  - `sources`: deduplicated list of evidence dicts `{title, url, section, captured_at, type}`.
  - `gaps_and_followups`: list of unresolved items with `item`, `status`, and `attempted_sources`.
- `report_markdown` must be written in {{report_language | default('English')}} and include sections: Overview, Entity Resolution, Corporate Profile, Capital & Ownership, Legal & Compliance Highlights, Social Reputation Snapshot, Sources & Evidence (with links and capture dates), Gaps & Next Steps, Disclaimer.

### Evidence & Methodology
- Use Helium-driven Chrome for all browsing; supplemental tools are allowed only for SERP previews.
- Log evidence with `print()` immediately after reading it on-screen; make sure the field is visible in the latest screenshot.
- Prioritize official/primary sources in the relevant jurisdiction; corroborate aggregator data before trusting it.
- Cover the most recent {{time_window_months | default(24)}} months for dynamic risks, while retaining older events if they remain critical context.
- Clearly document uncertainties, contradictions, and inaccessible sources (list the URLs you tried).

Do not fabricate or guess values. If information cannot be verified, mark the field status appropriately and record the research attempts.
"""

company_request = populate_template(
            campany_template,
            variables={
                "company_name": "墨小孔（杭州）机器人科技有限公司",
                "jurisdiction_hint": "CN",
                "time_window_months": 24,
                "report_language": "Chinese",
            },
        )


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run a web browser automation script with a specified model.")
    parser.add_argument(
        "prompt",
        type=str,
        nargs="?",  # Makes it optional
        default=company_request,
        help="The prompt to run with the agent",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="OpenAIServerModel",
        help="The model type to use (e.g., OpenAIServerModel, LiteLLMModel, TransformersModel, InferenceClientModel, VLLMModel)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="gpt-4o",
        help="The model ID to use for the specified model type",
    )
    parser.add_argument(
        "--provider",
        type=str,
        # default='novita',
        help="The inference provider to use for the model",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default="https://aihubmix.com/v1",
        help="The API base to use for the model",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="sk-WByFqYOyJ7daEzr081E864B7Fb4144Fb91C3038dC61aBc92",
        help="The API key to use for the model",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=100,
        help="The maximum number of steps the agent can take",
    )
    return parser.parse_args()


def save_screenshot(memory_step: ActionStep, agent: CodeAgent) -> None:
    sleep(1.0)  # Let JavaScript animations happen before taking the screenshot
    driver = helium.get_driver()
    current_step = memory_step.step_number
    if driver is not None:
        for previous_memory_step in agent.memory.steps:  # Remove previous screenshots from logs for lean processing
            if isinstance(previous_memory_step, ActionStep) and previous_memory_step.step_number <= current_step - 2:
                previous_memory_step.observations_images = None
        png_bytes = driver.get_screenshot_as_png()
        image = PIL.Image.open(BytesIO(png_bytes))
        print(f"Captured a browser screenshot: {image.size} pixels")
        memory_step.observations_images = [image.copy()]  # Create a copy to ensure it persists, important!

    # Update observations with current URL
    url_info = f"Current url: {driver.current_url}"
    memory_step.observations = (
        url_info if memory_step.observations is None else memory_step.observations + "\n" + url_info
    )
    return


def _escape_xpath_string(s: str) -> str:
    """
    Escapes a string for safe use in an XPath expression.

    Args:
        s (`str`): Arbitrary input string to escape.

    Returns:
        `str`: Valid XPath expression representing the literal value of `s`.
    """
    if "'" not in s:
        return f"'{s}'"
    if '"' not in s:
        return f'"{s}"'
    parts = s.split("'")
    return "concat(" + ', "\'", '.join(f"'{p}'" for p in parts) + ")"


@tool
def search_item_ctrl_f(text: str, nth_result: int = 1) -> str:
    """
    Searches for text on the current page via Ctrl + F and jumps to the nth occurrence.
    Args:
        text: The text to search for
        nth_result: Which occurrence to jump to (default: 1)
    """
    escaped_text = _escape_xpath_string(text)
    elements = driver.find_elements(By.XPATH, f"//*[contains(text(), {escaped_text})]")
    if nth_result > len(elements):
        raise Exception(f"Match n°{nth_result} not found (only {len(elements)} matches found)")
    result = f"Found {len(elements)} matches for '{text}'."
    elem = elements[nth_result - 1]
    driver.execute_script("arguments[0].scrollIntoView(true);", elem)
    result += f"Focused on element {nth_result} of {len(elements)}"
    return result


@tool
def go_back() -> None:
    """Goes back to previous page."""
    driver.back()


@tool
def close_popups() -> str:
    """
    Closes any visible modal or pop-up on the page. Use this to dismiss pop-up windows! This does not work on cookie consent banners.
    """
    webdriver.ActionChains(driver).send_keys(Keys.ESCAPE).perform()


def initialize_driver():
    """Initialize the Selenium WebDriver."""
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--force-device-scale-factor=1")
    chrome_options.add_argument("--window-size=1000,1350")
    chrome_options.add_argument("--disable-pdf-viewer")
    chrome_options.add_argument("--window-position=0,0")
    return helium.start_chrome(headless=False, options=chrome_options)


def initialize_agent(model, max_steps):
    """Initialize the CodeAgent with the specified model."""
    prompt = yaml.safe_load(
                importlib.resources.files("smolagents.prompts").joinpath("company_code_agent.yaml").read_text()
            )
    return CodeAgent(
        tools=[WebSearchTool(), go_back, close_popups, search_item_ctrl_f],
        model=model,
        prompt_templates = prompt,
        additional_authorized_imports=["helium"],
        step_callbacks=[save_screenshot],
        max_steps=max_steps,
        verbosity_level=2,
    )


def run_webagent(
    prompt: str,
    model_type: str,
    model_id: str,
    provider: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    max_steps: int = 20,
) -> None:
    # Load environment variables
    load_dotenv()

    # Initialize the model based on the provided arguments
    model = load_model(model_type, model_id, provider=provider, api_base=api_base, api_key=api_key)

    global driver
    driver = initialize_driver()
    agent = initialize_agent(model, max_steps)

    # Run the agent with the provided prompt
    agent.python_executor("from helium import *")
    agent.run(prompt + helium_template, max_steps=max_steps)


def main() -> None:
    # Parse command line arguments
    args = parse_arguments()
    run_webagent(args.prompt, args.model_type, args.model_id, args.provider, args.api_base, args.api_key, args.max_steps)


if __name__ == "__main__":
    main()
