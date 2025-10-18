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

helium_instructions = """
Use your web_search tool to obtain search results. Then use Helium ONLY for navigating specific websites from those results.
Do NOT use Helium to query search engines. Use Helium to open, read, scroll, and capture evidence from webpages.

Helium is already set up:
- We've already run: from helium import *
- You can go to pages with: 
  <code>
  go_to('https://example.com')
  </code>

Clicking:
- For buttons with visible text:
  <code>
  click("Top products")
  </code>
- For links:
  <code>
  click(Link("Top products"))
  </code>
- After a click, STOP and let the new page load (you will get a new screenshot). If the page is slow, you MAY:
  <code>
  import time; time.sleep(3.0)
  </code>
  (Use sparingly.)

Popups and cookies:
- Do NOT try to target an “X” icon. Instead:
  <code>
  close_popups()
  </code>
- If a consent banner text is clearly visible, you may:
  <code>
  if Text('Accept').exists(): click('Accept')
  </code>

Scrolling:
- Bring the relevant section into the viewport before you rely on the screenshot:
  <code>
  scroll_down(num_pixels=1200)
  scroll_up(num_pixels=1200)
  </code>

Finding on page:
- Prefer visually reading the latest screenshot.
- If available in your environment, you may use `search_item_ctrl_f("keyword")` to jump to text.
- Never attempt complex CSS selectors or DOM queries. Interact like a user.

Evidence capture:
- Each action concludes with an **automatic screenshot**. Ensure the target fields (e.g., legal name, registration ID, registered/paid-in capital, penalty details) are VISIBLE in the viewport by scrolling to them.
- Immediately `print()` a one-line EVIDENCE log after landing on key sections, e.g.:
  <code>
  print("EVIDENCE | url=", get_driver().current_url, "| section=ACRA entity profile")
  </code>

Critical do's and don'ts for compliance:
- NEVER attempt to log in. Do not bypass paywalls or CAPTCHAs.
- Respect robots.txt and terms of service.
- If a page is inaccessible (login/captcha/paywall/geo-block), STOP, `print()` a note like:
  <code>
  print("SOURCE_INACCESSIBLE | url=... | reason=captcha_or_login")
  </code>
  and move on to alternative authoritative sources.

Page listing and pagination:
- If long lists are present, scroll modestly (one or two screens) to reveal item summaries, not to exhaustively scrape.
- Prefer targeted entity pages (official profile, penalties, court documents).

Jurisdictional priority (for navigation order after web_search):
1) Official registries/regulators/courts in the entity's jurisdiction (e.g., CN enterprise publicity, ACRA, US state registries, SEC).
2) Mainstream media and government press releases.
3) Reputable third-party aggregators (as leads only).
4) Social/job platforms (sentiment side-evidence).

Social data hygiene:
- When copying quotes from public posts, anonymize handles/names automatically (e.g., “Employee—Glassdoor”).
- Avoid personal data; keep to public, non-sensitive content.

Workflow guidance:
- Proceed in several short steps rather than one huge step. After each navigation or extraction, stop to observe the screenshot and URL.
- When you have enough verified fields, stop browsing and assemble the final JSON + Markdown.
- At the end, only when your answer is ready, call:
  <code>
  final_answer({...})
  </code>
"""


'''
## Inputs
- company_name: "{{company_name}}"
- jurisdiction_hint (optional): "{{jurisdiction_hint}}"   # e.g., "CN", "SG", "US"; may be empty or unknown
- time_window_months (optional): {{time_window_months | default(24)}}  # default 24 months for risk/sentiment recency
- report_language (optional): "{{report_language | default('English')}}"
'''

campany_template = """
You are tasked with performing a **company background check** for job-seekers.

## Inputs
- company_name: "{{company_name}}"
- jurisdiction_hint: "{{jurisdiction_hint | default('US')}}"
- time_window_months: {{time_window_months | default(24)}}
- report_language: "{{report_language | default('English')}}"

## Goals
1) Resolve/Disambiguate the legal entity that best matches `company_name` (use the hint if provided). If multiple plausible entities exist, list 1-5 candidates with rationale and confidence; then select one for the main report and note your decision.
2) Collect verified **basic profile** fields, prioritizing **official/authoritative** sources:
   - legal/registered name, legal representative, registration/credit ID, established date, company type, industry, address, website/domain
   - registered capital and **paid-in (contributed) capital** (with sources and capture dates)
   - ownership: shareholders (with holding % if available), beneficial owners/ultimate controllers (with confidence), subsidiaries
   - employees range or headcount (with source)
3) Identify **legal & compliance risks** in the selected time window:
   - litigations/arbitrations, being-listed as executed/liens, administrative penalties, watchlists/sanctions, IP disputes, serious negative media, recalls, data/privacy penalties, business abnormalities
   - For each risk item: include date, authority/court, status, amount/penalty (if any), and source with capture date.
4) Analyze **social reputation** across job/social platforms for the selected time window:
   - topic clustering (e.g., workload, compensation, management, growth, layoffs, compliance)
   - sentiment distribution (pos/neu/neg) and 2-6 representative anonymized quotes with links where possible
   - note bias/noise and confidence
5) Produce **both**:
   - `report_json` (machine-readable JSON) exactly as defined in the System Prompt (including `generated_at` with Asia/Singapore date)
   - `report_markdown` (concise human-readable summary with sections: Overview, Ownership & Control, Legal/Compliance, Social Sentiment, Sources with links and capture dates, Gaps/Follow-ups, Disclaimer)

## Constraints & Evidence
- Always start with web_search to locate official registry pages and other primary sources.
- Use Helium only for navigation and reading on selected sites; ensure relevant fields are visible in screenshots.
- For **every critical data point**, include a clickable source URL and a `captured_at` date (YYYY-MM-DD, Asia/Singapore).
- If a source is inaccessible (login/captcha/paywall/geo-block), record it as `SOURCE_INACCESSIBLE` with reason and look for alternatives.
- Never fabricate. If not verifiable, mark `status: "not_found"` or `status: "inaccessible"` and include attempted sources.

## Deliverable
Return a single `final_answer({...})` call with:
- `report_json`: JSON-serializable dict (schema per System Prompt)
- `report_markdown`: Markdown string for job-seekers

Begin now. Disambiguate the entity, gather evidence, extract and verify fields, assess risks, analyze social sentiment, and then assemble the outputs exactly as specified.
""" 

company_request = populate_template(
            campany_template,
            variables={
                "company_name": "Apple Inc",
                "jurisdiction_hint": "US",
                "time_window_months": 12,
                "report_language": "English",
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
        default="claude-haiku-4-5",
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
    return helium.start_chrome(headless=True, options=chrome_options)


def initialize_agent(model):
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
        max_steps=20,
        verbosity_level=2,
    )


def run_webagent(
    prompt: str,
    model_type: str,
    model_id: str,
    provider: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
) -> None:
    # Load environment variables
    load_dotenv()

    # Initialize the model based on the provided arguments
    model = load_model(model_type, model_id, provider=provider, api_base=api_base, api_key=api_key)

    global driver
    driver = initialize_driver()
    agent = initialize_agent(model)

    # Run the agent with the provided prompt
    agent.python_executor("from helium import *")
    agent.run(prompt + helium_instructions, max_steps=100)


def main() -> None:
    # Parse command line arguments
    args = parse_arguments()
    run_webagent(args.prompt, args.model_type, args.model_id, args.provider, args.api_base, args.api_key)


if __name__ == "__main__":
    main()
