"""
Waterfall lead scoring: Apollo data → website scrape → DuckDuckGo fallback.
Uses Anthropic Claude 3 Haiku with strict JSON responses.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import re
from typing import TYPE_CHECKING, Any

import anthropic

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

MODEL = "claude-haiku-4-5"
# Claude Haiku 4.5: $1/M input + $5/M output → rough average assuming ~4:1 ratio
DEFAULT_COST_PER_MILLION_TOKENS_USD = 2.0

JSON_INSTRUCTION = """You must respond with a single JSON object only, no markdown fences, no other text.
Keys:
- "status": one of "YES", "NO", "MAYBE" (uppercase)
- "reason": short English explanation (one sentence)

Rules:
- YES: company clearly matches the ICP.
- NO: clearly does not match or is excluded (e.g. agency when ICP excludes agencies).
- MAYBE: insufficient or ambiguous information."""


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("empty model response")
    # Strip accidental markdown fences
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    data = json.loads(text)
    if "status" not in data or "reason" not in data:
        raise ValueError("missing status or reason")
    st = str(data["status"]).upper().strip()
    if st not in ("YES", "NO", "MAYBE"):
        raise ValueError(f"invalid status: {st}")
    data["status"] = st
    data["reason"] = str(data["reason"]).strip()[:2000]
    return data


def call_claude_json(
    api_key: str,
    user_content: str,
) -> tuple[dict[str, Any], int]:
    """
    Returns (parsed_json, total_tokens_used).
    On failure raises or returns MAYBE — caller wraps in try/except.
    """
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=MODEL,
        max_tokens=256,
        messages=[
            {
                "role": "user",
                "content": f"{JSON_INSTRUCTION}\n\n{user_content}",
            }
        ],
    )
    text = ""
    for block in message.content:
        t = getattr(block, "text", None)
        if t:
            text += t
        elif isinstance(block, dict) and block.get("type") == "text":
            text += block.get("text") or ""
    usage = getattr(message, "usage", None)
    tokens = 0
    if usage is not None:
        tokens = int(getattr(usage, "input_tokens", 0) or 0) + int(
            getattr(usage, "output_tokens", 0) or 0
        )
    return _extract_json(text), tokens


def _safe_call_claude(
    api_key: str, prompt: str
) -> tuple[dict[str, Any] | None, int, str | None]:
    try:
        data, tokens = call_claude_json(api_key, prompt)
        return data, tokens, None
    except Exception as e:
        logger.exception("Claude API error")
        return None, 0, str(e)


def step1_apollo(
    api_key: str,
    icp_description: str,
    short_description: str,
    technologies: str,
    keywords: str,
) -> tuple[str, str, str, int]:
    """
    Returns (status, reason, source, tokens).
    status in YES, NO, MAYBE — MAYBE triggers step 2.
    """
    has_any = any(
        str(x).strip()
        for x in (short_description, technologies, keywords)
        if x is not None and str(x).strip() and str(x).lower() != "nan"
    )
    if not has_any:
        return "MAYBE", "No Apollo description/technologies/keywords", "Apollo_Data", 0

    prompt = f"""ICP (Ideal Customer Profile) criteria:
{icp_description}

Company data from Apollo export:
- Short Description: {short_description or "(empty)"}
- Technologies: {technologies or "(empty)"}
- Keywords: {keywords or "(empty)"}

Based only on the above, does this company match the ICP?"""

    data, tokens, err = _safe_call_claude(api_key, prompt)
    if err or data is None:
        return "MAYBE", f"Step1 API/parse error: {err or 'unknown'}", "Apollo_Data", tokens

    return data["status"], data["reason"], "Apollo_Data", tokens


async def _crawl_text(url: str, max_chars: int = 3000) -> tuple[str | None, str | None]:
    try:
        from crawl4ai import AsyncWebCrawler  # lazy import

        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
        if result is None:
            return None, "empty result"
        if hasattr(result, "success") and not result.success:
            err = getattr(result, "error_message", None) or "crawl unsuccessful"
            code = getattr(result, "status_code", None)
            return None, f"{err} (status={code})"
        md = getattr(result, "markdown", None)
        text = str(md).strip() if md is not None else ""
        if not text:
            ch = getattr(result, "cleaned_html", None) or getattr(result, "html", None)
            text = (str(ch).strip() if ch else "")
        if not text and hasattr(result, "extracted_content"):
            text = (result.extracted_content or "").strip()
        if not text:
            return None, "no text extracted"
        return text[:max_chars], None
    except Exception as e:
        logger.exception("crawl4ai error")
        return None, str(e)


def fetch_website_text(url: str) -> tuple[str | None, str | None]:
    if not url or not str(url).strip() or str(url).lower() in ("nan", "none"):
        return None, "empty URL"
    u = str(url).strip()
    if not u.startswith(("http://", "https://")):
        u = "https://" + u

    def _run_crawl_in_thread() -> tuple[str | None, str | None]:
        """Streamlit / nested contexts may already have a running loop; fresh thread is safe."""
        return asyncio.run(_crawl_text(u))

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(_run_crawl_in_thread)
            return fut.result(timeout=180)
    except concurrent.futures.TimeoutError:
        return None, "crawl timeout (180s)"
    except Exception as e:
        logger.exception("async crawl wrapper")
        return None, str(e)


def step2_website(
    api_key: str,
    icp_description: str,
    website_url: str,
) -> tuple[str, str, str, int]:
    text, err = fetch_website_text(website_url)
    if err or not text:
        return "MAYBE", f"Website unavailable or empty: {err or 'unknown'}", "Website_Scraped", 0

    prompt = f"""ICP criteria:
{icp_description}

Below is text extracted from the company website (may be truncated):
---
{text}
---

Does this company match the ICP?"""

    data, tokens, api_err = _safe_call_claude(api_key, prompt)
    if api_err or data is None:
        return "MAYBE", f"Step2 Claude error: {api_err or 'unknown'}", "Website_Scraped", tokens

    return data["status"], data["reason"], "Website_Scraped", tokens


def step3_ddg(
    api_key: str,
    icp_description: str,
    company_name: str,
    linkedin_url: str,
) -> tuple[str, str, str, int]:
    query = f'{company_name or ""} {linkedin_url or ""} overview'.strip()
    snippets: list[str] = []
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        for r in results:
            title = r.get("title") or ""
            body = r.get("body") or ""
            snippets.append(f"{title}\n{body}")
    except Exception as e:
        logger.exception("duckduckgo_search error")
        return (
            "MANUAL_REVIEW",
            f"DDG search failed: {e}",
            "DDG_Search",
            0,
        )

    if not snippets:
        return "MANUAL_REVIEW", "No DDG results", "DDG_Search", 0

    combined = "\n\n---\n\n".join(snippets)[:4000]

    prompt = f"""ICP criteria:
{icp_description}

DuckDuckGo search snippets for "{query}":
---
{combined}
---

Final decision: does this company match the ICP? If still unclear, answer MAYBE."""

    data, tokens, api_err = _safe_call_claude(api_key, prompt)
    if api_err or data is None:
        return "MANUAL_REVIEW", f"Step3 Claude error: {api_err or 'unknown'}", "DDG_Search", tokens

    st = data["status"]
    if st == "MAYBE":
        return "MANUAL_REVIEW", data["reason"], "DDG_Search", tokens
    return st, data["reason"], "DDG_Search", tokens


def score_company_row(
    api_key: str,
    icp_description: str,
    row: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    """
    Full waterfall for one company row (dict-like).
    Returns (result_dict with ICP_Status, Reason, Data_Source), total_tokens).
    """
    def col(*names: str) -> str:
        for n in names:
            if n in row and row[n] is not None:
                v = row[n]
                if hasattr(v, "item"):
                    try:
                        v = v.item()
                    except Exception:
                        pass
                s = str(v).strip()
                if s and s.lower() != "nan":
                    return s
        return ""

    short_desc = col("Short Description", "short_description")
    tech = col("Technologies", "technologies")
    kws = col("Keywords", "keywords")
    website = col("Website", "website", "Company Website")
    company_name = col("Company Name", "Company", "Name", "company_name")
    linkedin = col("Company Linkedin Url", "Company LinkedIn Url", "linkedin_url", "Linkedin Url")

    total_tokens = 0

    status, reason, source, tok = step1_apollo(
        api_key, icp_description, short_desc, tech, kws
    )
    total_tokens += tok

    if status in ("YES", "NO"):
        return (
            {
                "ICP_Status": status,
                "Reason": reason,
                "Data_Source": source,
            },
            total_tokens,
        )

    # MAYBE → step 2
    status, reason, source, tok = step2_website(api_key, icp_description, website)
    total_tokens += tok

    if status in ("YES", "NO"):
        return (
            {
                "ICP_Status": status,
                "Reason": reason,
                "Data_Source": source,
            },
            total_tokens,
        )

    # Still MAYBE or website failed → step 3
    status, reason, source, tok = step3_ddg(api_key, icp_description, company_name, linkedin)
    total_tokens += tok

    return (
        {
            "ICP_Status": status,
            "Reason": reason,
            "Data_Source": source,
        },
        total_tokens,
    )


def estimate_cost_usd(
    total_tokens: int,
    cost_per_million: float = DEFAULT_COST_PER_MILLION_TOKENS_USD,
) -> float:
    return round(total_tokens * cost_per_million / 1_000_000, 6)


def export_colored_xlsx(df: "pd.DataFrame") -> bytes:
    """Write DataFrame to xlsx and color `ICP_Status` cells (YES/NO/else)."""
    import io

    from openpyxl import load_workbook
    from openpyxl.styles import PatternFill

    buf = io.BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl")
    buf.seek(0)
    wb = load_workbook(buf)
    ws = wb.active
    headers = [cell.value for cell in ws[1]]
    if "ICP_Status" not in headers:
        out = io.BytesIO()
        wb.save(out)
        return out.getvalue()

    col_idx = headers.index("ICP_Status") + 1
    green = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
    red = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
    yellow = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")

    for row in range(2, ws.max_row + 1):
        cell = ws.cell(row=row, column=col_idx)
        val = cell.value
        if val == "YES":
            cell.fill = green
        elif val == "NO":
            cell.fill = red
        else:
            cell.fill = yellow

    out = io.BytesIO()
    wb.save(out)
    return out.getvalue()
