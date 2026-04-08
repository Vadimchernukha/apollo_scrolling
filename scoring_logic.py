"""
Waterfall lead scoring: Apollo data → website scrape → DuckDuckGo fallback.
Uses Anthropic Claude 3 Haiku with strict JSON responses.
"""

from __future__ import annotations

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


def fetch_website_text(url: str, max_chars: int = 3000) -> tuple[str | None, str | None]:
    """Fetch plain text from a URL using requests + BeautifulSoup."""
    if not url or not str(url).strip() or str(url).lower() in ("nan", "none"):
        return None, "empty URL"
    u = str(url).strip()
    if not u.startswith(("http://", "https://")):
        u = "https://" + u
    try:
        import requests
        from bs4 import BeautifulSoup

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(u, headers=headers, timeout=15, allow_redirects=True)
        if resp.status_code >= 400:
            return None, f"HTTP {resp.status_code}"
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "form", "iframe"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        text = re.sub(r"\s{2,}", " ", text).strip()
        if not text:
            return None, "no text extracted"
        return text[:max_chars], None
    except Exception as e:
        logger.exception("fetch_website_text error")
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


# ── Enterprise profile ────────────────────────────────────────────────────────

ENTERPRISE_PROMPT_TEMPLATE = """You are a senior B2B analyst qualifying companies for an enterprise technology pipeline.
You are given the company's content below.

## TASK: Is this a technology company OR an enterprise that meaningfully uses technology in operations?

**Include** companies that either:
1. Have their own technology product, platform, or software, OR
2. Run industrial or scaled operations where technology, automation, or digital systems are plausibly central (manufacturing, logistics, regulated production, etc.)

**Do not** treat "services" as disqualifying by itself. Logistics, 3PL, healthcare operations, and B2B manufacturing are in scope. **Exclude** only the categories listed under REJECT (especially people-selling IT shops and pure non-tech services).

---

### QUALIFY — include if the company matches at least one:

**Product & Tech companies:**
- SaaS, software, or platform company (B2B or B2C)
- Hardware or industrial equipment manufacturer
- IoT, robotics, AI, or automation company
- Biotech, medtech, cleantech, deeptech startup
- Marketplace, e-commerce, or digital platform
- Fintech, insurtech, proptech, legaltech product
- Data analytics, BI, or AI/ML tools company

**Industrial & B2B production (often under-represented on the web — lean IN when production is clear):**
- Contract manufacturers: CDMO, CMO, toll manufacturing, private-label production (food, supplements, pharma-adjacent, medical devices)
- B2B ingredient, API, raw material, or formulation suppliers with real manufacturing or certified facilities (GMP, ISO, FSSC, etc.)
- Chemical, materials, or process companies supplying industry (not retail-only shops)

**Technology-using companies:**
- Manufacturing company (factory, plant, industrial production) — qualify regardless of how "digital" the website looks
- Logistics, transportation, supply chain, 3PL/4PL provider
- Healthcare provider or hospital system (clinical operations; digital maturity varies)
- Retail or e-commerce operator with own infrastructure
- Energy, utilities, or infrastructure company
- Construction or real estate operator (development, contracting) using modern project or field tools — not the same as a residential brokerage
- Agricultural tech or food production with an industrial scale
- Any credible signals of: software, systems, ERP, MES, QMS, LIMS, automation, digitalization, smart manufacturing, Industry 4.0, SCADA, traceability

---

### REJECT — exclude when the **primary** business is one of these:

- **IT Outsourcing / Outstaffing** — selling developer time, dedicated teams, staff augmentation, remote/nearshore/offshore dev benches as the main offer
- **Custom software agencies** — build software only for clients, no meaningful own product (classic dev/body shop positioning)
- **Recruitment / HR staffing / Headhunting** — placing people as the core revenue model
- **Pure strategy consulting** — no technology delivery (McKinsey-style-only); if they sell implementation or own tools, use BORDERLINE rules
- **Pure media / news / blog / content** businesses
- **Law, accounting, audit** firms unless they clearly sell a tech product
- **Clearly non-technical local services** — cleaning, catering, small retail with no industrial scale, traditional real estate brokerage (not proptech)

**Strong reject signal (IT staff-selling):** primary messaging is "hire developers", "dedicated team", "IT outsourcing partner", "staff augmentation" — reject.

---

### BORDERLINE RULES (prefer inclusion when the vertical is industrial B2B):

- Unclear whether the company is **industrial B2B** (manufacturing, ingredients, logistics, regulated production) vs **generic non-tech services** → **QUALIFY** with **confidence: medium** unless the text clearly fits REJECT above.
- Company does both consulting **and** has own product → QUALIFY (lean toward including).
- IT / systems consultancy for a **specific industry** (e.g. manufacturing ERP, logistics, quality, MES) with **named solutions, accelerators, or IP** → QUALIFY; if the site is indistinguishable from a generic dev shop → REJECT.
- **Systems integrator** (SAP, Oracle, Microsoft, etc.): **REJECT** if positioning is generic IT services or staff augmentation. **QUALIFY with medium** if the focus is enterprise/industrial delivery (ERP/MES/automation for factories, supply chain, regulated environments) even without a separate product brand.

---

### COMPANY TYPE — classify into one of:

- `Software Product / SaaS` — own software product or platform
- `Hardware / Industrial Manufacturer` — makes physical products, machines, equipment
- `Logistics / Supply Chain / 3PL` — transport, freight, warehousing
- `Manufacturing / Factory` — production of goods (incl. contract manufacturing)
- `Biotech / Medtech / Healthtech` — life sciences, medical devices, digital health
- `Cleantech / Energy` — renewables, energy management, sustainability
- `Fintech / Insurtech` — financial technology product
- `E-commerce / Marketplace` — digital commerce platforms
- `IT Consulting (Industry-Specific)` — tech consulting with own tools or strong vertical focus
- `Other Tech-Using Enterprise` — tech-adopting company not fitting above

---

## OUTPUT

Set confidence:
- **high** — clearly qualifies or clearly rejected, little ambiguity
- **medium** — qualifies but mixed signals, or industrial B2B where the site is thin
- **low** — very little text; do not use low as automatic reject — if industrial production is still evident, prefer medium + qualify

---

Company: {company_name}

Content:
---
{page_text}
---

Answer in JSON only, no markdown:
{{
  "company_name": "official company name as shown, or input name if unclear",
  "is_enterprise_match": true or false,
  "confidence": "high" or "medium" or "low",
  "company_type": "one of the types listed above",
  "rejection_reason": "if false — brief reason why rejected, else null",
  "reason": "max 15 words explaining the qualification decision"
}}"""


def _extract_enterprise_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("empty model response")
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    data = json.loads(text)
    required = {"is_enterprise_match", "confidence", "company_type", "reason"}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"missing enterprise fields: {missing}")
    data["is_enterprise_match"] = bool(data["is_enterprise_match"])
    conf = str(data.get("confidence", "low")).lower().strip()
    data["confidence"] = conf if conf in ("high", "medium", "low") else "low"
    data["company_type"] = str(data.get("company_type") or "").strip()
    data["reason"] = str(data.get("reason") or "").strip()[:2000]
    data["rejection_reason"] = str(data.get("rejection_reason") or "").strip()
    return data


def call_claude_enterprise(
    api_key: str,
    company_name: str,
    page_text: str,
) -> tuple[dict[str, Any], int]:
    """Enterprise-specific Claude call. Returns (parsed_json, total_tokens)."""
    prompt = ENTERPRISE_PROMPT_TEMPLATE.format(
        company_name=company_name or "Unknown",
        page_text=page_text,
    )
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
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
    return _extract_enterprise_json(text), tokens


def _safe_call_enterprise(
    api_key: str, company_name: str, page_text: str
) -> tuple[dict[str, Any] | None, int, str | None]:
    try:
        data, tokens = call_claude_enterprise(api_key, company_name, page_text)
        return data, tokens, None
    except Exception as e:
        logger.exception("Enterprise Claude API error")
        return None, 0, str(e)


def _enterprise_result(data: dict[str, Any] | None, source: str) -> dict[str, Any]:
    if data is None:
        return {
            "ICP_Status": "MANUAL_REVIEW",
            "Reason": f"No data or API error at {source}",
            "Data_Source": source,
            "Confidence": "low",
            "Company_Type": "",
            "Rejection_Reason": "",
        }
    return {
        "ICP_Status": "YES" if data["is_enterprise_match"] else "NO",
        "Reason": data["reason"],
        "Data_Source": source,
        "Confidence": data["confidence"],
        "Company_Type": data.get("company_type", ""),
        "Rejection_Reason": data.get("rejection_reason", ""),
    }


def score_company_row_enterprise(
    api_key: str,
    row: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    """
    Enterprise waterfall: Apollo data → website → DDG.
    Stops on confidence high/medium; continues on low (thin data).
    Returns (result_dict, total_tokens).
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

    # Step 1: Apollo data
    has_apollo = any(
        s for s in (short_desc, tech, kws) if s
    )
    if has_apollo:
        apollo_text = (
            f"Apollo export data:\n"
            f"- Short Description: {short_desc or '(empty)'}\n"
            f"- Technologies: {tech or '(empty)'}\n"
            f"- Keywords: {kws or '(empty)'}"
        )
        data, tokens, err = _safe_call_enterprise(api_key, company_name, apollo_text)
        total_tokens += tokens
        if data is not None and data["confidence"] in ("high", "medium"):
            return _enterprise_result(data, "Apollo_Data"), total_tokens

    # Step 2: Website
    page_text, web_err = fetch_website_text(website)
    if not web_err and page_text:
        data, tokens, err = _safe_call_enterprise(api_key, company_name, page_text)
        total_tokens += tokens
        if data is not None and data["confidence"] in ("high", "medium"):
            return _enterprise_result(data, "Website_Scraped"), total_tokens

    # Step 3: DDG
    query = f'{company_name} {linkedin} overview'.strip()
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
        logger.exception("duckduckgo_search error (enterprise step3)")

    if snippets:
        combined = "\n\n---\n\n".join(snippets)[:4000]
        data, tokens, err = _safe_call_enterprise(api_key, company_name, combined)
        total_tokens += tokens
        if data is not None:
            return _enterprise_result(data, "DDG_Search"), total_tokens

    return _enterprise_result(None, "DDG_Search"), total_tokens


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
