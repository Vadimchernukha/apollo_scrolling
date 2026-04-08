"""
Streamlit Lead Scoring app: Apollo export + ICP profiles + Claude Haiku waterfall.
Scoring runs in a background thread so the UI stays responsive (Stop button works).
Auth via streamlit-authenticator (cookie-based, persists across page refreshes).
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import streamlit_authenticator as stauth
import yaml

from scoring_logic import (
    DEFAULT_COST_PER_MILLION_TOKENS_USD,
    estimate_cost_usd,
    export_colored_xlsx,
    score_company_row,
)

BASE_DIR = Path(__file__).resolve().parent
PROFILES_PATH = BASE_DIR / "profiles.yaml"


# ── persistent job store ──────────────────────────────────────────────────────

@st.cache_resource
def _get_jobs() -> dict[str, dict[str, Any]]:
    """Created once per process, survives every st.rerun()."""
    return {}


# ── authenticator (cookie-based, survives page refresh) ──────────────────────

def _get_authenticator() -> stauth.Authenticate:
    """Create authenticator each run — credentials from st.secrets or defaults."""
    try:
        credentials = st.secrets.get("credentials", None)
        cookie_secret = st.secrets.get("COOKIE_SECRET", "apollo-brain-default-secret-2026")
    except Exception:
        credentials = None
        cookie_secret = "apollo-brain-default-secret-2026"

    if not credentials:
        credentials = {
            "usernames": {
                "admin": {"name": "Admin", "email": "", "password": "admin"}
            }
        }

    return stauth.Authenticate(
        credentials=dict(credentials),
        cookie_name="apollo_brain_auth",
        cookie_key=cookie_secret,
        cookie_expiry_days=30,
        auto_hash=True,
    )


# ── helpers ───────────────────────────────────────────────────────────────────

def load_profiles() -> dict:
    with open(PROFILES_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_api_key() -> str:
    """st.secrets first (Streamlit Cloud), then .env / environment."""
    key = ""
    try:
        key = st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:
        pass
    if not key:
        key = os.environ.get("ANTHROPIC_API_KEY", "")
    return key.strip()


# ── session init ──────────────────────────────────────────────────────────────

def init_session() -> None:
    defaults: dict[str, Any] = {
        "uploaded_df": None,
        "upload_name": "",
        "icp_profile": None,
        "job_id": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    if "cost_per_million_field" not in st.session_state:
        st.session_state.cost_per_million_field = float(DEFAULT_COST_PER_MILLION_TOKENS_USD)


# ── background worker ─────────────────────────────────────────────────────────

def _worker(job_id: str, api_key: str, icp_desc: str, df: pd.DataFrame) -> None:
    job = _get_jobs()[job_id]
    stop_event: threading.Event = job["stop_event"]
    out_rows: list[dict] = []
    total_tokens = 0
    errors = 0

    for idx, row in df.iterrows():
        if stop_event.is_set():
            job["log"].append("⛔ Остановлено пользователем.")
            break

        row_dict = row.to_dict()
        company = row_dict.get("Company Name", row_dict.get("Company", f"row_{idx}"))
        try:
            cname = str(company) if company is not None and str(company) != "nan" else f"row_{idx}"
        except Exception:
            cname = f"row_{idx}"

        job["log"].append(f"Analyzing [{cname}]...")
        job["processed"] = len(out_rows)

        try:
            result, tok = score_company_row(api_key, icp_desc, row_dict)
            total_tokens += tok
        except Exception as e:
            errors += 1
            job["log"].append(f"  ERROR: {e} → MANUAL_REVIEW")
            result = {
                "ICP_Status": "MANUAL_REVIEW",
                "Reason": f"Unhandled: {e}",
                "Data_Source": "Error",
            }

        out_rows.append({**row_dict, **result})
        job["processed"] = len(out_rows)
        job["errors"] = errors
        job["tokens"] = total_tokens

    result_df = pd.DataFrame(out_rows) if out_rows else pd.DataFrame()
    job["result_df"] = result_df
    if not result_df.empty:
        job["result_xlsx"] = export_colored_xlsx(result_df)
    job["log"].append(
        "Готово." if not stop_event.is_set()
        else "Остановлено — частичный результат доступен для скачивания."
    )
    job["done"] = True


def start_job(api_key: str, icp_desc: str, df: pd.DataFrame) -> str:
    job_id = str(uuid.uuid4())
    _get_jobs()[job_id] = {
        "stop_event": threading.Event(),
        "done": False,
        "processed": 0,
        "total": len(df),
        "errors": 0,
        "tokens": 0,
        "log": [],
        "result_df": None,
        "result_xlsx": None,
    }
    threading.Thread(
        target=_worker, args=(job_id, api_key, icp_desc, df), daemon=True
    ).start()
    return job_id


def get_job(job_id: str | None) -> dict[str, Any] | None:
    return _get_jobs().get(job_id) if job_id else None


# ── main UI ───────────────────────────────────────────────────────────────────

def main_ui(authenticator: stauth.Authenticate) -> None:
    st.title("Apollo Lead Scoring")
    st.caption("B2B ICP scoring: Claude Haiku 4.5 + Apollo → сайт → DuckDuckGo.")

    profiles = load_profiles()
    profile_names = list(profiles.keys())
    if not profile_names:
        st.error("Нет профилей в profiles.yaml")
        return

    job = get_job(st.session_state.job_id)
    is_running = bool(job and not job["done"])
    api_key = get_api_key()

    # ── sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Настройки")

        f = st.file_uploader(
            "Выгрузка Apollo (CSV / Excel)",
            type=["csv", "xlsx", "xls"],
            disabled=is_running,
        )
        if f is not None:
            try:
                st.session_state.uploaded_df = (
                    pd.read_csv(f) if f.name.lower().endswith(".csv")
                    else pd.read_excel(f)
                )
                st.session_state.upload_name = f.name
            except Exception as e:
                st.error(f"Ошибка чтения файла: {e}")
                st.session_state.uploaded_df = None

        if st.session_state.uploaded_df is not None:
            st.success(
                f"Загружено: {st.session_state.upload_name} "
                f"({len(st.session_state.uploaded_df)} строк)"
            )

        default_idx = 0
        if st.session_state.icp_profile in profile_names:
            default_idx = profile_names.index(st.session_state.icp_profile)
        icp = st.selectbox(
            "ICP профиль", profile_names, index=default_idx,
            key="icp_select", disabled=is_running,
        )
        st.session_state.icp_profile = icp

        st.number_input(
            "Цена за 1M токенов ($)",
            min_value=0.01,
            max_value=1000.0,
            step=0.05,
            format="%.2f",
            key="cost_per_million_field",
            disabled=is_running,
            help="Claude Haiku 4.5: $1/M input, $5/M output. Дефолт 2.0 — средняя.",
        )

        st.divider()
        authenticator.logout(button_name="Выйти", location="sidebar")

    df = st.session_state.uploaded_df
    icp_key = st.session_state.icp_profile
    icp_desc = profiles[icp_key].get("description", "") if icp_key else ""

    # ── START / STOP buttons ──────────────────────────────────────────────────
    btn_col, info_col = st.columns([2, 3])
    with btn_col:
        b_start = st.button(
            "▶ START SCORE",
            type="primary",
            disabled=is_running or df is None or not api_key,
            use_container_width=True,
        )
        b_stop = st.button(
            "⏹ STOP",
            disabled=not is_running,
            use_container_width=True,
        )
    with info_col:
        if df is None:
            st.info("Загрузите CSV или Excel в сайдбаре.")
        elif not api_key:
            st.error("ANTHROPIC_API_KEY не найден. Добавьте в .streamlit/secrets.toml или .env.")
        elif is_running:
            st.info("Идёт скоринг… нажмите ⏹ STOP чтобы прервать.")

    if b_start and df is not None and api_key:
        st.session_state.job_id = start_job(api_key, icp_desc, df)
        st.rerun()

    if b_stop:
        if job:
            job["stop_event"].set()

    # ── dashboard ─────────────────────────────────────────────────────────────
    job = get_job(st.session_state.job_id)

    progress_bar = st.progress(0.0)
    c1, c2, c3, c4 = st.columns(4)
    m_proc   = c1.empty()
    m_err    = c2.empty()
    m_tokens = c3.empty()
    m_cost   = c4.empty()
    log_box  = st.empty()

    rate = float(
        st.session_state.get("cost_per_million_field") or DEFAULT_COST_PER_MILLION_TOKENS_USD
    )

    def render_dashboard(j: dict) -> None:
        total = j["total"] or 1
        progress_bar.progress(min(j["processed"] / total, 1.0))
        m_proc.metric("Обработано", f"{j['processed']} / {j['total']}")
        m_err.metric("Ошибок", j["errors"])
        tokens = j["tokens"]
        m_tokens.metric("Токенов", f"{tokens:,}".replace(",", "\u202f"))
        m_cost.metric("Затраты ($)", f"${estimate_cost_usd(tokens, rate):.4f}")
        log_box.markdown("**Лог**\n```text\n{}\n```".format(
            "\n".join(j["log"][-200:])
        ))

    if job is None:
        m_proc.metric("Обработано", "—")
        m_err.metric("Ошибок", "—")
        m_tokens.metric("Токенов", "—")
        m_cost.metric("Затраты ($)", "—")
    elif not job["done"]:
        render_dashboard(job)
        time.sleep(0.6)
        st.rerun()
    else:
        render_dashboard(job)
        result_df = job.get("result_df")
        result_xlsx = job.get("result_xlsx")
        if result_df is not None and not result_df.empty:
            st.success(f"Готово! Обработано {job['processed']} компаний.")
            if result_xlsx:
                st.download_button(
                    label="⬇ Скачать результат (.xlsx)",
                    data=result_xlsx,
                    file_name="apollo_scored.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"dl_{st.session_state.job_id}",
                )


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title="Apollo Lead Scoring", layout="wide")
    init_session()

    authenticator = _get_authenticator()
    authenticator.login(
        location="main",
        fields={
            "Form name": "Apollo Lead Scoring",
            "Username": "Логин",
            "Password": "Пароль",
            "Login": "Войти",
        },
    )

    auth_status = st.session_state.get("authentication_status")

    if auth_status is False:
        st.error("Неверный логин или пароль.")
    elif auth_status is None:
        st.info("Введите логин и пароль.")
    else:
        main_ui(authenticator)


if __name__ == "__main__":
    main()
