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
    DEFAULT_COST_INPUT_PER_MILLION_USD,
    DEFAULT_COST_OUTPUT_PER_MILLION_USD,
    estimate_cost_usd,
    export_colored_xlsx,
    score_company_row,
)

BASE_DIR = Path(__file__).resolve().parent
PROFILES_PATH = BASE_DIR / "profiles.yaml"
RESULTS_DIR = BASE_DIR / "results"


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
    if "cost_input_per_million" not in st.session_state:
        st.session_state.cost_input_per_million = float(DEFAULT_COST_INPUT_PER_MILLION_USD)
    if "cost_output_per_million" not in st.session_state:
        st.session_state.cost_output_per_million = float(DEFAULT_COST_OUTPUT_PER_MILLION_USD)


# ── background worker ─────────────────────────────────────────────────────────

def _worker(job_id: str, api_key: str, icp_desc: str, df: pd.DataFrame) -> None:
    job = _get_jobs()[job_id]
    stop_event: threading.Event = job["stop_event"]
    out_rows: list[dict] = []
    total_in = total_out = 0
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

        job["processed"] = len(out_rows)

        try:
            result, (in_tok, out_tok) = score_company_row(api_key, icp_desc, row_dict)
            total_in += in_tok
            total_out += out_tok
        except Exception as e:
            errors += 1
            result = {
                "ICP_Status": "MANUAL_REVIEW",
                "Reason": f"Unhandled: {e}",
                "Data_Source": "Error",
            }

        status = result.get("ICP_Status", "?")
        source = result.get("Data_Source", "")
        reason = (result.get("Reason") or "")[:70]
        icon = {"YES": "✅", "NO": "❌", "MAYBE": "⚠️"}.get(status, "🔵")
        job["log"].append(f"{icon} {cname}  [{source}]  {reason}")

        out_rows.append({**row_dict, **result})
        job["processed"] = len(out_rows)
        job["errors"] = errors
        job["input_tokens"] = total_in
        job["output_tokens"] = total_out

    result_df = pd.DataFrame(out_rows) if out_rows else pd.DataFrame()
    job["result_df"] = result_df
    if not result_df.empty:
        xlsx_bytes = export_colored_xlsx(result_df)
        job["result_xlsx"] = xlsx_bytes
        # Persist to disk so results survive page refresh / app restart
        try:
            RESULTS_DIR.mkdir(exist_ok=True)
            result_path = RESULTS_DIR / f"apollo_scored_{job_id[:8]}.xlsx"
            result_path.write_bytes(xlsx_bytes)
            job["result_path"] = str(result_path)
            job["log"].append(f"💾 Результат сохранён: {result_path.name}")
        except Exception as e:
            logger.warning("Could not save result to disk: %s", e)
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
        "input_tokens": 0,
        "output_tokens": 0,
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

        st.caption("Цена токенов (Claude Haiku 4.5)")
        st.number_input(
            "Input: цена за 1M токенов ($)",
            min_value=0.01, max_value=1000.0, step=0.1, format="%.2f",
            key="cost_input_per_million",
            disabled=is_running,
            help="Входящие токены. Haiku 4.5 = $1.00/M",
        )
        st.number_input(
            "Output: цена за 1M токенов ($)",
            min_value=0.01, max_value=1000.0, step=0.1, format="%.2f",
            key="cost_output_per_million",
            disabled=is_running,
            help="Исходящие токены. Haiku 4.5 = $5.00/M",
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
    c1, c2, c3, c4, c5 = st.columns(5)
    m_proc    = c1.empty()
    m_err     = c2.empty()
    m_in_tok  = c3.empty()
    m_out_tok = c4.empty()
    m_cost    = c5.empty()
    log_placeholder = st.empty()

    cost_in  = float(st.session_state.get("cost_input_per_million")  or DEFAULT_COST_INPUT_PER_MILLION_USD)
    cost_out = float(st.session_state.get("cost_output_per_million") or DEFAULT_COST_OUTPUT_PER_MILLION_USD)

    def render_dashboard(j: dict) -> None:
        total = j["total"] or 1
        progress_bar.progress(min(j["processed"] / total, 1.0))
        m_proc.metric("Обработано", f"{j['processed']} / {j['total']}")
        m_err.metric("Ошибок", j["errors"])
        in_tok  = j["input_tokens"]
        out_tok = j["output_tokens"]
        m_in_tok.metric("Input токены",  f"{in_tok:,}".replace(",", "\u202f"))
        m_out_tok.metric("Output токены", f"{out_tok:,}".replace(",", "\u202f"))
        m_cost.metric("Затраты ($)", f"${estimate_cost_usd(in_tok, out_tok, cost_in, cost_out):.4f}")
        with log_placeholder.container(height=320, border=True):
            st.caption("Лог обработки")
            st.code("\n".join(j["log"][-100:]), language=None)

    if job is None:
        m_proc.metric("Обработано", "—")
        m_err.metric("Ошибок", "—")
        m_in_tok.metric("Input токены", "—")
        m_out_tok.metric("Output токены", "—")
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

    # ── saved results on disk (survive page refresh / process restart) ─────────
    if RESULTS_DIR.exists():
        saved = sorted(RESULTS_DIR.glob("*.xlsx"), key=lambda p: p.stat().st_mtime, reverse=True)
        if saved:
            st.divider()
            st.subheader("Сохранённые результаты")
            for i, path in enumerate(saved[:10]):
                mtime = path.stat().st_mtime
                import datetime
                ts = datetime.datetime.fromtimestamp(mtime).strftime("%d.%m.%Y %H:%M")
                col_name, col_btn, col_del = st.columns([4, 2, 1])
                col_name.write(f"📄 {path.name}  \n*{ts}*")
                with open(path, "rb") as f:
                    col_btn.download_button(
                        label="⬇ Скачать",
                        data=f.read(),
                        file_name=path.name,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"saved_dl_{i}",
                    )
                if col_del.button("🗑", key=f"del_{i}", help="Удалить файл"):
                    path.unlink(missing_ok=True)
                    st.rerun()


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
