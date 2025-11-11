# app.py
import os
import json
import sqlite3
from pathlib import Path
import streamlit as st
import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

# ---------------- CONFIG ----------------
DEFAULT_MODEL = "llama-3.1-8b-instant"
MAX_ROWS = 20

# ---------------- LLM SETUP ----------------
@st.cache_resource
def get_llm(groq_api_key: str | None):
    """Create and cache a ChatGroq instance."""
    key = groq_api_key or os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError("Groq API Key not provided. Set in sidebar or env GROQ_API_KEY")

    return ChatGroq(model=DEFAULT_MODEL, temperature=0, streaming=False, api_key=key)


# ---------------- DATABASE HELPERS ----------------
def connect_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def get_schema(conn: sqlite3.Connection) -> dict:
    schema = {}
    cur = conn.cursor()
    tables = cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
    ).fetchall()
    for (table_name,) in tables:
        cols = cur.execute(f"PRAGMA table_info({table_name});").fetchall()
        col_names = [c[1] for c in cols]
        schema[table_name] = col_names
    return schema


def schema_to_text(schema: dict) -> str:
    lines = []
    for table, cols in schema.items():
        preview = ", ".join(cols[:10])
        extra = " ..." if len(cols) > 10 else ""
        lines.append(f"- {table}({preview}{extra})")
    return "\n".join(lines)


# ---------------- AGENT LOGIC ----------------
def ask_llm_for_sql(llm: ChatGroq, question: str, schema_text: str) -> dict:
    """Ask LLM to propose a safe SELECT query and return parsed JSON result."""
    system = SystemMessage(
        content=(
            "You are 'MovieGenie', a helpful AI for SQLite movie databases.\n"
            "You MUST use only the tables and columns listed in SCHEMA below.\n"
            "Write only safe SELECT queries (no INSERT/UPDATE/DELETE, no DROP, etc.).\n"
            "If the question is vague, make a reasonable assumption and mention it in 'thinking'.\n"
            f"SCHEMA:\n{schema_text}"
        )
    )

    user = HumanMessage(
        content=(
            f"User question: {question}\n\n"
            "Reply ONLY in JSON like this:\n"
            '{"sql":"...","thinking":"...","followups":["...","..."]}'
        )
    )

    try:
        resp = llm.invoke([system, user])
    except Exception as e:
        # LLM call failed (network / key / API)
        return {
            "sql": "",
            "thinking": f"LLM invocation failed: {e}",
            "followups": []
        }

    text = (resp.content or "").strip()

    # try to extract JSON object from the model's text response
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        json_text = text[start:end]
        data = json.loads(json_text)
    except Exception:
        # If parsing fails, return a helpful fallback (not the crude SELECT 'Sorry' row)
        return {
            "sql": "",
            "thinking": "Model did not return strict JSON. Here is raw model output below.",
            "followups": [],
            "raw": text  # include raw for debugging / display
        }

    # normalize fields
    data.setdefault("sql", "")
    data.setdefault("thinking", "")
    data.setdefault("followups", [])
    if not isinstance(data["followups"], list):
        data["followups"] = [str(data["followups"])]

    return data


# ---------------- SQL EXECUTION ----------------
def run_sql(conn: sqlite3.Connection, sql: str) -> pd.DataFrame | str:
    """Execute the proposed SQL safely (only SELECT allowed)."""
    if not sql:
        return "No SQL to run."

    sql_clean = sql.strip().rstrip(";")
    if not sql_clean.lower().startswith("select"):
        return "Blocked: Only SELECT queries are allowed"

    if "limit" not in sql_clean.lower():
        sql_clean += f" LIMIT {MAX_ROWS}"

    try:
        df = pd.read_sql_query(sql_clean, conn)
        return df
    except Exception as e:
        return f"SQL Error: {e}"


# ---------------- STREAMLIT APP ----------------
def main():
    st.set_page_config(page_title="MovieGenie", page_icon="🎬", layout="wide")
    st.title("🎬 MovieGenie: Ask about Movies!")

    # Sidebar: DB + API key
    with st.sidebar:
        st.header("Step 1: Database")
        db_path = st.text_input("Path to movies.db", value="movies.db")

        st.header("Step 2: Groq API Key")
        key_input = st.text_input("GROQ_API_KEY", type="password")
        if key_input:
            os.environ["GROQ_API_KEY"] = key_input

        st.markdown("---")
        st.caption("If you store your key in a .env file, load_dotenv() is already called.")

    # validate DB path
    if not db_path or not Path(db_path).exists():
        st.warning("Please provide a valid movies.db path (or upload it).")
        return

    # Connect DB and read schema
    try:
        conn = connect_db(db_path)
    except Exception as e:
        st.error(f"Could not open database: {e}")
        return

    schema = get_schema(conn)
    if not schema:
        st.error("No tables found in this database.")
        return

    schema_text = schema_to_text(schema)
    st.subheader("📚 Tables & Columns")
    st.code(schema_text)

    # Prepare LLM
    try:
        llm = get_llm(os.getenv("GROQ_API_KEY"))
    except Exception as e:
        st.error(f"LLM initialization failed: {e}")
        return

    # chat history
    if "history" not in st.session_state:
        st.session_state["history"] = []

    for turn in st.session_state["history"]:
        role = turn.get("role", "user")
        with st.chat_message(role):
            st.markdown(turn.get("content", ""))

    # user input
    user_q = st.chat_input("Ask MovieGenie...")

    if not user_q:
        return

    # show user message
    with st.chat_message("user"):
        st.markdown(user_q)
    st.session_state["history"].append({"role": "user", "content": user_q})

    # assistant thinking
    with st.chat_message("assistant"):
        st.markdown("🤔 MovieGenie is generating SQL...")

        plan = ask_llm_for_sql(llm, user_q, schema_text)

        # show model raw output if parsing failed
        if plan.get("raw"):
            st.error("Model did not return strict JSON. Raw model output:")
            st.code(plan["raw"])

        sql = plan.get("sql", "")
        thinking = plan.get("thinking", "")
        followups = plan.get("followups", [])[:3]

        if thinking:
            st.markdown(f"🧠 Agent's thought: {thinking}")

        if sql:
            st.markdown("**Generated SQL:**")
            st.code(sql, language="sql")
        else:
            st.info("No SQL generated by model. You can ask simpler question or check model output above.")

        # run sql only if present
        result = run_sql(conn, sql) if sql else "No SQL to run."

        if isinstance(result, pd.DataFrame):
            if result.empty:
                st.info("Query ran successfully but returned **0 rows**.")
            else:
                # insert row numbers to look like line numbers
                display = result.reset_index(drop=True).copy()
                display.insert(0, "No.", range(1, len(display) + 1))
                st.dataframe(display, use_container_width=True)
        else:
            # result is an error string
            st.error(result)

        # store assistant output in history (brief)
        assistant_content = thinking or (sql if sql else str(result))
        st.session_state["history"].append({"role": "assistant", "content": assistant_content})

    # close DB connection
    try:
        conn.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
