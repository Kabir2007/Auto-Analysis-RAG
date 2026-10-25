"""Auto-Analysis-RAG: Streamlit Web Application Interface."""

from __future__ import annotations

import time

import pandas as pd
import plotly.io as pio
import streamlit as st

import data_tools
import supabase_backend

st.set_page_config(
    page_title="Auto-Analysis-RAG",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for polished look
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        border: 1px solid #e9ecef;
    }
    .flag-pill {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        background-color: #fff3cd;
        color: #856404;
        margin-right: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load the in-memory retrieval index once per Streamlit process. Reloading this
# module would discard the ephemeral, user-scoped collections on every rerun.
try:
    import rag
    rag_import_error = None
except Exception as exc:
    rag = None
    rag_import_error = exc

try:
    import graph
    graph_import_error = None
except Exception as exc:
    graph = None
    graph_import_error = exc


# ---------------------------------------------------------
# Authentication gate and user-scoped application setup
# ---------------------------------------------------------
try:
    secret_values = dict(st.secrets)
except Exception:
    secret_values = {}
supabase_url, supabase_publishable_key = supabase_backend.get_config(secret_values)
if not supabase_url or not supabase_publishable_key:
    st.error("Supabase is not configured. Add SUPABASE_URL and SUPABASE_PUBLISHABLE_KEY to Streamlit secrets to enable secure accounts and private dataset storage.")
    st.code('[supabase]\nSUPABASE_URL = "https://<project-ref>.supabase.co"\nSUPABASE_PUBLISHABLE_KEY = "sb_publishable_..."', language="toml")
    st.stop()
supabase_backend.configure(supabase_url, supabase_publishable_key)


def clear_user_session_state() -> None:
    """Remove credentials and account-specific UI state from this browser session."""
    for state_key in list(st.session_state):
        if state_key.startswith("api_key_") or state_key.startswith("model_"):
            st.session_state.pop(state_key, None)
    for state_key in (
        "supabase_session",
        "selected_dataset_id",
        "chat_history",
        "cleaning_log",
        "show_dataset_delete",
    ):
        st.session_state.pop(state_key, None)


auth_session = st.session_state.get("supabase_session")
if auth_session and auth_session.get("expires_at") and float(auth_session["expires_at"]) <= time.time() + 90:
    try:
        refreshed = supabase_backend.refresh_session(
            supabase_url,
            supabase_publishable_key,
            auth_session.get("refresh_token") or "",
        )
        if refreshed.get("access_token"):
            if not refreshed.get("expires_at"):
                refreshed["expires_at"] = time.time() + float(refreshed.get("expires_in", 3600))
            st.session_state["supabase_session"] = refreshed
            auth_session = refreshed
        else:
            clear_user_session_state()
            auth_session = None
    except Exception:
        clear_user_session_state()
        auth_session = None

if not auth_session or not auth_session.get("access_token") or not auth_session.get("user", {}).get("id"):
    st.title("Auto-Analysis-RAG")
    st.caption("Sign in to access your private datasets.")
    login_tab, signup_tab = st.tabs(["Sign in", "Create account"])
    with login_tab:
        with st.form("supabase_signin"):
            login_email = st.text_input("Email", key="signin_email")
            login_password = st.text_input("Password", type="password", key="signin_password")
            login_submit = st.form_submit_button("Sign in", type="primary")
        if login_submit:
            try:
                session = supabase_backend.sign_in(supabase_url, supabase_publishable_key, login_email.strip(), login_password)
                if session.get("access_token"):
                    clear_user_session_state()
                    if not session.get("expires_at"):
                        session["expires_at"] = time.time() + float(session.get("expires_in", 3600))
                    st.session_state["supabase_session"] = session
                    st.rerun()
                st.error("This account needs email confirmation before sign-in. Check your inbox and try again.")
            except Exception as exc:
                st.error(f"Sign-in failed: {exc}")
    with signup_tab:
        with st.form("supabase_signup"):
            signup_email = st.text_input("Email", key="signup_email")
            signup_password = st.text_input("Password", type="password", key="signup_password")
            signup_confirm = st.text_input("Confirm password", type="password", key="signup_confirm")
            signup_submit = st.form_submit_button("Create account", type="primary")
        if signup_submit:
            if signup_password != signup_confirm:
                st.error("The passwords do not match.")
            elif len(signup_password) < 8:
                st.error("Use a password with at least 8 characters.")
            else:
                try:
                    session = supabase_backend.sign_up(supabase_url, supabase_publishable_key, signup_email.strip(), signup_password)
                    if session.get("access_token"):
                        clear_user_session_state()
                        if not session.get("expires_at"):
                            session["expires_at"] = time.time() + float(session.get("expires_in", 3600))
                        st.session_state["supabase_session"] = session
                        st.rerun()
                    st.success("Account created. Confirm your email, then sign in.")
                except Exception as exc:
                    st.error(f"Account creation failed: {exc}")
    st.stop()

authenticated_user_id = auth_session["user"]["id"]
authenticated_access_token = auth_session["access_token"]

# ---------------------------------------------------------
# Sidebar: per-session provider configuration & user dataset storage
# ---------------------------------------------------------
st.sidebar.title("📁 Auto-Analysis-RAG")
st.sidebar.caption("Deterministic Statistical & ML Analysis")
st.sidebar.caption(f"Signed in as {auth_session['user'].get('email') or 'user'}")
if st.sidebar.button("Sign out", use_container_width=True):
    try:
        supabase_backend.sign_out(supabase_url, supabase_publishable_key, authenticated_access_token)
    except Exception:
        pass
    clear_user_session_state()
    st.rerun()

provider_settings = {
    "Google Gemini": {
        "provider": "google_genai",
        "models": ["google_genai:gemini-3.5-flash", "google_genai:gemini-3.8-flash", "google_genai:gemini-3.5-flash-lite"],
        "label": "Gemini API Key",
        "placeholder": "Enter your Gemini API key",
    },
    "OpenAI": {
        "provider": "openai",
        "models": ["openai:gpt-4o", "openai:gpt-4o-mini"],
        "label": "OpenAI API Key",
        "placeholder": "Enter your OpenAI API key",
    },
    "Anthropic Claude": {
        "provider": "anthropic",
        "models": ["anthropic:claude-sonnet-4-5", "anthropic:claude-3-5-haiku-latest"],
        "label": "Claude API Key",
        "placeholder": "Enter your Anthropic API key",
    },
}
with st.sidebar.expander("🔑 LLM & API Configuration", expanded=True):
    chosen_provider = st.selectbox("Provider", list(provider_settings))
    selected_provider = provider_settings[chosen_provider]
    provider_key_name = f"api_key_{selected_provider['provider']}"
    active_api_key = st.text_input(
        selected_provider["label"],
        value="",
        type="password",
        placeholder=selected_provider["placeholder"],
        key=provider_key_name,
    ).strip()
    if active_api_key:
        selected_model = st.selectbox("Model", selected_provider["models"], key=f"model_{selected_provider['provider']}")
        api_ready = True
        st.success(f"{chosen_provider} is ready · {selected_model.split(':', 1)[-1]}")
        st.caption("Your provider key stays in this signed-in session and is not saved to your account.")
    else:
        selected_model = None
        api_ready = False
        st.caption("Enter your own key for the selected provider. It is used only for this session.")

if not api_ready:
    st.info("Your saved datasets are available in Data & Cleaning and EDA. Add your own API key to use Analysis Chat.")
if graph is None and graph_import_error:
    st.sidebar.error(f"Analysis dependencies failed to load: {graph_import_error}")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or Excel dataset",
    type=["csv", "xlsx"],
    help="Maximum upload size: 50 MB, up to 200,000 rows.",
)

sheet_name = None
if uploaded_file and uploaded_file.name.endswith(".xlsx"):
    try:
        excel_file = pd.ExcelFile(uploaded_file)
        if len(excel_file.sheet_names) > 1:
            sheet_name = st.sidebar.selectbox("Select Excel Sheet", excel_file.sheet_names)
    except Exception:
        pass

if uploaded_file and st.sidebar.button("📥 Load Uploaded Dataset", use_container_width=True):
    try:
        file_bytes = uploaded_file.getvalue()
        ds_id = data_tools.save_upload_for_user(
            file_bytes,
            uploaded_file.name,
            sheet_name=sheet_name,
            owner_id=authenticated_user_id,
            access_token=authenticated_access_token,
        )
        st.session_state["selected_dataset_id"] = ds_id
        st.sidebar.success(f"Loaded '{uploaded_file.name}' successfully!")

        # Index in RAG if available
        if rag is not None:
            with st.spinner("Indexing dataset metadata into RAG..."):
                try:
                    rag.index_kb()
                    raw_df = data_tools.load_user_df(
                        ds_id,
                        version="raw",
                        owner_id=authenticated_user_id,
                        access_token=authenticated_access_token,
                    )
                    profile = data_tools.profile_dataset(raw_df)
                    rag.index_dataset(ds_id, profile, user_id=authenticated_user_id)
                except Exception as e:
                    st.sidebar.warning(f"RAG indexing notice: {e}")

        st.rerun()
    except Exception as e:
        st.sidebar.error(f"Upload error: {e}")

# Dataset Selector
try:
    datasets = data_tools.list_user_datasets(authenticated_user_id, authenticated_access_token)
except Exception as exc:
    datasets = []
    st.sidebar.error(f"Could not load your saved datasets: {exc}")
dataset_options = {d["dataset_id"]: f"{d['filename']} ({d['dataset_id'][:15]}...)" for d in datasets}
datasets_by_id = {d["dataset_id"]: d for d in datasets}

if dataset_options:
    st.sidebar.divider()
    st.sidebar.subheader("Select Dataset")
    current_selection = st.session_state.get("selected_dataset_id")
    default_idx = (
        list(dataset_options.keys()).index(current_selection)
        if current_selection in dataset_options
        else 0
    )
    selected_id = st.sidebar.selectbox(
        "Active Workspace Dataset",
        options=list(dataset_options.keys()),
        format_func=lambda k: dataset_options[k],
        index=default_idx,
    )
    st.session_state["selected_dataset_id"] = selected_id
else:
    st.session_state["selected_dataset_id"] = None

selected_dataset_id = st.session_state.get("selected_dataset_id")
selected_dataset_meta = datasets_by_id.get(selected_dataset_id) if selected_dataset_id else None

if selected_dataset_id and st.sidebar.checkbox("Show dataset deletion controls", key="show_dataset_delete"):
    if st.sidebar.button("Delete this dataset permanently", type="secondary", use_container_width=True):
        try:
            data_tools.delete_user_dataset(
                selected_dataset_id,
                owner_id=authenticated_user_id,
                access_token=authenticated_access_token,
            )
            if rag is not None:
                rag.delete_dataset(selected_dataset_id, user_id=authenticated_user_id)
            st.session_state.pop("chat_history", None)
            st.session_state.pop("selected_dataset_id", None)
            st.sidebar.success("Dataset and saved files deleted.")
            st.rerun()
        except Exception as exc:
            st.sidebar.error(f"Dataset deletion failed: {exc}")


# ---------------------------------------------------------
# Main UI Layout
# ---------------------------------------------------------
tab_data, tab_eda, tab_chat = st.tabs(["📋 Data & Cleaning", "📈 Exploratory Data Analysis (EDA)", "💬 Analysis Chat"])

if not selected_dataset_id:
    with tab_data:
        st.info("👋 Welcome! Please upload a `.csv` or `.xlsx` file in the sidebar to begin analysis.")
    with tab_eda:
        st.info("No active dataset selected. Upload or choose a dataset to view EDA.")
    with tab_chat:
        st.info("No active dataset selected. Upload or choose a dataset to chat.")

else:
    # Load dataset
    raw_df = data_tools.load_user_df(
        selected_dataset_id,
        version="raw",
        owner_id=authenticated_user_id,
        access_token=authenticated_access_token,
    )
    has_clean = bool(selected_dataset_meta and selected_dataset_meta.get("has_clean"))
    active_df = data_tools.load_user_df(
        selected_dataset_id,
        version="active",
        owner_id=authenticated_user_id,
        access_token=authenticated_access_token,
    )
    # Chroma is intentionally ephemeral. Rebuild this user's small schema index
    # on first access after an app restart, while keeping it isolated by owner.
    if rag is not None and not rag.has_dataset(selected_dataset_id, user_id=authenticated_user_id):
        try:
            rag.index_kb()
            rag.index_dataset(
                selected_dataset_id,
                data_tools.profile_dataset(active_df),
                cleaning_log=data_tools.get_user_clean_log(
                    selected_dataset_id,
                    owner_id=authenticated_user_id,
                    access_token=authenticated_access_token,
                ),
                user_id=authenticated_user_id,
            )
        except Exception as exc:
            st.sidebar.warning(f"Analysis context indexing notice: {exc}")

    # =========================================================
    # TAB 1: DATA & CLEANING
    # =========================================================
    with tab_data:
        st.subheader(f"Dataset Overview: `{selected_dataset_id}`")
        if has_clean:
            st.success("✨ Active version: **Cleaned dataset** (auto-cleaning applied).")
        else:
            st.info("ℹ️ Active version: **Raw dataset** (uncleaned). Click 'Run Auto-Clean' below to clean and impute.")

        # Metric row
        total_cells = active_df.shape[0] * active_df.shape[1]
        missing_count = int(active_df.isna().sum().sum())
        missing_pct = (missing_count / total_cells * 100) if total_cells > 0 else 0.0
        dup_count = int(active_df.duplicated().sum())

        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        mcol1.metric("Rows", f"{active_df.shape[0]:,}")
        mcol2.metric("Columns", f"{active_df.shape[1]:,}")
        mcol3.metric("Missing Values", f"{missing_pct:.2f}%", f"{missing_count:,} cells")
        mcol4.metric("Exact Duplicates", f"{dup_count:,}")

        st.divider()

        # 100-row preview
        with st.expander("🔍 View Dataset Preview (first 100 rows)", expanded=True):
            st.dataframe(active_df.head(100), use_container_width=True)

        # Profile table
        st.subheader("📊 Column Profiling & Quality Flags")
        profile = data_tools.profile_dataset(active_df)

        prof_rows = []
        for col_name, col_meta in profile["columns"].items():
            stats_display = ""
            if col_meta["semantic_type"] == "numeric":
                stats_display = f"min={col_meta.get('min', 0):.2f}, mean={col_meta.get('mean', 0):.2f}, max={col_meta.get('max', 0):.2f}"
            elif col_meta["semantic_type"] == "categorical":
                top = col_meta.get("top_values", {})
                stats_display = ", ".join([f"{k} ({v})" for k, v in list(top.items())[:3]])
            elif col_meta["semantic_type"] == "datetime":
                stats_display = f"{col_meta.get('min_date', '')} to {col_meta.get('max_date', '')}"

            prof_rows.append({
                "Column": col_name,
                "Dtype": col_meta["dtype"],
                "Semantic Type": col_meta["semantic_type"],
                "Missing Count": col_meta["missing_count"],
                "Missing %": f"{col_meta['missing_pct']*100:.2f}%",
                "Uniques": col_meta["n_unique"],
                "Summary / Top Values": stats_display,
            })

        st.dataframe(pd.DataFrame(prof_rows), use_container_width=True)

        if profile["flags"]:
            st.warning(f"Found {len(profile['flags'])} potential data quality flag(s):")
            for flag in profile["flags"]:
                st.markdown(f"- **{flag['column']}** (`{flag['flag']}`): {flag['detail']}")

        st.divider()

        # Auto-Clean Section
        st.subheader("🧹 Automated Cleaning & Imputation")
        col_clean_opt, col_clean_btn = st.columns([2, 1])
        with col_clean_opt:
            cap_outliers = st.checkbox(
                "Winsorize outliers to 1st/99th percentiles (Default: flag only via 1.5*IQR)",
                value=False,
            )
            drop_threshold = st.slider(
                "Drop columns with missing share exceeding",
                min_value=0.2,
                max_value=0.9,
                value=0.6,
                step=0.05,
            )

        with col_clean_btn:
            run_clean = st.button("🚀 Run Auto-Clean Pipeline", type="primary", use_container_width=True)

        if run_clean:
            with st.spinner("Executing rule-based cleaning..."):
                raw_to_clean = data_tools.load_user_df(
                    selected_dataset_id,
                    version="raw",
                    owner_id=authenticated_user_id,
                    access_token=authenticated_access_token,
                )
                cleaned_df, cleaning_log = data_tools.clean_dataset(
                    raw_to_clean,
                    drop_col_missing_threshold=drop_threshold,
                    cap_outliers=cap_outliers,
                )
                data_tools.save_user_clean(
                    selected_dataset_id,
                    cleaned_df,
                    cleaning_log,
                    owner_id=authenticated_user_id,
                    access_token=authenticated_access_token,
                )

                # Re-index RAG collection
                if rag is not None:
                    try:
                        clean_prof = data_tools.profile_dataset(cleaned_df)
                        rag.index_dataset(
                            selected_dataset_id,
                            clean_prof,
                            cleaning_log=cleaning_log,
                            user_id=authenticated_user_id,
                        )
                    except Exception as e:
                        st.warning(f"RAG re-indexing note: {e}")

                st.success("✅ Cleaned dataset saved to your private cloud storage.")
                st.session_state["cleaning_log"] = cleaning_log
                st.rerun()

        # Display cleaning log and downloads if clean exists
        if has_clean:
            try:
                saved_log = data_tools.get_user_clean_log(
                    selected_dataset_id,
                    owner_id=authenticated_user_id,
                    access_token=authenticated_access_token,
                )
                if saved_log:
                    st.write("#### 📜 Cleaning Audit Log")
                    st.dataframe(pd.DataFrame(saved_log), use_container_width=True)
            except Exception as exc:
                st.warning(f"Could not load the cleaning audit log: {exc}")

            st.write("#### 💾 Export Cleaned & Encoded Data")
            dcol1, dcol2 = st.columns(2)
            clean_csv = active_df.to_csv(index=False).encode("utf-8")
            dcol1.download_button(
                label="📥 Download Cleaned CSV",
                data=clean_csv,
                file_name=f"{selected_dataset_id}_clean.csv",
                mime="text/csv",
                use_container_width=True,
            )

            encoded_df = data_tools.encode_for_export(active_df)
            encoded_csv = encoded_df.to_csv(index=False).encode("utf-8")
            dcol2.download_button(
                label="📥 Download Model-Ready Encoded CSV",
                data=encoded_csv,
                file_name=f"{selected_dataset_id}_encoded.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # =========================================================
    # TAB 2: EXPLORATORY DATA ANALYSIS (EDA)
    # =========================================================
    with tab_eda:
        st.subheader("📈 Automated Exploratory Data Analysis")

        # Cache by account, dataset, and metadata revision so separate users
        # cannot share cached results even if they upload the same filename.
        cache_version = (selected_dataset_meta or {}).get("updated_at") or (selected_dataset_meta or {}).get("uploaded_at") or ""

        @st.cache_data(show_spinner=False)
        def _get_cached_eda(owner_id: str, ds_id: str, data_version: str, _access_token: str) -> dict:
            df = data_tools.load_user_df(
                ds_id,
                version="active",
                owner_id=owner_id,
                access_token=_access_token,
            )
            return data_tools.run_eda(df)

        with st.spinner("Generating EDA distributions, summaries, and charts..."):
            eda_results = _get_cached_eda(
                authenticated_user_id,
                selected_dataset_id,
                cache_version,
                authenticated_access_token,
            )

        # Insights banner
        if eda_results["insights"]:
            st.write("#### 💡 Key Insights & Anomalies Detected")
            for insight in eda_results["insights"]:
                st.info(f"• {insight}")

        st.divider()

        # EDA Tables
        st.write("#### 📑 Summary Tables")
        for table_name, table_df in eda_results["tables"].items():
            with st.expander(f"📋 {table_name.replace('_', ' ').title()}", expanded=(table_name == "numeric_summary")):
                st.dataframe(table_df, use_container_width=True)

        st.divider()

        # EDA Figures in a 2-column grid
        st.write("#### 📊 Exploratory Visualizations")
        figures = eda_results["figures"]
        fig_names = list(figures.keys())

        if fig_names:
            col_left, col_right = st.columns(2)
            for i, name in enumerate(fig_names):
                target_col = col_left if i % 2 == 0 else col_right
                with target_col:
                    st.plotly_chart(figures[name], use_container_width=True)
        else:
            st.info("No figures generated for this dataset.")

    # =========================================================
    # TAB 3: ANALYSIS CHAT
    # =========================================================
    with tab_chat:
        st.subheader("💬 Statistical & Machine Learning Decision Chat")
        st.caption("Ask questions to test business hypotheses, run regressions/classifications, or forecast trends.")

        # Initialize session chat history for dataset
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = {}
        if selected_dataset_id not in st.session_state["chat_history"]:
            st.session_state["chat_history"][selected_dataset_id] = []

        chat_turns = st.session_state["chat_history"][selected_dataset_id]

        # Render message history
        for turn in chat_turns:
            with st.chat_message(turn["role"]):
                st.markdown(turn["content"])

                # Render tables if present
                for tbl in turn.get("tables", []):
                    st.write(f"**{tbl.get('title', 'Table')}**")
                    if isinstance(tbl.get("records"), list):
                        st.dataframe(pd.DataFrame(tbl["records"]), use_container_width=True)

                # Render figures if present
                for fig_json in turn.get("figures", []):
                    try:
                        fig_obj = pio.from_json(fig_json)
                        st.plotly_chart(fig_obj, use_container_width=True)
                    except Exception:
                        pass

                # Render 'How I got this' expander
                if turn.get("role") == "assistant" and (turn.get("trace") or turn.get("plan")):
                    with st.expander("🔍 How I got this (Analysis Plan & Execution Trace)"):
                        if turn.get("plan"):
                            st.write("**Plan:**")
                            st.json(turn["plan"])
                        if turn.get("trace"):
                            st.write("**Execution Trace:**")
                            for tr in turn["trace"]:
                                st.text(f"• {tr}")

        # Chat Input
        user_query = st.chat_input("Ask a question about this dataset...")
        query_to_run = user_query

        if query_to_run:
            if not api_ready:
                st.error("Cannot run chat queries without an API key. Enter your own provider key in the sidebar.")
            elif graph is None:
                st.error(f"Analysis module could not be initialized: {graph_import_error or 'unknown import error'}")
                st.code(r".\venv\Scripts\python.exe -m streamlit run app.py", language="powershell")
                st.caption("Stop the running Streamlit server, then start it with the project virtual environment so it uses the installed requirements.")
            else:
                # Append user query to chat history
                chat_turns.append({"role": "user", "content": query_to_run})
                with st.chat_message("user"):
                    st.markdown(query_to_run)

                with st.chat_message("assistant"):
                    with st.spinner("Analyzing dataset with the selected built-in function..."):
                        try:
                            response = graph.run_query(
                                dataset_id=selected_dataset_id,
                                question=query_to_run,
                                chat_history=chat_turns[-6:],
                                user_id=authenticated_user_id,
                                access_token=authenticated_access_token,
                                provider=selected_provider["provider"],
                                model=selected_model,
                                api_key=active_api_key,
                            )
                            answer_text = response.get("answer", "No answer generated.")
                            st.markdown(answer_text)

                            # Tables
                            resp_tables = response.get("tables", [])
                            for tbl in resp_tables:
                                st.write(f"**{tbl.get('title', 'Table')}**")
                                if isinstance(tbl.get("records"), list):
                                    st.dataframe(pd.DataFrame(tbl["records"]), use_container_width=True)

                            # Figures
                            resp_figures = response.get("figures", [])
                            for fig_json in resp_figures:
                                try:
                                    fig_obj = pio.from_json(fig_json)
                                    st.plotly_chart(fig_obj, use_container_width=True)
                                except Exception:
                                    pass

                            # Trace / Plan / Code
                            with st.expander("🔍 How I got this (Analysis Plan & Execution Trace)"):
                                if response.get("plan"):
                                    st.write("**Plan:**")
                                    st.json(response["plan"])
                                if response.get("trace"):
                                    st.write("**Execution Trace:**")
                                    for tr in response["trace"]:
                                        st.text(f"• {tr}")

                            # Store to history
                            chat_turns.append({
                                "role": "assistant",
                                "content": answer_text,
                                "tables": resp_tables,
                                "figures": resp_figures,
                                "plan": response.get("plan"),
                                "trace": response.get("trace"),
                            })

                        except Exception as e:
                            err_msg = f"An error occurred while executing the analysis: {e}"
                            st.error(err_msg)
                            chat_turns.append({"role": "assistant", "content": err_msg})

