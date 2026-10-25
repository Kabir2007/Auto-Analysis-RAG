"""Autonomous Data Analysis Agent: Tool-calling pipeline for deterministic statistics and ML."""

from __future__ import annotations

import logging
from typing import Any, Literal, Optional

import plotly.io as pio
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

import data_tools
import rag
import stats_ml_tools

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("agent")


def get_llm(model_override: str, api_key: str, provider: Optional[str] = None):
    """Instantiate one user's selected provider/model without process-global keys."""
    if not api_key:
        raise ValueError("Enter your own API key for the selected provider.")
    raw_spec = model_override.strip()
    if not raw_spec:
        raise ValueError("Choose a model for the selected provider.")

    DEPRECATED_MAP = {
        "gemini-2.5-flash": "gemini-3.5-flash-lite",
        "gemini-2.5-flash-lite": "gemini-3.5-flash-lite",
        "gemini-1.5-flash": "gemini-3.5-flash-lite",
        "gemini-1.5-pro": "gemini-3.5-flash-lite",
        "gemini-3.5-flash": "gemini-3.5-flash-lite",
    }

    if ":" in raw_spec:
        model_provider, model = raw_spec.split(":", 1)
    else:
        model_provider, model = provider or "", raw_spec
    provider_aliases = {"google": "google_genai", "gemini": "google_genai", "claude": "anthropic"}
    model_provider = provider_aliases.get(model_provider, model_provider)
    if not model_provider:
        raise ValueError("The selected model must include its provider.")
    model = DEPRECATED_MAP.get(model, model)
    return init_chat_model(model, model_provider=model_provider, api_key=api_key, temperature=0)


class AgentDecision(BaseModel):
    """The agent's decision on which statistical function or tool to call and which columns to use."""

    thought: str = Field(
        ...,
        description="Reasoning explaining how to answer the prompt using the available columns and statistical functions.",
    )
    tool: Literal[
        "descriptive_statistics",
        "cohort_decision_test",
        "compare_groups",
        "correlation",
        "association",
        "ab_proportion",
        "train_predictive_model",
        "time_series_forecast",
        "generate_chart",
        "direct_answer",
    ] = Field(
        ...,
        description=(
            "The function to call: "
            "'descriptive_statistics' for averages, counts, totals, and grouped summaries without inferential claims; "
            "'cohort_decision_test' for testing a top/bottom percentile cohort against the remainder on an independent outcome; "
            "'compare_groups' for differences across categories (t-test/ANOVA/Mann-Whitney/Kruskal-Wallis); "
            "'correlation' for relationship between two numeric variables; "
            "'association' for relationship between categorical variables (Chi-Square); "
            "'ab_proportion' for conversion rates across two groups; "
            "'train_predictive_model' to predict a target from features and get feature importance; "
            "'time_series_forecast' to forecast future trend over time; "
            "'generate_chart' for visualizations requested by the user; "
            "'direct_answer' for conceptual explanations or general questions without computing on data."
        ),
    )
    target_col: Optional[str] = Field(None, description="Primary column (metric or prediction target) to analyze")
    group_col: Optional[str] = Field(None, description="Grouping column or secondary column")
    metric_columns: list[str] = Field(default_factory=list, description="Metric columns for a descriptive summary")
    group_columns: list[str] = Field(default_factory=list, description="Columns by which descriptive summaries should be grouped")
    descriptive_operation: str = Field("mean", description="Descriptive aggregation to compute; relevant only for descriptive_statistics")
    descriptive_mode: str = Field("column_aggregate", description="Descriptive calculation mode; relevant only for descriptive_statistics")
    entity_column: Optional[str] = Field(None, description="Entity identifier for item counts, such as employee ID")
    item_column: Optional[str] = Field(None, description="Item identifier to count per entity, such as project ID; omit to count rows")
    count_distinct_items: bool = Field(True, description="Count distinct item IDs when computing items per entity")
    ranking_column: Optional[str] = Field(None, description="Numeric column used to define a top or bottom percentile cohort")
    cohort_percentile: float = Field(5.0, description="Top or bottom cohort percentage; relevant only for cohort_decision_test")
    cohort_direction: str = Field("bottom", description="Which percentile cohort to test; relevant only for cohort_decision_test")
    features: Optional[list[str]] = Field(None, description="List of feature columns for ML modeling")
    time_col: Optional[str] = Field(None, description="Date/timestamp column for time series forecasting")
    task_type: str = Field("auto", description="Supervised prediction task; relevant only for train_predictive_model")
    horizon: int = Field(30, description="Forecast periods requested by the user; relevant only for time_series_forecast")
    chart_type: Optional[str] = Field(None, description="Requested chart type; relevant only when a chart is selected")
    chart_x: Optional[str] = Field(None, description="Dataset x-axis column")
    chart_y: Optional[str] = Field(None, description="Optional dataset y-axis column")
    chart_color: Optional[str] = Field(None, description="Optional grouping column")
    chart_agg: Optional[str] = Field(None, description="Chart aggregation; relevant only when a chart is selected")


class DescriptiveSelection(BaseModel):
    """Model-selected fields and operation for a descriptive request."""

    mode: Literal["column_aggregate", "mean_items_per_entity"] = Field(
        description="Use column_aggregate for an existing per-entity metric; use mean_items_per_entity only when rows are item-level records."
    )
    operation: Literal["mean", "median", "sum", "count", "min", "max", "std", "nunique"] = Field(
        description="Aggregation requested by the user; use mean for average."
    )
    metric_columns: list[str] = Field(
        description="Exact numeric dataset columns to aggregate in column_aggregate mode. Must not be empty in that mode."
    )
    group_columns: list[str] = Field(
        description="Exact dataset columns requested for a grouped breakdown; empty only when no breakdown was requested."
    )
    entity_column: Optional[str] = Field(
        description="Exact entity identifier only for item-level records requiring per-entity item counts."
    )
    item_column: Optional[str] = Field(
        description="Exact item identifier for item-level records, when present; otherwise omit and count rows."
    )
    count_distinct_items: bool = Field(
        description="Whether duplicate item IDs should count once per entity."
    )



# -------------------------------------------------------------
# Core Agent Pipeline
# -------------------------------------------------------------
def _resolve_column(col_name: Optional[str], available_cols: list[str]) -> Optional[str]:
    """Resolve a model-selected column to an existing dataset column."""
    if not col_name:
        return None
    if col_name in available_cols:
        return col_name

    lowered = col_name.casefold()
    for column in available_cols:
        if column.casefold() == lowered:
            return column

    # Do not silently substitute a vaguely similar field. A wrong metric can
    # produce a plausible but incorrect answer; the planner can be asked to
    # repair an invalid selection using the exact schema instead.
    return col_name


def _descriptive_plan_is_incomplete(decision: AgentDecision, available_cols: list[str]) -> bool:
    """Check that the model selected the fields needed by its descriptive mode."""
    if decision.descriptive_mode == "mean_items_per_entity":
        return _resolve_column(decision.entity_column, available_cols) not in available_cols
    metric_columns = [
        _resolve_column(name, available_cols)
        for name in decision.metric_columns
        if _resolve_column(name, available_cols) in available_cols
    ]
    return not metric_columns and _resolve_column(decision.target_col, available_cols) not in available_cols


def run_query(
    dataset_id: str,
    question: str,
    chat_history: Optional[list[dict]] = None,
    *,
    user_id: str,
    access_token: str,
    provider: str,
    model: str,
    api_key: str,
) -> dict:
    """Run autonomous data analysis agent: prompt + columns -> tool selection -> execution -> grounded synthesis."""
    # Ensure fresh modules in long-running environments like Streamlit
    import importlib
    try:
        importlib.reload(stats_ml_tools)
        importlib.reload(data_tools)
    except Exception:
        pass

    logger.info("Agent initiated for dataset '%s'", dataset_id)
    trace: list[str] = [f"[agent] Processing query: '{question}'"]

    df = data_tools.load_user_df(
        dataset_id,
        version="active",
        owner_id=user_id,
        access_token=access_token,
    )
    available_cols = list(df.columns)
    cols_summary = ", ".join(
        f"{column} (dtype={df[column].dtype}, "
        f"semantic_type={data_tools._detect_semantic_type(df[column])}, "
        f"unique_values={df[column].nunique(dropna=True)})"
        for column in available_cols
    )

    # 1. Retrieve RAG context if relevant
    rag_context = rag.retrieve(dataset_id, question, k_ds=3, k_kb=2, k_mem=2, user_id=user_id)
    rag_text = "\n".join([f"- [{c['source']}] {c['text'][:250]}" for c in rag_context]) if rag_context else "None"

    # 2. Agent Prompt with Columns & Tool Catalog
    system_prompt = f"""You are the analysis planner for a dataset analysis application. Select exactly one built-in analysis function, identify the relevant dataset columns, and provide its settings. The tools perform all calculations; you do not calculate results or generate code.

ACTIVE DATASET SCHEMA ({len(available_cols)} columns; {len(df)} rows):
{cols_summary}

RETRIEVED DATASET / STATISTICAL CONTEXT:
{rag_text}

ROUTE BY USER INTENT — APPLY THESE RULES IN ORDER:
1. Explicit request to forecast future values over time -> time_series_forecast. Choose the target, timestamp column, and requested horizon. Do not use a classifier or a hypothesis test for a forecast.
2. Request for a descriptive quantity (average, count, total, median, min/max, standard deviation, or breakdown by department/group) -> descriptive_statistics. This includes comparisons that ask only for values, such as "give me the average overall and by department". Do not run a hypothesis test just because groups are mentioned. You MUST select at least one exact metric column for column_aggregate mode. A target_col can also be used as the metric, but do not leave both empty.
3. Request for a chart/plot/visualization -> generate_chart, with chart type and exact dataset columns. If they also explicitly request a hypothesis test, select the test tool and populate the optional chart fields too.
4. Explicit prediction/classification request about unknown/future outcomes -> train_predictive_model. Select classification or regression from the target's meaning/type and use relevant features. A question asking whether to take an action based on existing evidence is not itself a request to train a predictive model.
5. A decision, recommendation, or claim about whether a difference/effect/relationship is real or meaningful -> use a hypothesis-test function before drawing a conclusion. Do not route a decision question to ML merely because the prompt mentions a predictive model, model score, or ranking. If the decision is about acting on a top/bottom percentile cohort, select cohort_decision_test and compare the cohort to the rest using a separate, relevant outcome column. Never use the same column both to define and evaluate the cohort. If no independent, decision-relevant outcome column exists, use direct_answer and state that the dataset does not support that inference; do not substitute a predictive model or invent evidence.
6. Purely conceptual/statistical explanation with no dataset calculation -> direct_answer.

SELECT THE HYPOTHESIS TEST FROM COLUMN TYPES AND QUESTION:
- Numeric outcome across a categorical grouping -> compare_groups. The deterministic function checks assumptions and selects a suitable test (for example Welch/Student t, Mann–Whitney, ANOVA, or Kruskal–Wallis).
- Two numeric measures whose relationship is in question -> correlation.
- Two categorical measures whose association is in question -> association.
- Binary success/conversion outcome across exactly two groups -> ab_proportion.
- Percentile cohort vs remainder -> cohort_decision_test chooses group comparison for numeric outcomes, two-proportion testing for 0/1 outcomes, or association testing for categorical outcomes.
The selected statistical function computes statistics, p-values, confidence intervals/effect sizes where supported, and warnings. Do not claim causation from observational data.

AVAILABLE BUILT-IN FUNCTIONS:
- descriptive_statistics: basic overall/grouped aggregations; also mean_items_per_entity when rows represent items assigned to entities. Always select exact metric_columns for column_aggregate mode. Use mean_items_per_entity only when rows are item-level records that need to be counted per entity. When the dataset has one row per entity and already has a numeric measure for the requested quantity, average that measure with column_aggregate. For that mode identify the entity ID, item ID if present, and grouping columns.
- cohort_decision_test: numeric ranking column, top/bottom percentile, and an independent outcome column.
- compare_groups: numeric outcome and categorical group.
- correlation: two numeric columns.
- association: two categorical columns.
- ab_proportion: binary outcome and two-level group.
- train_predictive_model: explicit supervised prediction/classification request.
- time_series_forecast: explicit future time-based forecast.
- generate_chart: user-requested chart using chart_type, chart_x, optional chart_y/color, and aggregation.
- direct_answer: conceptual response or insufficient data/columns for a valid computation.
The active dataset is passed only to registered functions. You receive its schema and metadata, not raw rows. Use exact column names. Do not generate Python or perform arithmetic.

ROUTING EXAMPLES (column names below are roles; always substitute exact fields from the active schema):
A. User asks for an average per entity overall and by category. If each row is one entity and the schema has a numeric measure, use descriptive_statistics with column_aggregate, that measure, mean, and the requested category field. If each row is an item assigned to an entity, use mean_items_per_entity with the entity ID, item ID when available, and requested category field. Decide from schema and row/unique counts. No hypothesis test.
B. User asks whether a numeric outcome differs significantly across categories. Use compare_groups with the numeric outcome and categorical group field.
C. User asks whether to act on a top/bottom percentile cohort. Use cohort_decision_test with a continuous ranking field and an independent, decision-relevant outcome. Do not train a model. If there is no suitable independent outcome, choose direct_answer and explain the limitation.
D. User asks to predict an unknown outcome. Use train_predictive_model with the target and relevant eligible predictors; choose classification or regression from target meaning/type.
E. User asks to forecast future values. Use time_series_forecast with the requested numeric target, dataset timestamp, and requested horizon.
F. User asks for a chart. Use generate_chart with the requested chart form and exact schema fields for axes/grouping/aggregation.

Return only a valid structured plan matching the AgentDecision schema."""

    llm = get_llm(model_override=model, api_key=api_key, provider=provider)
    try:
        structured_agent = llm.with_structured_output(AgentDecision)
        decision: AgentDecision = structured_agent.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=question),
        ])
    except Exception as e:
        logger.error(f"Analysis planning failed: {e}", exc_info=True)
        raise RuntimeError(f"Could not create an analysis plan: {e}") from e

    # Keep routing separate from field selection. For a descriptive request,
    # focus a second structured model pass on mapping the requested measure and
    # breakdown to this dataset's exact schema before deterministic execution.
    if decision.tool == "descriptive_statistics":
        selection_prompt = f"""You select dataset fields for a descriptive calculation. The user's request and dataset schema determine the fields; never rely on hardcoded column names or invent fields.

DATASET SHAPE: {len(df)} rows x {len(available_cols)} columns
EXACT SCHEMA:
{cols_summary}

Selection rules:
- Map the user's requested concept to the most relevant existing dataset column(s).
- For a requested average, choose operation=mean.
- Use column_aggregate when a row already represents the entity being summarized and the requested measure exists as a column. Cardinality helps identify this: an identifier with unique_values equal to row count indicates one record per entity.
- Use mean_items_per_entity only when rows represent individual items/assignments that must first be counted for each entity.
- Include every requested grouping/breakdown field in group_columns, using exact dataset column names.
- For column_aggregate, metric_columns must contain at least one exact dataset column. Do not substitute an ID, grouping field, or unrelated numeric field.
- If the request cannot be mapped to the schema, return empty metric_columns and the closest accurate mode; do not fabricate a calculation.

Initial route rationale: {decision.thought}
"""
        try:
            selector = llm.with_structured_output(DescriptiveSelection)
            selection: DescriptiveSelection = selector.invoke([
                SystemMessage(content=selection_prompt),
                HumanMessage(content=question),
            ])
            decision = decision.model_copy(update={
                "descriptive_mode": selection.mode,
                "descriptive_operation": selection.operation,
                "metric_columns": selection.metric_columns,
                "group_columns": selection.group_columns,
                "entity_column": selection.entity_column,
                "item_column": selection.item_column,
                "count_distinct_items": selection.count_distinct_items,
            })
        except Exception as e:
            logger.warning("Focused descriptive column selection failed; trying planner repair: %s", e)

    # Incomplete model output gets one repair attempt with exact validation
    # feedback; do not let a missing metric become a generic tool error.
    if decision.tool == "descriptive_statistics" and _descriptive_plan_is_incomplete(decision, available_cols):
        repair_prompt = system_prompt + f"""

PLAN REPAIR REQUIRED:
Your prior descriptive plan did not identify a usable metric/entity column.
Choose the exact field(s) from the ACTIVE DATASET SCHEMA above based on the user's request.
For an existing per-entity measure, use descriptive_mode=column_aggregate and populate
metric_columns (and group_columns for any requested breakdown). Do not assume an entity-count
calculation when the dataset already has one row per entity. Use mean_items_per_entity only
when records represent individual items and identify the entity column (and item column if present).
Never return an empty metric_columns list in column_aggregate mode or an entity_column that is absent.
Keep the user's requested analysis route as descriptive_statistics; do not invent a test.
Prior plan: {decision.model_dump()}
"""
        try:
            repaired: AgentDecision = structured_agent.invoke([
                SystemMessage(content=repair_prompt),
                HumanMessage(content=question),
            ])
            if repaired.tool == "descriptive_statistics" and not _descriptive_plan_is_incomplete(repaired, available_cols):
                decision = repaired
            else:
                logger.warning("Planner repair remained incomplete; retaining original plan for user-safe validation.")
        except Exception as e:
            logger.warning("Planner repair failed; retaining original plan for user-safe validation: %s", e)

    if decision.tool == "descriptive_statistics" and _descriptive_plan_is_incomplete(decision, available_cols):
        numeric_columns = [
            column for column in available_cols
            if data_tools._detect_semantic_type(df[column]) == "numeric"
        ]
        trace.append("[agent] Could not map the descriptive metric to an exact dataset column after model selection and repair.")
        return {
            "answer": (
                "I couldn't map the requested measure to a dataset column, so I did not calculate a potentially misleading result. "
                f"Numeric columns available for aggregation: {', '.join(numeric_columns) or 'none'}. "
                "Please specify which measure to summarize."
            ),
            "tables": [],
            "figures": [],
            "plan": decision.model_dump(),
            "trace": trace,
            "code": None,
        }

    trace.append(f"[agent] Selected tool: '{decision.tool}' | Thought: {decision.thought}")

    # Resolve column names cleanly
    target_col = _resolve_column(decision.target_col, available_cols)
    group_col = _resolve_column(decision.group_col, available_cols)

    # 3. Deterministic Tool Execution
    tool_output: dict[str, Any] = {}
    tables: list[dict] = []
    figures: list[str] = []

    try:
        if decision.tool == "cohort_decision_test":
            ranking_column = _resolve_column(decision.ranking_column, available_cols)
            if not ranking_column or not target_col:
                tool_output = {"error": "A cohort test requires both a ranking column and an independent outcome column."}
            else:
                tool_output = stats_ml_tools.run_cohort_hypothesis_test(
                    df,
                    ranking_column=ranking_column,
                    outcome_column=target_col,
                    percentile=decision.cohort_percentile,
                    direction=decision.cohort_direction,
                )
                if tool_output.get("test_result", {}).get("post_hoc"):
                    tables.append({"title": "Pairwise Post-Hoc Comparisons", "records": tool_output["test_result"]["post_hoc"]})

        elif decision.tool == "descriptive_statistics":
            metric_columns = [
                column for name in decision.metric_columns
                if (column := _resolve_column(name, available_cols)) in available_cols
            ]
            if not metric_columns and target_col in available_cols:
                metric_columns = [target_col]
            group_columns = [
                column for name in decision.group_columns
                if (column := _resolve_column(name, available_cols)) in available_cols
            ]
            if not group_columns and group_col in available_cols:
                group_columns = [group_col]
            entity_column = _resolve_column(decision.entity_column, available_cols)
            item_column = _resolve_column(decision.item_column, available_cols)
            if decision.descriptive_mode == "mean_items_per_entity" and not entity_column:
                tool_output = {"error": "Per-entity averages require an entity ID column from the dataset."}
            else:
                tool_output = data_tools.run_descriptive_statistics(
                    df,
                    operation=decision.descriptive_operation,
                    metric_columns=metric_columns,
                    group_columns=group_columns,
                    entity_column=entity_column if decision.descriptive_mode == "mean_items_per_entity" else None,
                    item_column=item_column if decision.descriptive_mode == "mean_items_per_entity" else None,
                    count_distinct_items=decision.count_distinct_items,
                )
            if tool_output.get("summary"):
                tables.append({"title": "Descriptive Summary", "records": tool_output["summary"]})

        elif decision.tool in ("compare_groups", "correlation", "association", "ab_proportion"):
            stat_req = {
                "test_type": decision.tool,
                "target": target_col,
                "group_col": group_col,
                "second_col": group_col,
                "alpha": 0.05,
            }
            tool_output = stats_ml_tools.run_stat_test(df, stat_req)
            if tool_output.get("post_hoc"):
                tables.append({"title": "Pairwise Post-Hoc Comparisons", "records": tool_output["post_hoc"]})

        elif decision.tool == "train_predictive_model":
            feat_cols = [_resolve_column(f, available_cols) for f in (decision.features or []) if _resolve_column(f, available_cols)]
            ml_req = {"task": decision.task_type, "target": target_col, "features": feat_cols or None}
            tool_output = stats_ml_tools.train_and_evaluate(df, ml_req)
            if tool_output.get("figure_json"):
                figures.append(tool_output["figure_json"])
            if tool_output.get("tables"):
                for tname, trecords in tool_output["tables"].items():
                    tables.append({"title": tname.replace("_", " ").title(), "records": trecords})

        elif decision.tool == "time_series_forecast":
            time_col = _resolve_column(decision.time_col, available_cols)
            if not 1 <= decision.horizon <= 365:
                tool_output = {"error": "Forecast horizon must be between 1 and 365 periods."}
            else:
                ml_req = {"task": "forecast", "target": target_col, "time_col": time_col, "horizon": decision.horizon}
                tool_output = stats_ml_tools.train_and_evaluate(df, ml_req)
                if tool_output.get("figure_json"):
                    figures.append(tool_output["figure_json"])
                if tool_output.get("tables"):
                    for tname, trecords in tool_output["tables"].items():
                        tables.append({"title": tname.replace("_", " ").title(), "records": trecords})

        elif decision.tool == "generate_chart":
            chart_x = _resolve_column(decision.chart_x, available_cols)
            chart_y = _resolve_column(decision.chart_y, available_cols)
            chart_color = _resolve_column(decision.chart_color, available_cols)
            if not chart_x or not decision.chart_type:
                tool_output = {"error": "Chart generation requires a chart type and valid x-axis column."}
            else:
                chart_spec = {"chart_type": decision.chart_type, "x": chart_x, "y": chart_y, "color": chart_color, "agg": decision.chart_agg, "title": question}
                fig = data_tools.build_chart(df, chart_spec)
                figures.append(pio.to_json(fig))
                tool_output = {"chart": chart_spec, "rows_used": int(len(df))}
        else:  # direct_answer
            tool_output = {"context": rag_text, "dataset_columns": available_cols}

        # A requested chart can accompany a statistical or ML analysis. The model
        # chooses the chart and columns; the built-in chart builder validates and renders it.
        if decision.chart_type and decision.tool != "generate_chart":
            chart_x = _resolve_column(decision.chart_x, available_cols)
            chart_y = _resolve_column(decision.chart_y, available_cols)
            chart_color = _resolve_column(decision.chart_color, available_cols)
            if not chart_x:
                tool_output = {"error": "Chart generation requires a valid x-axis column."}
            else:
                chart_spec = {"chart_type": decision.chart_type, "x": chart_x, "y": chart_y, "color": chart_color, "agg": decision.chart_agg, "title": question}
                figures.append(pio.to_json(data_tools.build_chart(df, chart_spec)))

    except Exception as e:
        logger.error(f"Tool execution failed: {e}", exc_info=True)
        tool_output = {"error": f"Tool execution failed: {e}"}

    # If the tool failed, never run the synthesizer or fabricate hypothetical reports
    if tool_output.get("error"):
        trace.append(f"[agent] Tool execution failed: {tool_output['error']}")
        return {
            "answer": f"⚠️ **Analysis Execution Error**\n\nThe requested analysis could not be completed:\n\n`{tool_output['error']}`\n\nPlease rephrase your query or verify the dataset columns.",
            "tables": tables,
            "figures": figures,
            "plan": decision.model_dump(),
            "trace": trace,
            "code": None,
        }

    trace.append("[agent] Executed tool successfully.")

    # 4. Synthesizer: Grounded Narrative
    synth_prompt = f"""You are a senior data scientist and decision strategist defending an analysis in a technical business review.
Answer the user's question completely using the deterministic TOOL OUTPUT.

CRITICAL RULES:
1. Ground your answer strictly in the provided TOOL OUTPUT numbers. Never invent numbers.
2. Use hypothesis-testing language only when a formal test was selected. For descriptive summaries, report the requested values directly and do not invent H0/H1 or p-values. For a selected test, explicitly structure the analysis with:
   - **Hypothesis Formulation**: Clearly state the Null Hypothesis ($H_0$) and Alternative Hypothesis ($H_1$).
   - **Statistical Test & Evidence**: State the statistical test conducted, test statistic, p-value, effect size, and whether the null hypothesis is rejected or failed to be rejected at alpha = 0.05.
   - **Statistical Significance vs Practical Impact**: Explain whether observed differences are statistically significant or well within expected random variation.
3. BUSINESS DECISION & RECOMMENDATION:
   If the user asks whether to make a decision (e.g., 'should we fire them?', 'which strategy should we pick?'), provide a clear, data-backed business recommendation grounded in the statistical test results.
4. ALL COLUMNS USED:
   Explicitly list **All Columns Used for This Decision** and what insight each provided.

Tool Output:
{str(tool_output)[:4000]}
"""

    try:
        resp = llm.invoke([
            SystemMessage(content=synth_prompt),
            HumanMessage(content=question),
        ])
        if isinstance(resp.content, list):
            answer_text = "".join([
                b.get("text", "") if isinstance(b, dict) else (getattr(b, "text", "") or str(b))
                for b in resp.content
            ])
        else:
            answer_text = str(resp.content)
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        answer_text = f"Analysis completed.\n\n**Tool Output:**\n```\n{tool_output}\n```"

    trace.append("[agent] Completed grounded response synthesis.")

    # 5. RAG Memory
    if not tool_output.get("error") and decision.tool != "direct_answer":
        try:
            rag.add_memory(dataset_id, question, answer_text[:380], user_id=user_id)
        except Exception:
            pass

    return {
        "answer": answer_text,
        "tables": tables,
        "figures": figures,
        "plan": decision.model_dump(),
        "trace": trace,
        "code": None,
    }
