"""
FinBot — Production-Grade Multi-Agent Financial Chatbot
=======================================================
Powered by Claude API (Anthropic) with advanced analytics

Agents: ROUTER · PLANNER · EXECUTOR · CRITIC · INSIGHTS · FILTER · KPI · FORECAST · ANOMALY

Key capabilities:
- Claude Sonnet 4 as the LLM backbone (replaces Gemini)
- Bulletproof JSON extraction with 4-layer fallback
- ROUTER agent: classifies intent before planning
- ANOMALY agent: IQR + Z-score + Benford's Law
- FORECAST agent: Linear Regression + Exponential Smoothing + confidence intervals
- KPI agent: auto-detects financial metrics with trend analysis
- Safe code sandbox with restricted builtins
- Indian number formatting (Cr / L) throughout
- Conversation context memory for multi-turn analysis
"""

import os, json, re, time, traceback, math
from typing import Any, Optional

import pandas as pd
import numpy as np

from anthropic import Anthropic

# ── CONFIG ────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY  = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL       = "claude-sonnet-4-20250514"
MAX_CRITIC_RETRIES = 2
MAX_STEPS          = 4
MAX_API_RETRIES    = 3

client = Anthropic(api_key=ANTHROPIC_API_KEY)

# ── SAFE BUILTINS for exec ────────────────────────────────────────────────────
SAFE_BUILTINS = {
    "abs": abs, "round": round, "len": len, "range": range,
    "min": min, "max": max, "sum": sum, "sorted": sorted,
    "list": list, "dict": dict, "set": set, "tuple": tuple,
    "str": str, "int": int, "float": float, "bool": bool,
    "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
    "isinstance": isinstance, "print": lambda *a, **k: None,
    "any": any, "all": all, "reversed": reversed, "type": type,
    "hasattr": hasattr, "getattr": getattr,
}

# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1: JSON EXTRACTION — 4-layer fallback, never raises
# ─────────────────────────────────────────────────────────────────────────────

def extract_json(raw: str, fallback: dict = None) -> dict:
    """Try 4 strategies to extract JSON from LLM output. Never raises."""
    if fallback is None:
        fallback = {}
    text = raw.strip()

    def try_parse(s: str) -> Optional[dict]:
        s = re.sub(r"```(?:json|python)?", "", s).replace("```", "").strip()
        start = s.find("{")
        end   = s.rfind("}") + 1
        if start == -1 or end <= start:
            return None
        candidate = s[start:end]
        try:
            return json.loads(candidate)
        except Exception:
            pass
        # Fix common LLM issues
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        candidate = re.sub(r"(?<![\\])'", '"', candidate)
        candidate = re.sub(r'(?<!\\)\n', ' ', candidate)
        try:
            return json.loads(candidate)
        except Exception:
            pass
        # Regex key-value extraction
        obj = {}
        for m in re.finditer(r'"(\w+)"\s*:\s*"([^"]*)"', candidate):
            obj[m.group(1)] = m.group(2)
        for m in re.finditer(r'"(\w+)"\s*:\s*([\d.]+)', candidate):
            try:
                obj[m.group(1)] = float(m.group(2))
            except Exception:
                pass
        for m in re.finditer(r'"(\w+)"\s*:\s*(true|false|null)', candidate):
            mapping = {"true": True, "false": False, "null": None}
            obj[m.group(1)] = mapping[m.group(2)]
        if obj:
            return obj
        return None

    result = try_parse(text)
    if result:
        return result

    # Strategy 4: ask Claude to repair it
    if len(text) > 30:
        try:
            repair_resp = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=800,
                temperature=0,
                messages=[{
                    "role": "user",
                    "content": (
                        "The following text should be valid JSON but has syntax errors. "
                        "Return ONLY the corrected JSON object, nothing else:\n\n" + text[:2000]
                    ),
                }],
            )
            repaired = repair_resp.content[0].text.strip()
            result = try_parse(repaired)
            if result:
                return result
        except Exception:
            pass

    return fallback


def call_claude(system: str, messages: list, max_tokens: int = 1200, temperature: float = 0.05) -> str:
    """Call Claude API with retry logic. Returns raw text response."""
    for attempt in range(MAX_API_RETRIES):
        try:
            resp = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=messages,
            )
            return resp.content[0].text.strip()
        except Exception as e:
            err = str(e)
            if attempt == MAX_API_RETRIES - 1:
                raise
            wait = 2 ** attempt
            if "429" in err or "rate" in err.lower():
                wait = max(wait, 15)
            print(f"[claude retry {attempt+1}] {err} — waiting {wait}s")
            time.sleep(wait)
    raise RuntimeError("Claude API failed after retries")


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def safe_val(v: Any) -> Any:
    if isinstance(v, (np.integer,)):   return int(v)
    if isinstance(v, (np.floating,)):  return round(float(v), 4)
    if isinstance(v, (np.bool_,)):     return bool(v)
    if isinstance(v, pd.Timestamp):    return v.strftime("%Y-%m-%d")
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    return v

def df_to_str(df: pd.DataFrame, max_rows: int = 15) -> str:
    if df.empty:
        return "(empty)"
    total   = len(df)
    preview = df.head(max_rows).fillna("").to_string(index=False)
    return preview + (f"\n…({total} total rows)" if total > max_rows else "")

def result_to_str(result: Any, max_rows: int = 15) -> str:
    if isinstance(result, pd.DataFrame):
        return df_to_str(result, max_rows)
    if isinstance(result, pd.Series):
        return df_to_str(result.reset_index(), max_rows)
    if isinstance(result, (int, float, np.integer, np.floating)):
        return f"{safe_val(result):,}"
    if isinstance(result, dict):
        return "\n".join(f"  {k}: {safe_val(v)}" for k, v in result.items())
    return str(result)[:2000]

def format_result_as_table(result: Any) -> Optional[list]:
    if isinstance(result, pd.DataFrame):
        return result.head(500).fillna("").to_dict(orient="records")
    if isinstance(result, pd.Series):
        return result.reset_index().head(500).fillna("").to_dict(orient="records")
    return None

def build_chart_data(result: Any, chart_spec: dict) -> Optional[dict]:
    if not chart_spec or chart_spec.get("type") in (None, "none", ""):
        return None
    chart_type = chart_spec.get("type", "bar")
    title      = chart_spec.get("title", "Chart")
    try:
        if isinstance(result, pd.Series):
            labels = [str(i) for i in result.index]
            values = [safe_val(v) for v in result.values]
            if chart_type == "pie":
                return {"type": "pie", "title": title, "labels": labels, "values": values}
            return {"type": chart_type, "title": title, "x": labels, "y": values,
                    "x_label": str(result.index.name or ""), "y_label": str(result.name or "")}
        if isinstance(result, pd.DataFrame):
            x_col = chart_spec.get("x_col") or (result.columns[0] if len(result.columns) > 0 else None)
            y_col = chart_spec.get("y_col")
            if not y_col:
                nums  = result.select_dtypes(include=[np.number]).columns
                y_col = nums[0] if len(nums) > 0 else (result.columns[1] if len(result.columns) > 1 else result.columns[0])
            if x_col and y_col and x_col in result.columns and y_col in result.columns:
                x_vals = [str(safe_val(v)) for v in result[x_col]]
                y_vals = [safe_val(v) for v in result[y_col]]
                if chart_type == "pie":
                    return {"type": "pie", "title": title, "labels": x_vals, "values": y_vals}
                return {"type": chart_type, "title": title, "x": x_vals, "y": y_vals,
                        "x_label": str(x_col), "y_label": str(y_col)}
    except Exception as e:
        print(f"[chart] {e}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

ROUTER_SYSTEM = """You are a financial query ROUTER. Classify the user's question into one category.

Categories:
- analysis    : aggregations, groupby, sums, counts, averages, rankings, comparisons, distributions
- filter      : user wants to slice/filter the data (contains: "only", "filter", "where", "show me X from Y")
- anomaly     : outliers, duplicates, suspicious, unusual, fraud detection, risk
- forecast    : predict, trend, future, next month/quarter/year, projection
- general     : greetings, help, dataset info, column questions

Dataset columns: {columns}

Respond with ONLY this JSON (no extra text):
{{"intent": "analysis|filter|anomaly|forecast|general", "reason": "one sentence"}}"""

PLANNER_SYSTEM = """You are the PLANNER agent for a production financial analysis system.

Dataset:
- Columns     : {columns}
- Numeric     : {numeric_cols}
- Date cols   : {date_cols}
- Categories  : {cat_cols}
- Shape       : {shape}
- Sample data :
{sample}

Create a step-by-step plan to answer the user's financial question.

You MUST respond with ONLY valid JSON in this exact structure — no extra text, no markdown:
{{"understanding": "one sentence what user wants", "steps": [{{"step": 1, "action": "description"}}], "chart": {{"type": "bar", "title": "Chart Title", "x_col": null, "y_col": null}}, "final_answer_hint": "what insight this provides"}}

Rules:
- chart type must be one of: bar, line, pie, scatter, heatmap, none
- 1 to 4 steps maximum
- steps array must never be empty
- For distribution questions prefer histogram (use bar type)
- For time series prefer line charts
- For composition/share prefer pie charts
- For correlation prefer scatter plots"""

EXECUTOR_SYSTEM = """You are the EXECUTOR agent. Write Python/pandas code for one analysis step.

DataFrame `df` columns: {columns}
Numeric: {numeric_cols} | Dates: {date_cols} | Categories: {cat_cols}

Code rules:
- Store final output in variable named exactly `result`
- result can be: number, string, DataFrame, Series, or dict
- Use `prev_result` if building on previous step
- Never use print()
- Handle NaN: use .fillna(0) or .dropna()
- For outliers: use IQR method or Z-score
- For dates: use pd.to_datetime() and .dt accessor
- For percentages: multiply by 100 and round to 2 decimals
- For top-N: use .nlargest() or .head() after sorting
- Always use .copy() when slicing to avoid SettingWithCopyWarning

You MUST respond with ONLY valid JSON — no extra text:
{{"reasoning": "why this approach in one sentence", "code": "single line or multiline python code here"}}"""

CRITIC_SYSTEM = """You are the CRITIC agent. Review if the analysis result correctly answers the question.

Check:
1. Does result answer what was asked?
2. Is result non-empty and non-None?
3. Is the logic sound?
4. Are numbers reasonable (not NaN, not all zeros)?
5. If a chart was requested, does the result have the right shape for it?

You MUST respond with ONLY valid JSON — no extra text:
{{"verdict": "PASS", "confidence": 0.9, "issues": [], "fix_instruction": null, "summary": "one sentence describing the result"}}

verdict must be exactly "PASS" or "FAIL". Be generous — PASS if result looks reasonable."""

INSIGHTS_SYSTEM = """You are a senior fintech analyst at a top-tier financial institution. Generate specific, data-backed insights.

Rules:
- Reference actual numbers from the result
- No generic advice — only data-backed observations
- Identify: concentration risk, outlier patterns, trend shifts, seasonal effects
- Tips: concrete actionable business decisions with expected impact
- Use financial terminology appropriately

You MUST respond with ONLY valid JSON — no extra text:
{{"insights": ["Insight referencing actual number 1", "Insight 2", "Insight 3"], "growth_tips": ["Concrete action 1", "Concrete action 2", "Concrete action 3"]}}"""

FILTER_SYSTEM = """You are the FILTER agent. Convert natural language to a pandas boolean expression.

Dataset:
- Columns    : {columns}
- Numeric    : {numeric_cols}
- Dates      : {date_cols}
- Categories : {cat_cols}
- Sample values per column: {sample}

Convert the filter request to a pandas expression for: df[<expression>]

Rules:
- Use exact column names (case-sensitive)
- Strings: df['col'].str.contains('val', case=False, na=False) or df['col'] == 'val'
- Dates: df['col'] >= pd.Timestamp('2024-01-01')
- Numbers: df['col'] > 1000
- Multiple: (expr1) & (expr2)
- "reset"/"clear"/"all data" → expression = "all"
- Cannot parse → expression = "error"

You MUST respond with ONLY valid JSON:
{{"expression": "df['col'] > 100", "description": "human readable description", "estimated_rows": "~40% of data"}}"""

KPI_SYSTEM = """You are the KPI agent for a financial analytics platform. Compute 8 key financial metrics from dataframe `df`.

Dataset columns: {columns}
Numeric: {numeric_cols} | Dates: {date_cols} | Categories: {cat_cols}

Write Python code that sets `result` = list of exactly 8 dicts.
Each dict: {{"name": "KPI Name", "value": 123.45, "format": "currency", "trend_hint": "up"}}
format options: "currency", "count", "percent", "number"
trend_hint: "up" (higher=better), "down" (lower=better), "neutral"

Priority KPIs (compute as many as data allows):
1. Total spend/amount/value
2. Average transaction value
3. Transaction count
4. Unique suppliers/vendors/entities
5. Top category spend concentration (%)
6. Month-over-month growth rate (%)
7. Median transaction value
8. Date range span or other relevant metric

Wrap each KPI computation in try/except, default value to 0 on error.

You MUST respond with ONLY valid JSON:
{{"reasoning": "which columns and why", "code": "python code setting result = [...]"}}"""

ANOMALY_SYSTEM = """You are the ANOMALY DETECTION agent for a financial fraud/risk analysis system.

Dataset columns: {columns}
Numeric: {numeric_cols} | Dates: {date_cols} | Categories: {cat_cols}

Write Python/pandas code to detect anomalies in `df`. Store findings in `result` as a DataFrame.

Apply MULTIPLE detection methods (use those relevant to available columns):

1. **IQR Outliers** — values beyond Q1-1.5*IQR or Q3+1.5*IQR on amount/value columns
2. **Z-Score Outliers** — |z| > 3 on numeric columns
3. **Duplicate Detection** — same key fields (PO number, invoice, amount+supplier+date)
4. **Benford's Law** — first-digit distribution anomaly for amounts > 10
5. **Round Number Bias** — amounts ending in 000 or 00 (potential fraud signal)
6. **Weekend/Holiday Transactions** — if date column exists
7. **Vendor Concentration** — vendors with single very-high-value transactions

Return a DataFrame with columns: anomaly_type, description, risk_level (High/Medium/Low), plus relevant original columns.

You MUST respond with ONLY valid JSON:
{{"reasoning": "what anomalies you're looking for", "code": "python code setting result = DataFrame"}}"""


# ─────────────────────────────────────────────────────────────────────────────
# FORECAST ENGINE — Multi-method
# ─────────────────────────────────────────────────────────────────────────────

def compute_forecast(df: pd.DataFrame, date_col: str, value_col: str, periods: int = 6) -> Optional[dict]:
    """
    Multi-method forecast:
    1. Linear Regression (baseline)
    2. Exponential Smoothing
    3. Weighted Moving Average
    Returns best model based on in-sample fit + confidence intervals.
    """
    try:
        from sklearn.linear_model import LinearRegression

        tmp = df[[date_col, value_col]].dropna().copy()
        tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
        tmp = tmp.dropna(subset=[date_col])
        tmp["_month"] = tmp[date_col].dt.to_period("M")
        monthly = tmp.groupby("_month")[value_col].sum().reset_index().sort_values("_month")

        if len(monthly) < 3:
            return {"error": f"Need at least 3 months of data (found {len(monthly)})"}

        monthly["_idx"] = range(len(monthly))
        X = monthly[["_idx"]].values
        y = monthly[value_col].values

        # Method 1: Linear Regression
        lr_model = LinearRegression()
        lr_model.fit(X, y)
        lr_r2    = float(lr_model.score(X, y))
        lr_slope = float(lr_model.coef_[0])

        # Method 2: Weighted Moving Average
        weights = np.array([0.5 ** i for i in range(min(6, len(y)))])[::-1]
        weights = weights / weights.sum()
        recent  = y[-len(weights):]
        wma     = float(np.dot(recent, weights))

        # Method 3: Exponential Smoothing
        alpha     = 0.3
        smoothed  = [y[0]]
        for val in y[1:]:
            smoothed.append(alpha * val + (1 - alpha) * smoothed[-1])

        # Choose best: use LR if R² > 0.5, else WMA
        use_lr = lr_r2 > 0.5

        hist_labels = [str(p) for p in monthly["_month"]]
        hist_values = [safe_val(v) for v in y]

        cur = monthly["_month"].iloc[-1]
        future_labels, future_values = [], []

        for i in range(1, periods + 1):
            cur = cur + 1
            future_labels.append(str(cur))
            if use_lr:
                pred = float(lr_model.predict([[len(monthly) + i - 1]])[0])
            else:
                pred = alpha * (wma * (1 + lr_slope / max(abs(wma), 1) * i)) + (1 - alpha) * smoothed[-1]
            future_values.append(safe_val(max(pred, 0)))

        method_name = "Linear Regression" if use_lr else "Exponential Smoothing"

        # Confidence intervals
        if use_lr:
            preds     = lr_model.predict(X)
            residuals = y - preds.flatten()
            std_err   = float(np.std(residuals))
        else:
            std_err = float(np.std(y[-6:]) if len(y) >= 6 else np.std(y))

        confidence_upper = [safe_val(v + 1.96 * std_err) for v in future_values]
        confidence_lower = [safe_val(max(0, v - 1.96 * std_err)) for v in future_values]

        return {
            "date_col":          date_col,
            "value_col":         value_col,
            "hist_labels":       hist_labels,
            "hist_values":       hist_values,
            "future_labels":     future_labels,
            "future_values":     future_values,
            "confidence_upper":  confidence_upper,
            "confidence_lower":  confidence_lower,
            "r2":                round(lr_r2, 3),
            "trend":             "upward 📈" if lr_slope > 0 else "downward 📉",
            "slope_per_month":   safe_val(lr_slope),
            "total_months":      len(monthly),
            "method":            method_name,
            "std_error":         safe_val(std_err),
        }
    except ImportError:
        return {"error": "scikit-learn not installed. Run: pip install scikit-learn"}
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CHATBOT CLASS
# ─────────────────────────────────────────────────────────────────────────────

class FinancialChatbot:

    def __init__(self, df: pd.DataFrame):
        self.df_original        = df.copy()
        self.df                 = df.copy()
        self.active_filter_desc = ""
        self.conversation_ctx   = []

        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.date_cols    = df.select_dtypes(include=["datetime64"]).columns.tolist()
        self.cat_cols     = df.select_dtypes(include=["object", "category"]).columns.tolist()

        # Sample unique values for filter agent
        sample_vals = {}
        for col in self.cat_cols[:8]:
            sample_vals[col] = [str(v) for v in df[col].dropna().unique()[:10].tolist()]

        self.ctx = dict(
            columns      = list(df.columns),
            numeric_cols = self.numeric_cols,
            date_cols    = self.date_cols,
            cat_cols     = self.cat_cols,
            shape        = df.shape,
            sample       = df_to_str(df, max_rows=4),
        )
        self.filter_ctx = {**self.ctx, "sample": json.dumps(sample_vals, default=str, ensure_ascii=False)}

        # Per-agent conversation histories (Claude messages format)
        self.planner_hist:  list = []
        self.executor_hist: list = []
        self.critic_hist:   list = []
        self.insights_hist: list = []

    # ── HELPERS ───────────────────────────────────────────────────────────────

    def _build_system(self, template: str, use_filter_ctx: bool = False) -> str:
        ctx = self.filter_ctx if use_filter_ctx else self.ctx
        return template.format(**ctx)

    def _chat(self, system_template: str, history: list, message: str,
              max_tokens: int = 1200, temperature: float = 0.05,
              use_filter_ctx: bool = False) -> str:
        system = self._build_system(system_template, use_filter_ctx)
        history.append({"role": "user", "content": message})
        text = call_claude(system, history, max_tokens=max_tokens, temperature=temperature)
        history.append({"role": "assistant", "content": text})
        if len(history) > 16:
            history[:] = history[-16:]
        return text

    def _run_code(self, code: str, prev_result: Any = None) -> tuple:
        local = {
            "df": self.df, "pd": pd, "np": np,
            "prev_result": prev_result, "result": None,
        }
        globs = {"pd": pd, "np": np, "__builtins__": SAFE_BUILTINS}
        try:
            exec(compile(code, "<finbot>", "exec"), globs, local)
            return local.get("result"), None
        except Exception as e:
            return None, f"{type(e).__name__}: {e}"

    # ── AGENT: ROUTER ─────────────────────────────────────────────────────────

    def _route(self, question: str) -> str:
        try:
            system = self._build_system(ROUTER_SYSTEM)
            raw = call_claude(system, [{"role": "user", "content": question}],
                              max_tokens=200, temperature=0)
            parsed = extract_json(raw, {"intent": "analysis"})
            return parsed.get("intent", "analysis")
        except Exception:
            return "analysis"

    # ── AGENT: PLANNER ────────────────────────────────────────────────────────

    def _planner(self, question: str) -> dict:
        ctx_block = ""
        if self.conversation_ctx:
            recent = self.conversation_ctx[-3:]
            ctx_block = "Recent conversation context:\n" + "\n".join(
                f"  Q: {c['q']}\n  A: {c['a']}" for c in recent
            ) + "\n\n"
        if self.active_filter_desc:
            ctx_block += f"Active filter: {self.active_filter_desc}\n\n"

        prompt = f"{ctx_block}Question: {question}"
        raw    = self._chat(PLANNER_SYSTEM, self.planner_hist, prompt, max_tokens=800)

        fallback = {
            "understanding": question,
            "steps":         [{"step": 1, "action": f"Analyze: {question}"}],
            "chart":         {"type": "none", "title": "", "x_col": None, "y_col": None},
            "final_answer_hint": "Analysis result",
        }
        return extract_json(raw, fallback)

    # ── AGENT: EXECUTOR ───────────────────────────────────────────────────────

    def _executor(self, action: str, prev_result: Any, plan: dict, fix_note: str = "") -> dict:
        prev_str  = result_to_str(prev_result) if prev_result is not None else "None (first step)"
        fix_block = f"\n\nPREVIOUS ERROR — FIX THIS: {fix_note}" if fix_note else ""
        prompt    = (
            f"Task: {action}\n"
            f"Overall goal: {plan.get('understanding', '')}\n"
            f"Previous result:\n{prev_str}"
            f"{fix_block}\n\n"
            "Write pandas code. Store final answer in `result`."
        )
        raw      = self._chat(EXECUTOR_SYSTEM, self.executor_hist, prompt)
        fallback = {"reasoning": "direct analysis", "code": f"result = 'Could not generate code for: {action}'"}
        return extract_json(raw, fallback)

    # ── AGENT: CRITIC ─────────────────────────────────────────────────────────

    def _critic(self, question: str, plan: dict, code: str, result: Any) -> dict:
        prompt = (
            f"Question: {question}\n"
            f"Goal: {plan.get('understanding', '')}\n"
            f"Code executed:\n{code}\n\n"
            f"Result:\n{result_to_str(result)}\n\n"
            "Is this result correct and complete?"
        )
        raw      = self._chat(CRITIC_SYSTEM, self.critic_hist, prompt, max_tokens=500, temperature=0)
        fallback = {"verdict": "PASS", "confidence": 0.7, "issues": [], "fix_instruction": None, "summary": "Analysis complete"}
        return extract_json(raw, fallback)

    # ── AGENT: INSIGHTS ───────────────────────────────────────────────────────

    def _insights(self, question: str, result: Any) -> dict:
        prompt = (
            f"User question: {question}\n\n"
            f"Analysis result:\n{result_to_str(result, max_rows=25)}\n\n"
            "Generate 3 specific insights and 3 growth tips based on this data."
        )
        try:
            raw = self._chat(INSIGHTS_SYSTEM, self.insights_hist, prompt, max_tokens=700, temperature=0.3)
            return extract_json(raw, {"insights": [], "growth_tips": []})
        except Exception:
            return {"insights": [], "growth_tips": []}

    # ── AGENT: FILTER ─────────────────────────────────────────────────────────

    def apply_filter(self, user_text: str) -> dict:
        try:
            system = self._build_system(FILTER_SYSTEM, use_filter_ctx=True)
            raw  = call_claude(system, [{"role": "user", "content": f"Filter request: {user_text}"}],
                               max_tokens=400, temperature=0)
            resp = extract_json(raw, {"expression": "error", "description": user_text})
        except Exception as e:
            return {"success": False, "error": str(e), "rows": len(self.df)}

        expr = resp.get("expression", "error").strip()
        desc = resp.get("description", user_text)

        if expr in ("error", "", "null", "None"):
            return {"success": False, "error": "Could not parse filter — try rephrasing", "rows": len(self.df)}

        if expr == "all":
            self.df = self.df_original.copy()
            self.active_filter_desc = ""
            return {"success": True, "cleared": True, "description": "All filters cleared", "rows": len(self.df)}

        try:
            local = {"df": self.df_original, "pd": pd, "np": np}
            mask  = eval(expr, {"pd": pd, "np": np, "__builtins__": {}}, local)
            df_f  = self.df_original[mask].reset_index(drop=True)
            if len(df_f) == 0:
                return {"success": False, "error": "Filter returned 0 rows — try different values", "rows": len(self.df)}
            self.df = df_f
            self.active_filter_desc = desc
            return {
                "success": True, "description": desc,
                "rows": len(self.df),
                "pct": round(len(self.df) / max(len(self.df_original), 1) * 100, 1),
            }
        except Exception as e:
            return {"success": False, "error": f"Filter eval failed: {e}", "rows": len(self.df)}

    # ── AGENT: KPI ────────────────────────────────────────────────────────────

    def compute_kpis(self) -> list:
        try:
            system = self._build_system(KPI_SYSTEM)
            raw    = call_claude(system, [{"role": "user", "content": "Compute 8 KPIs for the current dataset."}],
                                 max_tokens=1200, temperature=0.05)
            resp = extract_json(raw, {})
            code = resp.get("code", "")
            if code:
                result, err = self._run_code(code)
                if not err and isinstance(result, list) and len(result) >= 4:
                    for k in result:
                        k["value"] = safe_val(k.get("value", 0))
                    return result[:8]
        except Exception as e:
            print(f"[kpi agent] {e}")
        return self._fallback_kpis()

    def _fallback_kpis(self) -> list:
        df   = self.df
        kpis = []
        try:
            num_cols    = df.select_dtypes(include=[np.number]).columns
            cat_cols    = df.select_dtypes(include=["object"]).columns
            amount_cols = [c for c in num_cols if any(k in c.lower() for k in ["amount","net","value","cost","price","spend","qty","quantity"])]
            main_col    = amount_cols[0] if amount_cols else (list(num_cols)[0] if len(num_cols) > 0 else None)

            kpis.append({"name": "Total Records", "value": len(df), "format": "count", "trend_hint": "neutral"})
            if main_col:
                kpis.append({"name": f"Total {main_col}", "value": safe_val(df[main_col].sum()), "format": "currency", "trend_hint": "up"})
                kpis.append({"name": f"Avg {main_col}", "value": safe_val(df[main_col].mean()), "format": "currency", "trend_hint": "neutral"})
                kpis.append({"name": f"Median {main_col}", "value": safe_val(df[main_col].median()), "format": "currency", "trend_hint": "neutral"})
                kpis.append({"name": f"Max {main_col}", "value": safe_val(df[main_col].max()), "format": "currency", "trend_hint": "neutral"})
                kpis.append({"name": "Std Dev", "value": safe_val(df[main_col].std()), "format": "currency", "trend_hint": "down"})
            for col in list(cat_cols)[:2]:
                kpis.append({"name": f"Unique {col}", "value": int(df[col].nunique()), "format": "count", "trend_hint": "neutral"})
            if len(self.date_cols) > 0:
                try:
                    date_range = (df[self.date_cols[0]].max() - df[self.date_cols[0]].min()).days
                    kpis.append({"name": "Date Range (days)", "value": int(date_range), "format": "count", "trend_hint": "neutral"})
                except Exception:
                    pass
        except Exception as e:
            print(f"[fallback kpi] {e}")
        while len(kpis) < 8:
            kpis.append({"name": "—", "value": 0, "format": "count", "trend_hint": "neutral"})
        return kpis[:8]

    # ── AGENT: ANOMALY ────────────────────────────────────────────────────────

    def detect_anomalies(self) -> dict:
        try:
            system = self._build_system(ANOMALY_SYSTEM)
            raw    = call_claude(system, [{"role": "user", "content": "Detect all anomalies in the dataset."}],
                                  max_tokens=1500, temperature=0.05)
            resp = extract_json(raw, {})
            code = resp.get("code", "")
            if code:
                result, err = self._run_code(code)
                if not err and isinstance(result, pd.DataFrame) and len(result) > 0:
                    anomaly_types = result["anomaly_type"].nunique() if "anomaly_type" in result.columns else "?"
                    summary = f"Found **{len(result):,} anomalies** across {anomaly_types} categories."
                    return {"found": True, "count": len(result), "table": format_result_as_table(result), "summary": summary}
                elif not err:
                    return {"found": False, "count": 0, "table": None, "summary": "No anomalies detected in current dataset."}
        except Exception as e:
            print(f"[anomaly agent] {e}")
        return self._fallback_anomalies()

    def _fallback_anomalies(self) -> dict:
        df   = self.df
        rows = []
        num_cols = df.select_dtypes(include=[np.number]).columns
        amount_cols = [c for c in num_cols if any(k in c.lower() for k in ["amount","net","value","cost","price"])]

        for col in (amount_cols or list(num_cols))[:3]:
            try:
                Q1  = df[col].quantile(0.25)
                Q3  = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[col] < Q1 - 3*IQR) | (df[col] > Q3 + 3*IQR)].copy()
                if len(outliers) > 0:
                    outliers["anomaly_type"] = f"IQR Outlier ({col})"
                    outliers["risk_level"]   = "High"
                    outliers["description"]  = f"{col} outside 3×IQR range [{round(Q1-3*IQR,2)}, {round(Q3+3*IQR,2)}]"
                    rows.append(outliers)
            except Exception:
                pass

        # Benford's Law check
        for col in (amount_cols or list(num_cols))[:1]:
            try:
                vals = df[col].dropna()
                vals = vals[vals > 10]
                if len(vals) > 100:
                    first_digits = vals.astype(str).str.lstrip('-').str[0].astype(int)
                    first_digits = first_digits[first_digits > 0]
                    observed = first_digits.value_counts(normalize=True).sort_index()
                    benford  = pd.Series({d: math.log10(1 + 1/d) for d in range(1, 10)})
                    chi2 = sum((observed.get(d, 0) - benford[d])**2 / benford[d] for d in range(1, 10))
                    if chi2 > 0.05:
                        benford_row = pd.DataFrame([{
                            "anomaly_type": "Benford's Law Deviation",
                            "risk_level": "Medium" if chi2 < 0.15 else "High",
                            "description": f"First-digit distribution in {col} deviates from Benford's Law (chi²={chi2:.4f}). Possible data manipulation.",
                        }])
                        rows.append(benford_row)
            except Exception:
                pass

        if rows:
            combined = pd.concat(rows, ignore_index=True)
            return {"found": True, "count": len(combined), "table": format_result_as_table(combined),
                    "summary": f"Found **{len(combined)}** anomalies using IQR + Benford's Law analysis."}
        return {"found": False, "count": 0, "table": None, "summary": "No significant anomalies detected."}

    # ── AGENT: FORECAST ───────────────────────────────────────────────────────

    def get_forecast(self, periods: int = 6) -> dict:
        if not self.date_cols:
            return {"available": False, "reason": "No date columns found in dataset."}
        num_cols    = self.df.select_dtypes(include=[np.number]).columns
        amount_cols = [c for c in num_cols if any(k in c.lower() for k in ["amount","net","value","cost","price","spend"])]
        value_col   = (amount_cols or list(num_cols))[0] if len(num_cols) > 0 else None
        if not value_col:
            return {"available": False, "reason": "No numeric columns found."}
        result = compute_forecast(self.df, self.date_cols[0], value_col, periods)
        if "error" in result:
            return {"available": False, "reason": result["error"]}
        return {"available": True, **result}

    # ── MAIN ANSWER ───────────────────────────────────────────────────────────

    def answer(self, user_message: str) -> dict:
        agent_log = []

        # 0. ROUTE
        intent = self._route(user_message)
        agent_log.append(f"🔀 **Router**: intent = `{intent}`")

        if intent == "general":
            cols_info = ", ".join(list(self.df.columns)[:8])
            return {
                "answer": f"I'm FinBot, your AI financial analyst powered by Claude!\n\nYour dataset has **{len(self.df):,} rows** and **{len(self.df.columns)} columns**.\nKey columns: {cols_info}…\n\nAsk me to analyze spend, find trends, detect anomalies, or apply filters!",
                "chart": None, "table": None, "code": None,
                "agent_log": agent_log, "insights": [], "growth_tips": [],
            }

        if intent == "anomaly":
            agent_log.append("🔍 **Anomaly Agent** running multi-method detection…")
            result = self.detect_anomalies()
            agent_log.append(f"  → {result['summary']}")
            return {
                "answer": result["summary"], "chart": None, "table": result["table"], "code": None,
                "agent_log": agent_log, "insights": [result["summary"]],
                "growth_tips": [
                    "Review high-risk transactions with your finance team",
                    "Set up automated alerts for values exceeding 3× IQR",
                    "Cross-check flagged vendors against approved vendor list",
                ],
            }

        if intent == "forecast":
            agent_log.append("📈 **Forecast Agent** computing trend projection…")
            fc = self.get_forecast()
            if not fc.get("available"):
                return {"answer": f"Cannot forecast: {fc.get('reason')}", "chart": None, "table": None, "code": None, "agent_log": agent_log, "insights": [], "growth_tips": []}
            method = fc.get("method", "Linear Regression")
            trend_desc = (
                f"Spend trend is **{fc['trend']}** based on {fc['total_months']} months of data.\n\n"
                f"- Method: **{method}**\n"
                f"- Monthly change: **{fc['slope_per_month']:+,.0f}** per month\n"
                f"- Model R² score: **{fc['r2']}** ({'strong' if fc['r2'] > 0.7 else 'moderate' if fc['r2'] > 0.4 else 'weak'} fit)\n"
                f"- Std Error: ±{fc.get('std_error', 0):,.0f}\n\n"
                f"See the 📈 Forecast button for the interactive chart with confidence intervals."
            )
            return {
                "answer": trend_desc, "chart": None, "table": None, "code": None, "agent_log": agent_log,
                "insights": [
                    f"Trend direction: {fc['trend']}",
                    f"R²={fc['r2']} — {'reliable' if fc['r2']>0.7 else 'approximate'} forecast using {method}",
                ],
                "growth_tips": [
                    "Use forecast to negotiate annual contracts with top suppliers",
                    "Budget next quarter based on projected spend trajectory",
                    "Investigate causes if trend diverges from budget plan",
                ],
            }

        # 1. PLAN
        try:
            plan = self._planner(user_message)
        except Exception as e:
            return {"answer": f"Planning failed: {e}\n\nTry rephrasing your question.",
                    "chart": None, "table": None, "code": None, "agent_log": agent_log, "insights": [], "growth_tips": []}

        understanding = plan.get("understanding", user_message)
        steps         = plan.get("steps", [])
        agent_log.append(f"🧠 **Planner**: {understanding}")
        for s in steps:
            agent_log.append(f"  • Step {s.get('step','?')}: {s.get('action','')}")
        if not steps:
            steps = [{"step": 1, "action": f"Analyze: {user_message}"}]

        # 2. EXECUTE + CRITIC loop
        prev_result = None; final_result = None; all_code = []; review = {}

        for step in steps[:MAX_STEPS]:
            action = step.get("action", ""); fix_note = ""; accepted = False
            for attempt in range(MAX_CRITIC_RETRIES + 1):
                agent_log.append(f"⚙️ **Executor** step {step.get('step','?')} (try {attempt+1}): {action[:80]}")
                try:
                    exec_resp = self._executor(action, prev_result, plan, fix_note)
                    code      = exec_resp.get("code", "").strip()
                    reasoning = exec_resp.get("reasoning", "")
                    if reasoning:
                        agent_log.append(f"  💭 {reasoning}")
                except Exception as e:
                    agent_log.append(f"  ❌ Executor failed: {e}")
                    break
                if not code:
                    agent_log.append("  ⚠️ No code generated"); break
                all_code.append(f"# Step {step.get('step','?')}: {action}\n{code}")
                result, error = self._run_code(code, prev_result)
                if error:
                    agent_log.append(f"  ❌ Runtime: {error}")
                    fix_note = f"Code raised: {error}. Fix the code and try again."
                    continue

                agent_log.append("🔍 **Critic** reviewing…")
                try:
                    review  = self._critic(user_message, plan, code, result)
                    verdict = review.get("verdict", "PASS")
                    conf    = float(review.get("confidence", 1.0))
                    issues  = review.get("issues", [])
                    fix_note = review.get("fix_instruction") or ""
                    agent_log.append(f"  → {verdict} ({conf:.0%} confidence)")
                    if issues:
                        agent_log.append(f"  → Issues: {'; '.join(str(i) for i in issues)}")
                    if verdict == "FAIL" and fix_note and attempt < MAX_CRITIC_RETRIES:
                        agent_log.append("  → Requesting fix…"); continue
                except Exception as e:
                    agent_log.append(f"  ⚠️ Critic error: {e}"); review = {}

                prev_result = result; final_result = result; accepted = True; break
            if not accepted and prev_result is not None:
                final_result = prev_result

        # 3. FORMAT ANSWER
        if final_result is None:
            return {
                "answer": "Could not compute a result. Try rephrasing or break it into a simpler question.",
                "chart": None, "table": None, "code": "\n\n".join(all_code),
                "agent_log": agent_log, "insights": [], "growth_tips": [],
            }

        summary = review.get("summary", "") if review else ""
        lines   = [summary or plan.get("final_answer_hint", "Analysis complete."), ""]
        if isinstance(final_result, (int, float, np.integer, np.floating)):
            lines.append(f"**Result: {safe_val(final_result):,}**")
        elif isinstance(final_result, str):
            lines.append(final_result)
        elif isinstance(final_result, pd.DataFrame):
            lines.append(f"Found **{len(final_result):,} rows**. See table below.")
        elif isinstance(final_result, pd.Series):
            lines.append(f"Found **{len(final_result):,} entries**. See chart below.")
        elif isinstance(final_result, dict):
            lines += [f"- **{k}**: {safe_val(v)}" for k, v in final_result.items()]
        if self.active_filter_desc:
            lines.append(f"\n*🔍 Active filter: {self.active_filter_desc}*")

        self.conversation_ctx.append({"q": user_message, "a": summary or understanding})
        if len(self.conversation_ctx) > 10:
            self.conversation_ctx = self.conversation_ctx[-10:]

        # 4. INSIGHTS
        insights_data = self._insights(user_message, final_result)

        return {
            "answer":      "\n".join(lines),
            "chart":       build_chart_data(final_result, plan.get("chart", {})),
            "table":       format_result_as_table(final_result),
            "code":        "\n\n".join(all_code),
            "agent_log":   agent_log,
            "insights":    insights_data.get("insights", []),
            "growth_tips": insights_data.get("growth_tips", []),
        }