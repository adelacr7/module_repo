"""
Northbridge Bank - Credit Risk Natural-Language Query Engine (Streamlit app)

Converted from the notebook `Credit_Risk_Query_Engine.ipynb`.

Pipeline (unchanged from the notebook):
    intent classification -> verified template OR SQL generation
    -> 5-check validation gate -> one retry (generated track only)
    -> escalation to a human analyst -> read-only execution -> narrative response
with the SQL used, the raw data, a confidence score and an audit record for
every answer.
"""

import os
import re
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sqlparse
import streamlit as st

from langchain_openai import ChatOpenAI

# ----------------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------------
st.set_page_config( 
    page_title="Credit Risk Query Engine",
    page_icon="  ",
    layout="wide",
)

# Fix NameError in notebook by checking if __file__ is defined
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

DB_PATH = os.environ.get("CREDIT_RISK_DB_PATH", str(BASE_DIR / "credit_risk_portfolio.db"))
TEST_QUERIES_PATH = os.environ.get("TEST_QUERIES_PATH", str(BASE_DIR / "test_queries.csv"))
CONFIG_PATH = BASE_DIR / "config.json"
AUDIT_LOG_PATH = os.environ.get("AUDIT_LOG_PATH", str(BASE_DIR / "audit_log.jsonl"))

LLM_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
EVALUATOR_MODEL = os.environ.get("OPENAI_EVAL_MODEL", "gpt-4o")


# ----------------------------------------------------------------------------
# Credentials: Streamlit secrets -> environment variables -> config.json
# (the notebook read config.json; the other two sources make deployment easier)
# ----------------------------------------------------------------------------
def load_credentials():
    """Load OpenAI credentials and export them as environment variables."""
    api_key, api_base = None, None

    # 1) Streamlit secrets (.streamlit/secrets.toml or Streamlit Community Cloud secrets)
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        api_key = None
    try:
        api_base = st.secrets.get("OPENAI_API_BASE")
    except Exception:
        api_base = None

    # 2) Environment variables
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    api_base = api_base or os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL")

    # 3) config.json (same format as the notebook)
    if (not api_key or not api_base) and CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r") as file:
                config = json.load(file)
            api_key = api_key or config.get("OPENAI_API_KEY")
            api_base = api_base or config.get("OPENAI_API_BASE")
        except Exception:
            pass

    if api_key:
        os.environ["OPENAI_API_KEY"] = str(api_key)
    if api_base:
        os.environ["OPENAI_BASE_URL"] = str(api_base)
        os.environ["OPENAI_API_BASE"] = str(api_base)

    return bool(api_key), api_base


@st.cache_resource(show_spinner=False)
def init_llms(model_name, evaluator_name, _cache_buster):
    """Set up the LLM(s) - same settings as the notebook."""
    _llm = ChatOpenAI(model=model_name, temperature=0)
    _evaluator_llm = ChatOpenAI(model=evaluator_name, temperature=0)
    return _llm, _evaluator_llm


def get_connection():
    """Read-only connection to the SQLite database (prevents any write operation)."""
    return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, check_same_thread=False)


# ----------------------------------------------------------------------------
# Database schema provided to the LLM in the prompts
# ----------------------------------------------------------------------------
database_schema = """
sector_master:
  sector_code (TEXT, PK): internal sector identifier (e.g., SEC_RE, SEC_INFRA)
  sector_name (TEXT): human-readable sector name (e.g., Real Estate, Infrastructure)
  naics_code (TEXT): NAICS industry classification code
  naics_description (TEXT): NAICS code description
  is_sensitive_sector (INTEGER): 1 if sensitive sector, 0 otherwise

loan_master:
  loan_account_number (TEXT, PK): unique loan identifier
  borrower_id (TEXT): borrower identifier (joins to borrower_rating.borrower_id)
  borrower_name (TEXT): registered legal name of the borrower
  borrower_type (TEXT): entity type (C-Corporation, S-Corporation, LLC, LP, Partnership, Sole Proprietorship)
  group_name (TEXT): business group affiliation, NULL if standalone
  state (TEXT): state of registered office
  product_type (TEXT): Term Loan, Working Capital, Cash Credit, Overdraft, Bill Discounting, Letter of Credit
  loan_category (TEXT): Corporate, Mid-Corporate, SME
  sector_code (TEXT, FK): joins to sector_master.sector_code
  sanctioned_amount (REAL): original approved loan amount in USD
  disbursed_amount (REAL): total amount disbursed in USD
  outstanding_principal (REAL): current principal outstanding in USD
  outstanding_interest (REAL): accrued interest outstanding in USD
  total_outstanding (REAL): outstanding_principal + outstanding_interest in USD
  interest_rate (REAL): current interest rate as percentage
  rate_type (TEXT): Fixed, Floating, MCLR-linked, Repo-linked
  sanction_date (DATE): date of original sanction
  maturity_date (DATE): contractual maturity date
  repayment_frequency (TEXT): Monthly, Quarterly, Bullet
  branch_code (TEXT): originating branch identifier
  branch_name (TEXT): originating branch name
  relationship_manager (TEXT): assigned relationship manager name
  is_consortium (INTEGER): 1 if consortium loan, 0 otherwise
  is_restructured (INTEGER): 1 if restructured, 0 otherwise
  restructuring_date (DATE): date of last restructuring, NULL if not restructured
  is_secured (INTEGER): 1 if secured, 0 if unsecured
  days_past_due (INTEGER): current maximum days past due for the loan
  asset_classification (TEXT): Pass, Special Mention, Substandard, Doubtful, Loss
  classification_date (DATE): date current classification was assigned

borrower_rating:
  rating_id (INTEGER, PK): auto-increment identifier
  borrower_id (TEXT, FK): joins to loan_master.borrower_id
  rating_date (DATE): date of rating assessment
  internal_rating (TEXT): bank's internal rating grade (AAA through D, 18-grade scale)
  previous_rating (TEXT): rating grade from prior assessment
  rating_direction (TEXT): Upgraded, Downgraded, Maintained
  external_rating_agency (TEXT): S&P, Moody's, Fitch, DBRS Morningstar, Kroll, or NULL
  external_rating (TEXT): external agency rating
  pd_estimate (REAL): probability of default (decimal, e.g., 0.02 for 2%)
  rating_model_version (TEXT): internal rating model version

provisioning:
  provision_id (INTEGER, PK): auto-increment identifier
  loan_account_number (TEXT, FK): joins to loan_master.loan_account_number
  reporting_date (DATE): quarter-end reporting date
  ifrs9_stage (INTEGER): IFRS 9 stage (1, 2, or 3)
  stage_rationale (TEXT): reason for stage assignment
  pd_12_month (REAL): 12-month probability of default
  pd_lifetime (REAL): lifetime probability of default
  lgd_estimate (REAL): loss given default (decimal)
  ead_amount (REAL): exposure at default in USD
  ecl_amount (REAL): expected credit loss in USD
  provision_held (REAL): provision amount held in USD
  provision_coverage_ratio (REAL): provision_held / total_outstanding * 100
  is_individually_assessed (INTEGER): 1 if individually assessed, 0 if modeled

Available reporting_date values in provisioning: 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Available rating_date values in borrower_rating: 2024-09-30, 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Latest reporting_date: 2025-09-30
Latest rating_date: 2025-09-30
NPA definition: asset_classification IN ('Substandard', 'Doubtful', 'Loss')
"""

# ----------------------------------------------------------------------------
# Verified Query Template Library
# ----------------------------------------------------------------------------

# VQ1: Sector-wise Outstanding and NPA Breakdown
sql_1 = """
SELECT
    sm.sector_name,
    ROUND(SUM(lm.total_outstanding) / 1000000.0, 2) AS total_outstanding_millions,
    ROUND(
        SUM(
            CASE
                WHEN lm.asset_classification IN ('Substandard', 'Doubtful', 'Loss')
                THEN lm.total_outstanding
                ELSE 0
            END
        ) / 1000000.0, 2
    ) AS npa_exposure_millions
FROM loan_master lm
JOIN sector_master sm
    ON lm.sector_code = sm.sector_code
GROUP BY sm.sector_name
ORDER BY total_outstanding_millions DESC
"""

# VQ2: Portfolio Outstanding by Loan Category
sql_2 = """
SELECT
    loan_category,
    COUNT(*) AS loan_count,
    ROUND(SUM(total_outstanding) / 1000000.0, 2) AS total_outstanding_millions
FROM loan_master
GROUP BY loan_category
ORDER BY total_outstanding_millions DESC
"""

# VQ3: IFRS 9 Stage-wise ECL Summary
sql_3 = """
SELECT
    ifrs9_stage,
    COUNT(*) AS loan_count,
    ROUND(SUM(ead_amount) / 1000000.0, 2) AS total_ead_millions,
    ROUND(SUM(ecl_amount) / 1000000.0, 2) AS total_ecl_millions
FROM provisioning
WHERE reporting_date = '2025-09-30'
GROUP BY ifrs9_stage
ORDER BY ifrs9_stage ASC
"""

# VQ4: Provision Coverage Ratio by Sector
sql_4 = """
SELECT
    sm.sector_name,
    ROUND(AVG(p.provision_coverage_ratio), 2) AS avg_provision_coverage_ratio
FROM provisioning p
JOIN loan_master lm
    ON p.loan_account_number = lm.loan_account_number
JOIN sector_master sm
    ON lm.sector_code = sm.sector_code
WHERE p.reporting_date = '2025-09-30'
GROUP BY sm.sector_name
ORDER BY avg_provision_coverage_ratio DESC
"""

# VQ5: Top 10 Loan Exposures
sql_5 = """
SELECT
    lm.borrower_name,
    sm.sector_name,
    ROUND(lm.total_outstanding / 1000000.0, 2) AS outstanding_millions,
    lm.asset_classification
FROM loan_master lm
JOIN sector_master sm
    ON lm.sector_code = sm.sector_code
ORDER BY lm.total_outstanding DESC
LIMIT 10
"""

# VQ6: Top 5 Business Group Exposures
sql_6 = """
SELECT
    group_name,
    COUNT(*) AS loan_count,
    ROUND(SUM(total_outstanding) / 1000000.0, 2) AS total_exposure_millions
FROM loan_master
WHERE group_name IS NOT NULL
  AND TRIM(group_name) != ''
GROUP BY group_name
ORDER BY SUM(total_outstanding) DESC
LIMIT 5
"""

# VQ7: All Overdue Loan Accounts
sql_7 = """
SELECT
    lm.loan_account_number,
    lm.borrower_name,
    sm.sector_name,
    ROUND(lm.total_outstanding / 1000000.0, 2) AS outstanding_millions,
    lm.days_past_due,
    lm.asset_classification
FROM loan_master lm
JOIN sector_master sm
    ON lm.sector_code = sm.sector_code
WHERE lm.days_past_due > 0
ORDER BY lm.days_past_due DESC
"""

# VQ8: DPD Bucket Distribution
sql_8 = """
SELECT
    CASE
        WHEN days_past_due = 0 THEN '0 (Current)'
        WHEN days_past_due BETWEEN 1 AND 30 THEN '1-30'
        WHEN days_past_due BETWEEN 31 AND 60 THEN '31-60'
        WHEN days_past_due BETWEEN 61 AND 90 THEN '61-90'
        ELSE '90+'
    END AS dpd_bucket,
    COUNT(*) AS loan_count,
    ROUND(SUM(total_outstanding) / 1000000.0, 2) AS total_exposure_millions
FROM loan_master
GROUP BY dpd_bucket
ORDER BY
    CASE dpd_bucket
        WHEN '0 (Current)' THEN 1
        WHEN '1-30' THEN 2
        WHEN '31-60' THEN 3
        WHEN '61-90' THEN 4
        WHEN '90+' THEN 5
    END ASC
"""

# VQ9: Latest Rating Downgrades
sql_9 = """
SELECT
    borrower_id,
    previous_rating,
    internal_rating,
    pd_estimate
FROM borrower_rating
WHERE rating_date = '2025-09-30'
  AND rating_direction = 'Downgraded'
ORDER BY pd_estimate DESC
"""

# VQ10: ECL Trend Across Reporting Quarters
sql_10 = """
SELECT
    reporting_date,
    ROUND(SUM(ecl_amount) / 1000000.0, 2) AS total_ecl_millions
FROM provisioning
GROUP BY reporting_date
ORDER BY reporting_date ASC
"""

# NOTE: the VQ1 and VQ8 descriptions in the notebook were built from adjacent string
# literals that were missing separating spaces ("...sectors(including..." / "...booksize...").
# Spaces were added here; the wording is otherwise unchanged.
verified_query_library = {
    'VQ1': {
        'description': 'Sector-wise total outstanding exposure and NPA (non-performing asset) breakdown across all sectors '
                       '(including Real Estate, Infrastructure, Manufacturing, etc.). Can answer questions about total book '
                       'size and NPA exposure for any individual sector or all sectors',
        'sql': sql_1
    },
    'VQ2': {
        'description': 'Total portfolio outstanding broken down by loan category (Corporate, Mid-Corporate, SME)',
        'sql': sql_2
    },
    'VQ3': {
        'description': 'IFRS 9 stage-wise summary showing loan count, exposure at default, and expected credit loss for the latest quarter',
        'sql': sql_3
    },
    'VQ4': {
        'description': 'Average provision coverage ratio by sector for the latest reporting quarter',
        'sql': sql_4
    },
    'VQ5': {
        'description': 'Top 10 largest loan exposures by outstanding amount at the borrower level',
        'sql': sql_5
    },
    'VQ6': {
        'description': 'Top 5 largest exposures aggregated at the business group level',
        'sql': sql_6
    },
    'VQ7': {
        'description': 'All overdue loan accounts with their days past due and asset classification',
        'sql': sql_7
    },
    'VQ8': {
        'description': 'Distribution of overdue and current loans across Days Past Due (DPD) aging buckets '
                       '(0 Current, 1-30, 31-60, 61-90, 90+ days) showing loan count and outstanding exposure',
        'sql': sql_8
    },
    'VQ9': {
        'description': 'Borrowers whose internal rating was downgraded in the latest rating cycle',
        'sql': sql_9
    },
    'VQ10': {
        'description': 'Expected credit loss trend across all reporting quarters showing provisioning movement over time',
        'sql': sql_10
    }
}


# ----------------------------------------------------------------------------
# Tool 1: Intent classification
# ----------------------------------------------------------------------------
def classify_intent(user_question, query_library):
    library_descriptions = "\n".join(
        [f"{qid}: {entry['description']}" for qid, entry in query_library.items()]
    )

    classification_prompt = (
        "You are a precision credit risk analytics intent router. Decide whether the user's question "
        "can be answered using the data returned by one of the pre-approved queries in the Verified Query "
        "Template Library, or if it requires generating a new SQL query.\n\n"
        "### VERIFIED QUERY TEMPLATE LIBRARY\n"
        f"{library_descriptions}\n\n"
        "### ROUTING RULES\n"
        "1. SUBSET RULE (CRITICAL): Each verified query returns a complete breakdown table (e.g., all sectors, "
        "all loan categories, all DPD buckets). If the user asks about ONE specific sector, category, or bucket "
        "using the exact metrics already calculated by a template, you MUST route to 'verified' with that query_id.\n"
        "2. SYNONYM RULE: Treat 'non-performing', 'NPA', and 'bad loans' as identical. Treat 'overdue book', "
        "'delinquency aging', and 'DPD buckets' as identical.\n"
        "3. GENERATED RULE: Route to 'generated' ONLY when the user asks for columns, flags, ratios, or multi-quarter "
        "trends that do NOT exist in any template output (for example: restructured loans, impairment percentage, "
        "average interest rates, or multi-quarter Stage 3 trends).\n\n"
        "### EXAMPLES\n"
        "- Question: 'How much of our book is in real estate, and how much of it is non-performing?'\n"
        "  Output: {\"route\": \"verified\", \"query_id\": \"VQ1\", \"match_reason\": \"VQ1 provides total outstanding and NPA exposure by sector, which includes Real Estate.\"}\n"
        "- Question: 'Show me how our overdue book is distributed across DPD buckets.'\n"
        "  Output: {\"route\": \"verified\", \"query_id\": \"VQ8\", \"match_reason\": \"VQ8 calculates loan counts and exposure across all DPD buckets.\"}\n"
        "- Question: 'List every restructured loan, flag which are impaired, and show the overall impaired percentage.'\n"
        "  Output: {\"route\": \"generated\", \"query_id\": null, \"match_reason\": \"No template lists restructured loans or calculates impairment percentages.\"}\n"
        "- Question: 'How has the Stage 3 book size and its expected credit loss moved over the last four quarters?'\n"
        "  Output: {\"route\": \"generated\", \"query_id\": null, \"match_reason\": \"VQ3 only covers the latest quarter, whereas this asks for a 4-quarter trend.\"}\n\n"
        "### USER QUESTION\n"
        f"{user_question}\n\n"
        "### OUTPUT\n"
        "Return ONLY a valid JSON dictionary with keys 'route', 'query_id', and 'match_reason'. Do not include any other text."
    )

    response = llm.invoke(classification_prompt).content.strip()

    json_match = re.search(r"\{.*\}", response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass  # fall through to the safe default below

    return {"route": "generated", "query_id": None, "match_reason": "Could not parse classification"}


# ----------------------------------------------------------------------------
# Tool 2: Query generation
# NOTE: the notebook's pipeline calls `generate_query`, but the notebook cell titled
# "Query Generation Tool" actually contained a duplicate of `retry_generation`, so
# `generate_query` was never defined. It is implemented here using the same prompt
# style and rules as the notebook's retry prompt.
# ----------------------------------------------------------------------------
def _clean_sql(raw_text):
    """Strip markdown fences / preamble and return raw SQL starting at SELECT or WITH."""
    cleaned_sql = re.sub(r"```(?:sql)?", "", raw_text, flags=re.IGNORECASE)
    cleaned_sql = cleaned_sql.replace("```", "").strip()

    match = re.search(r"\b(SELECT|WITH)\b.*", cleaned_sql, flags=re.IGNORECASE | re.DOTALL)
    if match:
        cleaned_sql = match.group(0).strip()

    return cleaned_sql


def generate_query(user_question, schema_context):
    generation_prompt = (
        "You are an expert SQLite data analyst specializing in banking and credit risk. "
        "Write ONE read-only SQLite query that answers the user's question using only the database schema below.\n\n"
        "### DATABASE SCHEMA\n"
        f"{schema_context}\n\n"
        "### USER QUESTION\n"
        f"{user_question}\n\n"
        "### SQL GENERATION RULES\n"
        "1. Output ONLY raw SQL starting immediately with SELECT or WITH.\n"
        "2. Do NOT include any introductory text, explanations, labels, or markdown code blocks.\n"
        "3. Use valid SQLite syntax only and exact table/column names from the schema.\n"
        "4. The query must be a single read-only statement (no INSERT, UPDATE, DELETE, DROP, ALTER, ATTACH, etc.).\n"
        "5. NPA (non-performing) means asset_classification IN ('Substandard', 'Doubtful', 'Loss').\n"
        "6. Convert USD amounts to millions using ROUND(value / 1000000.0, 2) and name such columns with a '_millions' suffix.\n"
        "7. For 'latest' provisioning use reporting_date = '2025-09-30'; for 'latest' ratings use rating_date = '2025-09-30'. "
        "For trends over quarters include every reporting_date and order chronologically.\n"
        "8. Round percentages and averages to 2 decimal places.\n"
        "9. Join sector_master on sector_code and return sector_name (not the code) when sectors are requested.\n"
    )

    response = llm.invoke(generation_prompt).content.strip()
    return _clean_sql(response)


# ----------------------------------------------------------------------------
# Tool 3: Query validation (5 checks)
# ----------------------------------------------------------------------------
def validate_query(user_question, candidate_sql, db_connection, query_library, query_id=None):
    '''
    Validates a candidate SQL query through five checks before execution.

    Parameters:
    - user_question (str): The original user question.
    - candidate_sql (str): The SQL query to validate.
    - db_connection: SQLite connection object.
    - query_library (dict): Verified query library (for integrity check).
    - query_id (str, optional): Template ID if from verified track.

    Returns:
    - dict: Contains 'passed' (bool), 'failed_check' (str or None), 'details' (str),
            and 'relevance_confidence' (int, 0-1).
    '''

    result = {
        'passed': False,
        'failed_check': None,
        'details': '',
        'relevance_confidence': None,
        'unknown_identifiers': []
    }

    # Check 1: Read-only shape check
    sql_upper = candidate_sql.upper().strip()
    forbidden_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'TRUNCATE', 'REPLACE', 'ATTACH']
    if not (sql_upper.startswith('SELECT') or sql_upper.startswith('WITH')):
        result['failed_check'] = 'read_only_shape'
        result['details'] = 'Query must start with SELECT or WITH'
        return result
    for kw in forbidden_keywords:
        if re.search(r'\b' + kw + r'\b', sql_upper):
            result['failed_check'] = 'read_only_shape'
            result['details'] = f'Forbidden keyword detected: {kw}'
            return result
    if ';' in candidate_sql.rstrip(';').rstrip():
        result['failed_check'] = 'read_only_shape'
        result['details'] = 'Multiple statements are not allowed'
        return result

    # Check 2: Schema conformance check
    cur = db_connection.cursor()
    real_tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    real_columns = set()
    for t in real_tables:
        for col_info in cur.execute(f"PRAGMA table_info({t})").fetchall():
            real_columns.add(col_info[1].lower())
    parsed = sqlparse.parse(candidate_sql)[0]
    tokens = [str(t).strip().lower() for t in parsed.flatten() if t.ttype is None or 'Name' in str(t.ttype)]
    referenced_identifiers = re.findall(r'\b[a-z_][a-z0-9_]*\b', candidate_sql.lower())
    sql_keywords = {'select', 'from', 'where', 'and', 'or', 'group', 'by', 'order', 'having', 'limit', 'join', 'on', 'as', 'case',
                    'when', 'then', 'else', 'end', 'sum', 'count', 'avg', 'min', 'max', 'round', 'desc', 'asc', 'left', 'right',
                    'inner', 'outer', 'distinct', 'null', 'is', 'not', 'in', 'like', 'with', 'union', 'all', 'between', 'coalesce'}
    unknown = [tok for tok in referenced_identifiers
               if tok not in sql_keywords and tok not in real_columns and tok not in real_tables
               and not tok.isdigit() and tok not in ('s', 'l', 'p', 'r', 'e6')]
    # As in the notebook, `unknown` is advisory (aliases and string literals produce
    # false positives) - the hard schema check is the EXPLAIN dry run in Check 3.
    # It is recorded in the audit trail for reviewers.
    result['unknown_identifiers'] = sorted(set(unknown))

    # Check 3: Parse-and-plan dry run using EXPLAIN
    try:
        cur.execute(f"EXPLAIN {candidate_sql}")
        cur.fetchall()
    except sqlite3.Error as e:
        result['failed_check'] = 'parse_plan_dry_run'
        result['details'] = f'SQL failed to parse or plan: {str(e)}'
        return result

    # Check 4: LLM relevance check
    is_verified_track = query_id is not None and query_id in query_library
    track_context = (
        "This SQL is a pre-approved VERIFIED TEMPLATE. It is intentionally broad "
        "(e.g., it may return all sectors/categories/stages rather than filtering to "
        "just what the user asked). A separate response-generation step will filter and "
        "highlight the relevant rows afterward. Do NOT fail this query for lacking a "
        "WHERE clause that narrows to the user's specific sector/category/stage — judge "
        "only whether the underlying metric, tables, and aggregation logic match the "
        "question's intent."
        if is_verified_track else
        "This SQL was freshly generated for this specific question and should be "
        "appropriately scoped/filtered to answer it directly."
    )

    relevance_prompt = f"""You are a strict SQL query validation judge for a banking and credit risk database.
Your job is to evaluate whether the candidate SQL query accurately answers the user's question.

{track_context}

### USER QUESTION
{user_question}

### CANDIDATE SQL
{candidate_sql}

### OUTPUT
Return ONLY a JSON dictionary:
{{
  "verdict": "yes" or "no",
  "confidence": 0.0 to 1.0,
  "reason": "one short sentence"
}}

"""
    relevance_response = evaluator_llm.invoke(relevance_prompt).content.strip()
    json_match = re.search(r'\{.*\}', relevance_response, re.DOTALL)
    relevance_json = None
    if json_match:
        try:
            relevance_json = json.loads(json_match.group())
        except json.JSONDecodeError:
            relevance_json = None

    if relevance_json is None:
        # Fail closed: an unparseable judge response must not silently approve a query.
        result['failed_check'] = 'llm_relevance'
        result['details'] = 'Relevance check failed: could not parse the validation judge response'
        return result

    try:
        confidence = float(relevance_json.get('confidence', 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    result['relevance_confidence'] = confidence
    if str(relevance_json.get('verdict', '')).strip().lower() == 'no' or confidence < 0.6:
        result['failed_check'] = 'llm_relevance'
        result['details'] = f"Relevance check failed: {relevance_json.get('reason', 'unknown')}"
        return result

    # Check 5: Verified template integrity check (verified track only)
    if query_id and query_id in query_library:
        expected_sql = query_library[query_id]['sql']
        try:
            # Wrapped as a subquery so templates that already end in LIMIT (VQ5, VQ6)
            # don't break when the notebook's "LIMIT 0" probe is applied.
            expected_body = expected_sql.strip().rstrip(';')
            actual_body = candidate_sql.strip().rstrip(';')
            expected_cols = [d[0] for d in cur.execute(f"SELECT * FROM ({expected_body}) LIMIT 0").description]
            actual_cols = [d[0] for d in cur.execute(f"SELECT * FROM ({actual_body}) LIMIT 0").description]
            if len(expected_cols) != len(actual_cols):
                result['failed_check'] = 'template_integrity'
                result['details'] = f'Expected {len(expected_cols)} columns, got {len(actual_cols)}'
                return result
        except sqlite3.Error as e:
            result['failed_check'] = 'template_integrity'
            result['details'] = f'Template integrity check failed: {str(e)}'
            return result

    result['passed'] = True
    result['details'] = 'All validation checks passed'
    return result


# ----------------------------------------------------------------------------
# Tool 4: Retry generation
# ----------------------------------------------------------------------------
def retry_generation(user_question, failed_sql, error_message, schema_context):
    '''
    Regenerates SQL after a validation failure, feeding the error back to the LLM.

    Parameters:
    - user_question (str): The original user question.
    - failed_sql (str): The SQL that failed validation.
    - error_message (str): The specific failure reason.
    - schema_context (str): Database schema description.

    Returns:
    - str: Revised SQL as a string.
    '''

    retry_prompt = f"""You are an expert SQLite data analyst specializing in banking and credit risk.
Your previous attempt to write a SQLite query for the user's question failed validation. Your task is to analyze the error message, fix the query, and return a corrected version.

User Question:
{user_question}

Failed SQL:
{failed_sql}

Validation Error:
{error_message}

Database Schema:
{schema_context}

Output ONLY the corrected raw SQL (a single read-only SELECT/WITH statement) with no explanation and no markdown.
"""

    revised_sql = llm.invoke(retry_prompt).content.strip()
    revised_sql = re.sub(r'^```sql\s*|\s*```$', '', revised_sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    revised_sql = re.sub(r'^```\s*|\s*```$', '', revised_sql, flags=re.MULTILINE).strip()
    return _clean_sql(revised_sql)


# ----------------------------------------------------------------------------
# Tool 5: Query execution
# ----------------------------------------------------------------------------
def execute_query(validated_sql, db_connection):
    '''
    Executes a gate-passed SQL query and returns the result as a DataFrame.

    Parameters:
    - validated_sql (str): SQL query that has passed all validation checks.
    - db_connection: Read-only SQLite connection object.

    Returns:
    - dict: Contains 'dataframe' (pandas DataFrame), 'reasonable' (bool),
            and 'warnings' (list of warning strings).
    '''

    result = {
        'dataframe': None,
        'reasonable': True,
        'warnings': []
    }

    df = pd.read_sql_query(validated_sql, db_connection)
    result['dataframe'] = df

    # Reasonableness checks
    if df.empty:
        result['warnings'].append('Query returned an empty result')

    for col in df.select_dtypes(include='number').columns:
        if (df[col] < 0).any() and 'deviation' not in col.lower() and 'change' not in col.lower():
            result['warnings'].append(f'Column {col} contains negative values')
        if df[col].isnull().any():
            null_count = df[col].isnull().sum()
            if null_count > len(df) * 0.5:
                result['warnings'].append(f'Column {col} has {null_count} null values')

    if len(result['warnings']) > 2:
        result['reasonable'] = False

    return result


# ----------------------------------------------------------------------------
# Tool 6: Response generation
# ----------------------------------------------------------------------------
def generate_response(user_question, dataframe, route, query_id=None):
    '''
    Generates a focused natural language response from the query result.

    Parameters:
    - user_question (str): The original user question.
    - dataframe (pd.DataFrame): The full query result.
    - route (str): 'verified' or 'generated'.
    - query_id (str, optional): Template ID if from verified track.

    Returns:
    - str: Natural language response focused on what the user asked.
    '''

    response_prompt = f"""You are a senior banking and credit risk analyst presenting query results to a stakeholder.
Your task is to write a clear, accurate, and concise natural language answer to the user's question based strictly on the provided SQL query results.

### USER QUESTION
{user_question}

### SQL QUERY RESULT
{dataframe.to_string()}

"""

    narrative = llm.invoke(response_prompt).content.strip()
    return narrative


# ----------------------------------------------------------------------------
# Pipeline orchestration
# ----------------------------------------------------------------------------
def run_pipeline(user_question, db_connection, query_library, schema_context, verbose=True):
    '''
    Runs the complete query engine pipeline for a single user question.

    Parameters:
    - user_question (str): The natural language question.
    - db_connection: SQLite connection object.
    - query_library (dict): Verified query template library.
    - schema_context (str): Database schema description.
    - verbose (bool): If True, records intermediate pipeline stages in log['trace']
                      (replaces the notebook's print() statements).

    Returns:
    - dict: Complete pipeline output including narrative, SQL, data, and log.
    '''

    log = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'user_question': user_question,
        'route': None,
        'query_id': None,
        'match_reason': None,
        'candidate_sql': None,
        'gate_result': None,
        'retry_used': False,
        'escalated': False,
        'executed_sql': None,
        'row_count': None,
        'warnings': [],
        'confidence': None,
        'narrative': None,
        'trace': []
    }

    def say(message):
        if verbose:
            log['trace'].append(message)

    # Step 1: Intent classification
    classification = classify_intent(user_question, query_library)
    log['route'] = classification.get('route')
    log['query_id'] = classification.get('query_id')
    log['match_reason'] = classification.get('match_reason')

    say(f"[1] Intent Classification: route={log['route']}, query_id={log['query_id']}")
    say(f"    Reason: {log['match_reason']}")

    # Step 2: Query construction
    if log['route'] == 'verified' and log['query_id'] in query_library:
        candidate_sql = query_library[log['query_id']]['sql']
    else:
        # A 'verified' route pointing at an unknown query_id falls back to generation
        if log['route'] == 'verified':
            log['route'] = 'generated'
            log['query_id'] = None
        candidate_sql = generate_query(user_question, schema_context)
    log['candidate_sql'] = candidate_sql

    say(f"[2] Query Construction: {'loaded from library' if log['route'] == 'verified' else 'generated fresh SQL'}")

    # Step 3: Validation gate
    gate = validate_query(user_question, candidate_sql, db_connection, query_library, log['query_id'])
    log['gate_result'] = gate

    say(f"[3] Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
    if not gate['passed']:
        say(f"    Failed check: {gate.get('failed_check')}")
        say(f"    Details: {gate.get('details')}")

    # Step 4: Retry once on generated track if validation fails
    if not gate['passed'] and log['route'] == 'generated':
        say(f"    Retrying: {gate['details']}")
        candidate_sql = retry_generation(user_question, candidate_sql, gate['details'], schema_context)
        log['candidate_sql'] = candidate_sql
        log['retry_used'] = True
        gate = validate_query(user_question, candidate_sql, db_connection, query_library, None)
        log['gate_result'] = gate

        say(f"    Retry Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
        if not gate['passed']:
            say(f"    Retry failed check: {gate.get('failed_check')}")
            say(f"    Retry details: {gate.get('details')}")

    # Step 5: Escalate if still failing
    if not gate['passed']:
        log['escalated'] = True
        log['narrative'] = f"Query could not be reliably resolved. Escalated to human analyst. Failure: {gate['details']}"
        log['confidence'] = 'ESCALATED'
        say(f"[!] Escalated to human: {gate['details']}")
        return {'log': log, 'dataframe': None, **log}

    # Step 6: Execute
    log['executed_sql'] = candidate_sql
    exec_result = execute_query(candidate_sql, db_connection)
    df = exec_result['dataframe']
    log['row_count'] = len(df)
    log['warnings'] = exec_result['warnings']

    say(f"[4] Execute: {len(df)} rows returned")
    if exec_result['warnings']:
        say(f"    Warnings: {exec_result['warnings']}")

    # Step 7: Response generation
    narrative = generate_response(user_question, df, log['route'], log['query_id'])
    log['narrative'] = narrative

    # Confidence: carried directly from the validation gate's relevance check (0-1)
    log['confidence'] = gate.get('relevance_confidence')

    say(f"[6] Response Generation: confidence={log['confidence']}")

    return {'log': log, 'dataframe': df, **log}


# ----------------------------------------------------------------------------
# Streamlit helpers (audit trail, safe pipeline call, rendering)
# ----------------------------------------------------------------------------
def run_pipeline_safe(user_question):
    """Open a fresh read-only connection, run the pipeline, always close the connection."""
    conn = get_connection()
    try:
        return run_pipeline(user_question, conn, verified_query_library, database_schema, verbose=True)
    finally:
        conn.close()


def audit_record(result):
    """Build a JSON-serialisable audit record for one analytical output."""
    record = dict(result['log'])
    record['executed_sql'] = result.get('executed_sql')
    return json.loads(json.dumps(record, default=str))


def write_audit(result):
    """Append the audit record to the session history and (best effort) to a JSONL file."""
    record = audit_record(result)
    st.session_state.setdefault('audit_history', []).append(record)
    try:
        with open(AUDIT_LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass  # filesystem may be read-only / ephemeral; the session history still holds the record
    return record


def render_result(result, key_prefix="result"):
    """Display the answer, SQL used, raw data and confidence score for one pipeline result."""
    route = result.get('route')
    query_id = result.get('query_id')
    confidence = result.get('confidence')

    if result.get('escalated'):
        st.error("☐  Escalated to a human analyst")
        st.write(result.get('narrative'))
    else:
        st.subheader("Answer")
        st.write(result.get('narrative'))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Route", str(route).title() if route else "-")
    c2.metric("Verified Query ID", query_id if query_id else "-")
    if isinstance(confidence, (int, float)):
        c3.metric("Confidence", f"{confidence:.2f}")
    else:
        c3.metric("Confidence", str(confidence) if confidence is not None else "n/a")
    c4.metric("Rows returned", result.get('row_count') if result.get('row_count') is not None else "-")

    if isinstance(confidence, (int, float)):
        st.progress(min(max(float(confidence), 0.0), 1.0))

    if result.get('retry_used'):
        st.info("A retry was used: the first generated SQL failed validation and was regenerated once.")
    for w in result.get('warnings') or []:
        st.warning(w)

    sql_to_show = result.get('executed_sql') or result.get('candidate_sql')
    if sql_to_show:
        st.markdown("**SQL query used**" if result.get('executed_sql') else "**Candidate SQL (not executed)**")
        st.code(sql_to_show.strip(), language="sql")

    df = result.get('dataframe')
    if df is not None:
        st.markdown("**Raw data returned**")
        st.dataframe(df)
        st.download_button(
            "Download result as CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name="query_result.csv",
            mime="text/csv",
            key=f"{key_prefix}_download",
        )

    with st.expander("Routing & validation details (audit)"):
        st.write(f"**Match reason:** {result.get('match_reason')}")
        gate = result.get('gate_result') or {}
        st.write(
            f"**Validation gate:** passed={gate.get('passed')}, "
            f"failed_check={gate.get('failed_check')}, "
            f"relevance_confidence={gate.get('relevance_confidence')}"
        )
        st.write(f"**Gate details:** {gate.get('details')}")
        if gate.get('unknown_identifiers'):
            st.caption(
                "Identifiers not found in the schema (advisory only; aliases and literals can appear here): "
                + ", ".join(gate['unknown_identifiers'])
            )
        if result['log'].get('trace'):
            st.code("\n".join(result['log']['trace']), language="text")


def load_ground_truth():
    try:
        if Path(TEST_QUERIES_PATH).exists():
            return pd.read_csv(TEST_QUERIES_PATH)
    except Exception:
        pass
    return None


# ----------------------------------------------------------------------------
# App body
# ----------------------------------------------------------------------------
st.title("   Credit Risk Query Engine")
st.caption(
    "Northbridge Bank  Commercial lending portfolio  Read-only natural-language analytics "
    "with verified templates, SQL validation and a full audit trail."
)

has_key, api_base = load_credentials()

if not has_key:
    st.error(
        "OpenAI API key not found. Set `OPENAI_API_KEY` in Streamlit secrets "
        "(`.streamlit/secrets.toml`), as an environment variable, or in a `config.json` file."
    )
    st.stop()

if not Path(DB_PATH).exists():
    st.error(f"Database file not found at `{DB_PATH}`. Place `credit_risk_portfolio.db` next to `app.py`.")
    st.stop()

# LLMs are module-level globals used by the tool functions (same as the notebook)
llm, evaluator_llm = init_llms(LLM_MODEL, EVALUATOR_MODEL, os.environ.get("OPENAI_API_KEY", "")[-6:])

ground_truth = load_ground_truth()

# Fallback example questions (used only if test_queries.csv is not available)
DEFAULT_EXAMPLES = [
    "How much of our book is in real estate, and how much of it is non-performing?",
    "Show me how our overdue book is distributed across DPD buckets.",
    "List every restructured loan, flag which are impaired, and show the overall impaired percentage.",
    "Show me the average interest rate by sector, sorted from highest to lowest.",
    "How has the Stage 3 book size and its expected credit loss moved over the last four quarters?",
]
if ground_truth is not None and 'User Query' in ground_truth.columns:
    example_questions = ground_truth['User Query'].dropna().astype(str).tolist()
else:
    example_questions = DEFAULT_EXAMPLES


def _set_question(q):
    st.session_state['question_input'] = q


# ---- Sidebar ----
with st.sidebar:
    st.header("About")
    st.markdown(
        "- **Verified track:** pre-approved SQL templates (VQ1–VQ10)\n"
        "- **Generated track:** LLM-written SELECT-only SQL\n"
        "- **Validation gate:** read-only shape, schema, EXPLAIN plan, LLM relevance, template integrity\n"
        "- **Retry once**, then **escalate** to a human analyst\n"
        "- **Audit trail** for every output"
    )
    st.divider()
    st.markdown(f"**Models:** `{LLM_MODEL}` (generation)  `{EVALUATOR_MODEL}` (validation)")
    st.markdown("**Database:** read-only SQLite")
    if api_base:
        st.markdown(f"**API base:** `{api_base}`")
    st.divider()
    st.subheader("Example questions")
    for i, q in enumerate(example_questions):
        st.button(q, key=f"example_{i}", on_click=_set_question, args=(q,))

tab_ask, tab_library, tab_eval, tab_audit = st.tabs(
    ["   Ask a question", "   Verified query library", "   Evaluation", "   Audit trail"]
)

# ---- Tab 1: Ask ----
with tab_ask:
    st.text_area(
        "Ask a question about the commercial lending portfolio",
        key="question_input",
        height=100,
        placeholder="e.g. How much of our book is in real estate, and how much of it is non-performing?",
    )
    run_clicked = st.button("Run query", type="primary")

    if run_clicked:
        question = (st.session_state.get('question_input') or "").strip()
        if not question:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Running the query engine pipeline..."):
                try:
                    res = run_pipeline_safe(question)
                    write_audit(res)
                    st.session_state['last_result'] = res
                except Exception as e:
                    st.session_state['last_result'] = None
                    st.error(f"Pipeline error: {e}")

    if st.session_state.get('last_result'):
        render_result(st.session_state['last_result'], key_prefix="ask")

# ---- Tab 2: Verified query library ----
with tab_library:
    st.subheader("Verified Query Template Library")
    st.caption("Pre-approved, tested SQL templates used for recurring questions.")
    for qid, entry in verified_query_library.items():
        with st.expander(f"{qid}: {entry['description']}"):
            st.code(entry['sql'].strip(), language="sql")

# ---- Tab 3: Evaluation against ground truth ----
with tab_eval:
    st.subheader("Evaluation Against Ground Truth")
    if ground_truth is None:
        st.info("Place `test_queries.csv` next to `app.py` to enable the evaluation against ground truth.")
    else:
        st.dataframe(ground_truth)
        if st.button("Run all test cases"):
            test_results = []
            progress = st.progress(0.0)
            for i, (_, gt) in enumerate(ground_truth.iterrows()):
                with st.spinner(f"Running {gt['Test Case']}..."):
                    try:
                        tr = run_pipeline_safe(gt['User Query'])
                        write_audit(tr)
                    except Exception as e:
                        tr = {'route': None, 'query_id': None, 'confidence': None, 'row_count': None, 'failed': True}
                        st.error(f"{gt['Test Case']} failed: {e}")
                test_results.append(tr)
                progress.progress((i + 1) / len(ground_truth))

            evaluation_rows = []
            for i, (_, gt) in enumerate(ground_truth.iterrows()):
                tr = test_results[i]
                evaluation_rows.append({
                    'Test Case': gt['Test Case'],
                    'Expected Route': gt['Expected Route'],
                    'Actual Route': tr['route'],
                    'Route Match': (
                        str(tr['route']).strip().lower() == str(gt['Expected Route']).strip().lower()
                    ),
                    'Expected Query ID': gt['Expected Query ID'],
                    'Actual Query ID': tr['query_id'],
                    'Query ID Match': (
                        pd.isna(gt['Expected Query ID']) and pd.isna(tr['query_id'])
                    ) or tr['query_id'] == gt['Expected Query ID'],
                    'Confidence': tr['confidence'],
                    'Rows Returned': tr['row_count']
                })

            evaluation_df = pd.DataFrame(evaluation_rows)

            path_accuracy = evaluation_df['Route Match'].mean() * 100

            verified = evaluation_df['Expected Route'].str.strip().str.lower() == 'verified'
            query_accuracy = evaluation_df.loc[verified, 'Query ID Match'].mean() * 100 if verified.any() else float('nan')

            average_confidence = pd.to_numeric(evaluation_df['Confidence'], errors='coerce').mean()

            # Persist so the results survive Streamlit reruns (e.g. clicking another widget)
            st.session_state['eval_state'] = {
                'evaluation_df': evaluation_df,
                'path_accuracy': path_accuracy,
                'query_accuracy': query_accuracy,
                'average_confidence': average_confidence,
                'test_results': test_results,
            }

        eval_state = st.session_state.get('eval_state')
        if eval_state:
            m1, m2, m3 = st.columns(3)
            m1.metric("Selected Path Accuracy", f" {eval_state['path_accuracy']:.1f}%")
            m2.metric("Selected Query Accuracy", f" {eval_state['query_accuracy']:.1f}%")
            m3.metric("Average Confidence Score", f" {eval_state['average_confidence']:.2f}")
            st.dataframe(eval_state['evaluation_df'].astype(str))

            st.markdown("**Per-test-case outputs**")
            for i, (_, gt) in enumerate(ground_truth.iterrows()):
                tr = eval_state['test_results'][i]
                with st.expander(f"{gt['Test Case']}: {gt['User Query']}"):
                    if tr.get('failed'):
                        st.error("This test case raised an error and produced no result.")
                    else:
                        if 'Expected Answer' in ground_truth.columns:
                            st.caption(f"Expected answer: {gt['Expected Answer']}")
                        render_result(tr, key_prefix=f"eval_{i}")

# ---- Tab 4: Audit trail ----
with tab_audit:
    st.subheader("Audit trail (this session)")
    history = st.session_state.get('audit_history', [])
    if not history:
        st.info("No analytical outputs yet. Run a query to populate the audit trail.")
    else:
        summary = pd.DataFrame([{
            'timestamp_utc': h.get('timestamp_utc'),
            'question': h.get('user_question'),
            'route': h.get('route'),
            'query_id': h.get('query_id'),
            'retry_used': h.get('retry_used'),
            'escalated': h.get('escalated'),
            'confidence': h.get('confidence'),
            'rows': h.get('row_count'),
        } for h in history])
        st.dataframe(summary.astype(str))
        st.download_button(
            "Download full audit log (JSON)",
            json.dumps(history, indent=2).encode("utf-8"),
            file_name="audit_log.json",
            mime="application/json",
        )
        with st.expander("Full records"):
            st.json(history)
    st.caption(f"Each output is also appended to `{AUDIT_LOG_PATH}` (best effort) as JSON lines.")
