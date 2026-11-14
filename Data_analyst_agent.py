import os
import json
import re

import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from Data_Loader import load_data


# ==========================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ==========================================================

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY not found. Please check your .env file."
    )


# ==========================================================
# 2. LOAD DATASET
# ==========================================================

df = load_data()

if not isinstance(df, pd.DataFrame):
    raise TypeError("load_data() must return a pandas DataFrame.")

CO2_COL = "CO2 Emissions(g/km)"

if CO2_COL not in df.columns:
    raise ValueError(
        f"Column '{CO2_COL}' not found. "
        f"Available columns: {list(df.columns)}"
    )

# Convert emissions into numeric values
df[CO2_COL] = pd.to_numeric(df[CO2_COL], errors="coerce")

# Remove rows without a valid CO2 value
df = df.dropna(subset=[CO2_COL]).copy()

# Only allow columns that actually exist in the dataset
POSSIBLE_GROUP_COLUMNS = [
    "Make",
    "Model",
    "Vehicle Class",
    "Transmission",
    "Fuel Type",
    "Cylinders"
]

GROUP_COLUMNS = [
    column
    for column in POSSIBLE_GROUP_COLUMNS
    if column in df.columns
]

FILTER_COLUMNS = GROUP_COLUMNS.copy()

if not GROUP_COLUMNS:
    raise ValueError(
        "No supported grouping columns found in the dataset."
    )

print("Dataset loaded successfully.")
print("Available columns:", list(df.columns))
print("Allowed grouping columns:", GROUP_COLUMNS)


# ==========================================================
# 3. INITIALIZE GROQ MODEL
# ==========================================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=api_key
)


# ==========================================================
# 4. ALLOWED AGGREGATIONS
# ==========================================================

AGGREGATIONS = [
    "mean",
    "sum",
    "max",
    "min",
    "count"
]


# ==========================================================
# 5. GENERATE ANALYSIS PLAN
# ==========================================================

def generate_analysis_plan(question: str) -> dict:

    columns_text = ", ".join(GROUP_COLUMNS)

    system_prompt = f"""
You are an expert CO2 emissions data analyst.

You analyse a vehicle emissions dataset using Pandas.

Available dataset columns:

{columns_text}, {CO2_COL}

Your task is to convert the user's question into one JSON
analysis plan.

Allowed operations:

1. overall
2. group_analysis
3. filter_analysis
4. compare

Allowed aggregations:

1. mean
2. sum
3. max
4. min
5. count

Return exactly this JSON structure:

{{
    "operation": "group_analysis",
    "group_by": ["Transmission"],
    "filter_column": null,
    "filter_value": null,
    "aggregation": "mean",
    "top_n": 100,
    "sort": "descending"
}}

IMPORTANT RULES:

1. Use only the available dataset columns listed above.
2. Never invent a column that is not in the dataset.
3. For grouping by multiple columns, include every requested
   column in group_by.
4. For grouping by one column, include only that column.
5. Average or average emissions means aggregation "mean".
6. Total or total emissions means aggregation "sum".
7. Highest or maximum means aggregation "max".
8. Lowest or minimum means aggregation "min".
9. Number of records means aggregation "count".
10. Overall emissions questions use operation "overall"
    and group_by [].
11. Questions asking for a breakdown by categories use
    operation "group_analysis".
12. Questions filtering a particular manufacturer, vehicle
    class, fuel type, transmission or cylinder value use
    operation "filter_analysis".
13. For filter_analysis, provide filter_column and filter_value.
14. If the user asks for an average for a particular Make,
    use filter_analysis with aggregation "mean".
15. If the user asks for results by Transmission, use
    group_analysis and group_by ["Transmission"].
16. If the user asks for results by Make and Transmission,
    use group_analysis and group_by ["Make", "Transmission"].
17. If the user asks for top N, set top_n to N.
18. If no top N is requested, set top_n to 100.
19. Use sort "ascending" for lowest or smallest results.
20. Use sort "descending" for highest or largest results.
21. Do not generate Python code or SQL.
22. Return valid JSON only. Do not include Markdown fences
    or explanations.

Examples:

Question: Average CO2 emissions by Transmission

{{
    "operation": "group_analysis",
    "group_by": ["Transmission"],
    "filter_column": null,
    "filter_value": null,
    "aggregation": "mean",
    "top_n": 100,
    "sort": "descending"
}}

Question: Average CO2 emissions by Make and Fuel Type

{{
    "operation": "group_analysis",
    "group_by": ["Make", "Fuel Type"],
    "filter_column": null,
    "filter_value": null,
    "aggregation": "mean",
    "top_n": 100,
    "sort": "descending"
}}

Question: Average CO2 emissions for Mercedes-Benz

{{
    "operation": "filter_analysis",
    "group_by": [],
    "filter_column": "Make",
    "filter_value": "Mercedes-Benz",
    "aggregation": "mean",
    "top_n": 100,
    "sort": "descending"
}}

Question: Top 10 vehicle classes with highest average CO2 emissions

{{
    "operation": "group_analysis",
    "group_by": ["Vehicle Class"],
    "filter_column": null,
    "filter_value": null,
    "aggregation": "mean",
    "top_n": 10,
    "sort": "descending"
}}
"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)
    ])

    content = response.content

    # Handle different response content formats
    if isinstance(content, list):
        content = "".join(
            item.get("text", "") if isinstance(item, dict)
            else str(item)
            for item in content
        )
    else:
        content = str(content)

    # Remove Markdown code fences if present
    content = re.sub(
        r"```(?:json)?",
        "",
        content,
        flags=re.IGNORECASE
    ).replace("```", "").strip()

    # Extract JSON safely without an invalid regex
    try:
        start = content.index("{")
        plan, _ = json.JSONDecoder().raw_decode(content[start:])
        return plan

    except (ValueError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Could not extract valid JSON from LLM response: {content}"
        ) from exc


# ==========================================================
# 6. VALIDATE ANALYSIS PLAN
# ==========================================================

def validate_plan(plan: dict) -> dict:

    if not isinstance(plan, dict):
        raise ValueError("Analysis plan must be a JSON object.")

    allowed_operations = [
        "overall",
        "group_analysis",
        "filter_analysis",
        "compare"
    ]

    operation = plan.get("operation")

    if operation not in allowed_operations:
        raise ValueError(
            f"Unsupported operation: {operation}"
        )

    # -----------------------------
    # Validate group_by
    # -----------------------------

    group_by = plan.get("group_by", [])

    if group_by is None:
        group_by = []

    if isinstance(group_by, str):
        group_by = [group_by]

    if not isinstance(group_by, list):
        raise ValueError("group_by must be a list.")

    if len(group_by) != len(set(group_by)):
        raise ValueError(
            "Duplicate grouping columns are not allowed."
        )

    for column in group_by:

        if column not in GROUP_COLUMNS:
            raise ValueError(
                f"Unsupported grouping column: {column}. "
                f"Available columns: {GROUP_COLUMNS}"
            )

    if operation == "overall" and group_by:
        raise ValueError(
            "Overall analysis requires group_by to be empty."
        )

    if operation == "group_analysis" and not group_by:
        raise ValueError(
            "At least one grouping column is required."
        )

    if operation == "compare" and len(group_by) != 1:
        raise ValueError(
            "Compare operation requires exactly one grouping column."
        )

    # -----------------------------
    # Validate aggregation
    # -----------------------------

    aggregation = plan.get("aggregation", "mean")

    if aggregation not in AGGREGATIONS:
        raise ValueError(
            f"Unsupported aggregation: {aggregation}"
        )

    # -----------------------------
    # Validate filter
    # -----------------------------

    filter_column = plan.get("filter_column")
    filter_value = plan.get("filter_value")

    if filter_column is not None:
        if filter_column not in FILTER_COLUMNS:
            raise ValueError(
                f"Unsupported filter column: {filter_column}"
            )

    if operation == "filter_analysis":

        if not filter_column:
            raise ValueError(
                "filter_column is required for filter_analysis."
            )

        if filter_value is None or str(filter_value).strip() == "":
            raise ValueError(
                "filter_value is required for filter_analysis."
            )

    # -----------------------------
    # Validate top_n
    # -----------------------------

    top_n = plan.get("top_n", 100)

    try:
        if top_n is None or isinstance(top_n, bool):
            top_n = 100
        else:
            top_n = int(top_n)

    except (ValueError, TypeError, OverflowError):
        top_n = 100

    if not 1 <= top_n <= 100:
        top_n = 100

    # -----------------------------
    # Validate sorting
    # -----------------------------

    sort_order = plan.get("sort", "descending")

    if sort_order not in ["ascending", "descending"]:
        sort_order = "descending"

    # Return normalized plan
    plan["operation"] = operation
    plan["group_by"] = group_by
    plan["aggregation"] = aggregation
    plan["filter_column"] = filter_column
    plan["filter_value"] = filter_value
    plan["top_n"] = top_n
    plan["sort"] = sort_order

    return plan


# ==========================================================
# 7. FILTER DATA
# ==========================================================

def filter_data(
    data: pd.DataFrame,
    column: str,
    value
) -> pd.DataFrame:

    if column not in data.columns:
        raise ValueError(
            f"Column not found: {column}"
        )

    if value is None or str(value).strip() == "":
        raise ValueError(
            "Filter value is missing."
        )

    search_value = str(value).strip()

    mask = (
        data[column]
        .astype("string")
        .str.contains(
            re.escape(search_value),
            case=False,
            na=False
        )
    )

    filtered_df = data[mask].copy()

    if filtered_df.empty:
        available_values = (
            data[column]
            .dropna()
            .astype(str)
            .unique()[:20]
            .tolist()
        )

        raise ValueError(
            f"No records found for '{value}' in '{column}'. "
            f"Some available values: {available_values}"
        )

    return filtered_df


# ==========================================================
# 8. AGGREGATE EMISSIONS
# ==========================================================

def aggregate_emissions(
    data: pd.DataFrame,
    aggregation: str
):

    if aggregation not in AGGREGATIONS:
        raise ValueError(
            f"Unsupported aggregation: {aggregation}"
        )

    if aggregation == "count":
        return data[CO2_COL].count()

    return getattr(data[CO2_COL], aggregation)()


# ==========================================================
# 9. GROUPED EMISSIONS
# ==========================================================

def grouped_emissions(
    data: pd.DataFrame,
    group_by: list,
    aggregation: str,
    sort_order: str = "descending",
    top_n: int = 100
) -> pd.DataFrame:

    for column in group_by:
        if column not in data.columns:
            raise ValueError(
                f"Grouping column '{column}' not found."
            )

    value_column = f"{aggregation.title()} CO2 Emissions"

    result = (
        data.groupby(
            group_by,
            dropna=False
        )[CO2_COL]
        .agg(aggregation)
        .reset_index(name=value_column)
    )

    result = result.sort_values(
        by=value_column,
        ascending=(sort_order == "ascending")
    )

    result = result.head(top_n)

    return result.reset_index(drop=True)


# ==========================================================
# 10. EXECUTE ANALYSIS
# ==========================================================

def execute_analysis(plan: dict):

    operation = plan["operation"]
    aggregation = plan["aggregation"]
    sort_order = plan["sort"]
    top_n = plan["top_n"]

    # -----------------------------
    # Overall analysis
    # -----------------------------

    if operation == "overall":

        return aggregate_emissions(
            df,
            aggregation
        )

    # -----------------------------
    # Group analysis
    # -----------------------------

    if operation in ["group_analysis", "compare"]:

        return grouped_emissions(
            data=df,
            group_by=plan["group_by"],
            aggregation=aggregation,
            sort_order=sort_order,
            top_n=top_n
        )

    # -----------------------------
    # Filter analysis
    # -----------------------------

    if operation == "filter_analysis":

        filtered_df = filter_data(
            data=df,
            column=plan["filter_column"],
            value=plan["filter_value"]
        )

        group_by = plan.get("group_by", [])

        # If user wants a filtered breakdown
        if group_by:

            return grouped_emissions(
                data=filtered_df,
                group_by=group_by,
                aggregation=aggregation,
                sort_order=sort_order,
                top_n=top_n
            )

        # If user wants one overall result for the filter
        return aggregate_emissions(
            filtered_df,
            aggregation
        )

    raise ValueError(
        f"Unsupported operation: {operation}"
    )


# ==========================================================
# 11. FORMAT RESULTS
# ==========================================================

def format_result(result) -> str:

    if isinstance(result, pd.DataFrame):

        if result.empty:
            return "No matching results found."

        return result.to_markdown(
            index=False,
            floatfmt=".2f"
        )

    if isinstance(result, pd.Series):
        return str(result)

    if pd.isna(result):
        return "No valid CO2 emission values found."

    if isinstance(result, (int, float)):
        return f"**Result: {result:,.2f} g/km**"

    return str(result)


# ==========================================================
# 12. MAIN DATA ANALYST AGENT
# ==========================================================

def data_analyst(question: str) -> str:

    try:

        if not question or not question.strip():
            return "Please enter a question."

        print("\nGenerating analysis plan...")

        plan = generate_analysis_plan(question)

        print("\nGenerated Analysis Plan:")
        print(json.dumps(plan, indent=4))

        plan = validate_plan(plan)

        print("\nValidated Analysis Plan:")
        print(json.dumps(plan, indent=4))

        print("\nExecuting Pandas analysis...")

        result = execute_analysis(plan)

        print("\nAnalysis completed.")

        return format_result(result)

    except Exception as exc:

        print("Analysis error:", str(exc))

        return f"Analysis error: {str(exc)}"


# ==========================================================
# 13. TEST THE AGENT DIRECTLY
# ==========================================================

if __name__ == "__main__":

    print("\nCO2 EMISSIONS DATA ANALYST")
    print("=========================")

    print("\nAvailable grouping columns:")
    print(GROUP_COLUMNS)

    while True:

        question = input(
            "\nAsk your data question (type exit to stop): "
        )

        if question.lower().strip() == "exit":
            print("Agent stopped.")
            break

        print("\nANALYSIS RESULT:")

        print(data_analyst(question))