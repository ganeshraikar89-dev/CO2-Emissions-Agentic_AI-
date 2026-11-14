import streamlit as st
import pandas as pd
import plotly.express as px
from langchain_core.messages import HumanMessage

from Data_Loader import load_data
import streamlit as st
import os

# Load Groq API key from Streamlit Cloud Secrets
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# Import agents only after loading API key
from supervisor_agent import supervisor_app

from supervisor_agent import supervisor_app
from Create_prediction import render_prediction_page


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="CO₂ Emissions Intelligence",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 2. CUSTOM DASHBOARD STYLING
# =========================================================
st.markdown("""
<style>
    .stApp {
        background: #F4F7FB;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.4rem;
        padding-bottom: 2.5rem;
    }

    .main-header {
        background: linear-gradient(115deg, #102A43 0%, #174E78 60%, #087E8B 100%);
        padding: 26px 30px;
        border-radius: 16px;
        margin-bottom: 22px;
        color: white;
    }

    .main-header h1 {
        color: white;
        font-size: 29px;
        margin: 0 0 5px 0;
        font-weight: 750;
    }

    .main-header p {
        color: #D9EAF4;
        font-size: 14px;
        margin: 0;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #E3EAF2;
        border-radius: 13px;
        padding: 17px 19px;
        box-shadow: 0 3px 12px rgba(16, 42, 67, 0.045);
    }

    div[data-testid="stMetricLabel"] {
        color: #61758A;
        font-size: 13px;
        font-weight: 600;
    }

    div[data-testid="stMetricValue"] {
        color: #123B5D;
        font-weight: 750;
        font-size: 27px;
    }

    div[data-testid="stPlotlyChart"],
    div[data-testid="stDataFrame"] {
        background: white;
        border: 1px solid #E3EAF2;
        border-radius: 12px;
        padding: 8px;
    }

    section[data-testid="stSidebar"] {
        background: #EAF0F6;
    }

    .section-caption {
        color: #718096;
        font-size: 13px;
        margin-top: -10px;
        margin-bottom: 12px;
    }

    div.stButton > button {
        border-radius: 9px;
        font-weight: 600;
    }

    .small-note {
        color: #718096;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. LOAD DATA
# =========================================================
@st.cache_data
def get_data():
    data = load_data()
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data_Loader.load_data() must return a pandas DataFrame.")
    return data


try:
    df = get_data()
except Exception as exc:
    st.error(f"Unable to load the emissions dataset: {exc}")
    st.stop()


CO2_COL = "CO2 Emissions(g/km)"

if CO2_COL not in df.columns:
    st.error(
        f"Required column '{CO2_COL}' was not found. "
        f"Available columns: {list(df.columns)}"
    )
    st.stop()

df = df.copy()
df[CO2_COL] = pd.to_numeric(df[CO2_COL], errors="coerce")
df = df.dropna(subset=[CO2_COL])

if df.empty:
    st.error("The dataset does not contain any valid CO₂ emission values.")
    st.stop()


# =========================================================
# 4. SIDEBAR FILTERS
# =========================================================
st.sidebar.title("🌱 Dashboard Controls")
st.sidebar.caption("Filter the vehicle dataset")

filtered_df = df.copy()

filter_columns = ["Make", "Vehicle Class", "Fuel Type", "Transmission", "Cylinders"]

for column in filter_columns:
    if column in filtered_df.columns:
        values = sorted(filtered_df[column].dropna().astype(str).unique().tolist())
        selected = st.sidebar.multiselect(
            column,
            options=values,
            default=[],
            key=f"filter_{column}"
        )
        if selected:
            filtered_df = filtered_df[
                filtered_df[column].astype(str).isin(selected)
            ]

st.sidebar.divider()
st.sidebar.metric("Filtered records", f"{len(filtered_df):,}")
st.sidebar.caption(f"Total dataset records: {len(df):,}")

if filtered_df.empty:
    st.warning("No records match the selected filters. Change or clear the sidebar filters.")
    st.stop()


# =========================================================
# 5. HEADER
# =========================================================
st.markdown("""
<div class="main-header">
    <h1>🌱 CO₂ Emissions Intelligence Dashboard</h1>
    <p>Automotive emissions monitoring · Interactive analytics · AI-powered insights</p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# 6. KPI CARDS
# =========================================================
total_co2 = filtered_df[CO2_COL].sum()
average_co2 = filtered_df[CO2_COL].mean()
maximum_co2 = filtered_df[CO2_COL].max()
minimum_co2 = filtered_df[CO2_COL].min()

k1, k2, k3, k4 = st.columns(4)

k1.metric("Total CO₂ emissions", f"{total_co2:,.0f} g/km")
k2.metric("Average emissions", f"{average_co2:,.2f} g/km")
k3.metric("Highest emissions", f"{maximum_co2:,.0f} g/km")
k4.metric("Lowest emissions", f"{minimum_co2:,.0f} g/km")

st.caption(
    f"Showing {len(filtered_df):,} of {len(df):,} records based on current filters."
)


# =========================================================
# 7. CHART HELPERS
# =========================================================
PLOT_TEMPLATE = "plotly_white"
BLUE = "#1769AA"
TEAL = "#0E9F9A"


def average_by(data, column):
    """Return average CO₂ emissions by a dataset column."""
    if column not in data.columns:
        return pd.DataFrame()

    result = (
        data.groupby(column, dropna=False)[CO2_COL]
        .mean()
        .reset_index()
        .sort_values(CO2_COL, ascending=False)
    )
    result[column] = result[column].fillna("Unknown").astype(str)
    return result


def style_figure(fig, height=390):
    fig.update_layout(
        template=PLOT_TEMPLATE,
        height=height,
        margin=dict(l=15, r=15, t=55, b=25),
        title_font=dict(size=16, color="#173B57"),
        font=dict(family="Arial", size=12, color="#425466"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend_title_text=""
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#E9EFF5", zeroline=False)
    return fig


# =========================================================
# 8. DASHBOARD TABS
# =========================================================
overview_tab, detail_tab, data_tab, ai_tab, prediction_tab = st.tabs([
    "📊 Overview",
    "📈 Detailed Analysis",
    "🚗 Data Explorer",
    "🤖 AI Analyst",
    "🔮 CO₂ Prediction"
])


# =========================================================
# TAB 1 - OVERVIEW
# =========================================================
with overview_tab:
    st.subheader("Emissions overview")
    st.markdown(
        '<div class="section-caption">Explore average CO₂ emissions across vehicle categories.</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns([1.35, 1])

    with left:
        if "Vehicle Class" in filtered_df.columns:
            vehicle_data = average_by(filtered_df, "Vehicle Class")
            if not vehicle_data.empty:
                fig_vehicle = px.bar(
                    vehicle_data,
                    x="Vehicle Class",
                    y=CO2_COL,
                    color=CO2_COL,
                    color_continuous_scale=["#B8DDF2", "#368AC5", "#123F70"],
                    title="Average CO₂ emissions by vehicle class",
                    labels={CO2_COL: "Average CO₂ (g/km)"}
                )
                fig_vehicle.update_layout(coloraxis_showscale=False)
                fig_vehicle.update_xaxes(tickangle=-30)
                st.plotly_chart(style_figure(fig_vehicle, 430), use_container_width=True)
            else:
                st.info("No Vehicle Class data is available.")

    with right:
        if "Fuel Type" in filtered_df.columns:
            fuel_data = (
                filtered_df.groupby("Fuel Type", dropna=False)
                .size()
                .reset_index(name="Vehicle count")
            )
            fuel_data["Fuel Type"] = fuel_data["Fuel Type"].fillna("Unknown").astype(str)
            fig_fuel = px.pie(
                fuel_data,
                names="Fuel Type",
                values="Vehicle count",
                hole=0.58,
                title="Vehicle distribution by fuel type",
                color_discrete_sequence=["#1769AA", "#0E9F9A", "#F4A261", "#8B7EC8", "#E76F51"]
            )
            fig_fuel.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(style_figure(fig_fuel, 430), use_container_width=True)
        else:
            st.info("No Fuel Type column is available.")

    left2, right2 = st.columns(2)

    with left2:
        if "Cylinders" in filtered_df.columns:
            cylinder_data = average_by(filtered_df, "Cylinders")
            if not cylinder_data.empty:
                cylinder_data["Cylinders"] = cylinder_data["Cylinders"].astype(str)
                fig_cylinder = px.bar(
                    cylinder_data,
                    x="Cylinders",
                    y=CO2_COL,
                    color_discrete_sequence=[TEAL],
                    title="Average CO₂ emissions by cylinders",
                    labels={CO2_COL: "Average CO₂ (g/km)"}
                )
                st.plotly_chart(style_figure(fig_cylinder, 350), use_container_width=True)

    with right2:
        if "Transmission" in filtered_df.columns:
            transmission_data = average_by(filtered_df, "Transmission")
            if not transmission_data.empty:
                fig_transmission = px.bar(
                    transmission_data,
                    x="Transmission",
                    y=CO2_COL,
                    color_discrete_sequence=[BLUE],
                    title="Average CO₂ emissions by transmission",
                    labels={CO2_COL: "Average CO₂ (g/km)"}
                )
                fig_transmission.update_xaxes(tickangle=-20)
                st.plotly_chart(style_figure(fig_transmission, 350), use_container_width=True)


# =========================================================
# TAB 2 - DETAILED ANALYSIS
# =========================================================
with detail_tab:
    st.subheader("Detailed emissions analysis")

    available_dimensions = [
        c for c in ["Make", "Model", "Vehicle Class", "Fuel Type", "Transmission", "Cylinders"]
        if c in filtered_df.columns
    ]

    if available_dimensions:
        selected_dimension = st.selectbox(
            "Choose a dimension",
            available_dimensions,
            key="detail_dimension"
        )

        metric_choice = st.selectbox(
            "Choose calculation",
            ["Average", "Total", "Maximum", "Minimum", "Vehicle count"],
            key="detail_metric"
        )

        top_n = st.slider(
            "Number of categories to display",
            min_value=5,
            max_value=30,
            value=10,
            key="detail_top_n"
        )

        if metric_choice == "Vehicle count":
            analysis = (
                filtered_df.groupby(selected_dimension, dropna=False)
                .size()
                .reset_index(name="Vehicle count")
            )
            value_col = "Vehicle count"
            y_label = "Number of vehicles"
        else:
            agg_map = {
                "Average": "mean",
                "Total": "sum",
                "Maximum": "max",
                "Minimum": "min"
            }
            analysis = (
                filtered_df.groupby(selected_dimension, dropna=False)[CO2_COL]
                .agg(agg_map[metric_choice])
                .reset_index()
            )
            value_col = CO2_COL
            y_label = f"{metric_choice} CO₂ emissions (g/km)"

        analysis[selected_dimension] = analysis[selected_dimension].fillna("Unknown").astype(str)
        analysis = analysis.sort_values(value_col, ascending=False).head(top_n)

        fig_detail = px.bar(
            analysis,
            x=selected_dimension,
            y=value_col,
            color=value_col,
            color_continuous_scale=["#B8DDF2", "#368AC5", "#123F70"],
            title=f"{metric_choice} by {selected_dimension}",
            labels={value_col: y_label}
        )
        fig_detail.update_layout(coloraxis_showscale=False)
        fig_detail.update_xaxes(tickangle=-30)
        st.plotly_chart(style_figure(fig_detail, 450), use_container_width=True)

        st.dataframe(analysis, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("#### Compare two manufacturers")

    if "Make" in filtered_df.columns:
        makes = sorted(filtered_df["Make"].dropna().astype(str).unique().tolist())
        if len(makes) >= 2:
            c1, c2 = st.columns(2)
            with c1:
                make_a = st.selectbox("Manufacturer A", makes, index=0)
            with c2:
                make_b = st.selectbox("Manufacturer B", makes, index=1)

            comparison = filtered_df[
                filtered_df["Make"].astype(str).isin([make_a, make_b])
            ]

            comparison_data = (
                comparison.groupby("Make")[CO2_COL]
                .mean()
                .reset_index()
                .rename(columns={CO2_COL: "Average CO₂ (g/km)"})
            )

            fig_compare = px.bar(
                comparison_data,
                x="Make",
                y="Average CO₂ (g/km)",
                color="Make",
                title="Average emissions comparison",
                text_auto=".2f"
            )
            st.plotly_chart(style_figure(fig_compare, 330), use_container_width=True)
        else:
            st.info("At least two manufacturers are required for comparison.")


# =========================================================
# TAB 3 - DATA EXPLORER
# =========================================================
with data_tab:
    st.subheader("Vehicle emissions data explorer")
    st.caption("Search, inspect and export the currently filtered dataset.")

    search_text = st.text_input(
        "Search across available columns",
        placeholder="Enter a make, model, fuel type, transmission..."
    )

    display_df = filtered_df.copy()

    if search_text.strip():
        search_mask = display_df.astype(str).apply(
            lambda col: col.str.contains(search_text, case=False, na=False)
        ).any(axis=1)
        display_df = display_df[search_mask]

    st.write(f"**{len(display_df):,} records found**")

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=480
    )

    csv_data = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download filtered data as CSV",
        data=csv_data,
        file_name="filtered_co2_emissions.csv",
        mime="text/csv"
    )


# =========================================================
# TAB 4 - AI ANALYST
# =========================================================
with ai_tab:
    st.subheader("🤖 CO₂ Emissions AI Analyst")
    st.caption(
        "Ask questions in natural language. The LangGraph supervisor "
        "will route the request to the appropriate analytics agent."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    suggested_questions = [
        "What is the average CO₂ emissions by Transmission?",
        "Compare average CO₂ emissions by Make and Fuel Type.",
        "Which manufacturer has the highest average CO₂ emissions?",
        "Show the top 10 vehicle classes by average CO₂ emissions."
    ]

    with st.expander("Example questions"):
        for question in suggested_questions:
            if st.button(question, key=f"example_{question}"):
                st.session_state.pending_question = question
                st.rerun()

    user_question = st.chat_input(
        "Ask a question about your vehicle emissions data...",
        key="co2_chat_input"
    )

    if "pending_question" in st.session_state:
        user_question = st.session_state.pop("pending_question")

    if user_question:
        st.session_state.messages.append({
            "role": "user",
            "content": user_question
        })

        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Supervisor is analysing your question..."):
                try:
                    response = supervisor_app.invoke({
                        "messages": [HumanMessage(content=user_question)]
                    })

                    messages = response.get("messages", [])
                    if not messages:
                        raise ValueError(
                            "Supervisor returned no messages. "
                            "Check the final node in supervisor_agent.py."
                        )

                    final_message = messages[-1]
                    if hasattr(final_message, "content"):
                        answer = final_message.content
                    else:
                        answer = str(final_message)

                    if isinstance(answer, list):
                        answer = "\n".join(
                            item.get("text", str(item))
                            if isinstance(item, dict) else str(item)
                            for item in answer
                        )

                    st.markdown(answer)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": str(answer)
                    })

                except Exception as exc:
                    error_message = f"Supervisor Agent error: {exc}"
                    st.error(error_message)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_message
                    })

    if st.button("🗑️ Clear AI chat history"):
        st.session_state.messages = []
        st.rerun()


# =========================================================
# TAB 5 - MACHINE LEARNING PREDICTION
# =========================================================
with prediction_tab:
    st.subheader("🔮 CO₂ Emissions Prediction")
    st.caption(
        "Estimate vehicle CO₂ emissions using the trained machine learning model. "
        "This tab uses the full dataset for its input choices, independently of "
        "the analytics sidebar filters."
    )

    try:
        # Use the full dataset, not filtered_df, so sidebar filters do not
        # unintentionally restrict prediction input options.
        render_prediction_page(df)
    except FileNotFoundError:
        st.error(
            "The trained model was not found. Run `python train_model.py` "
            "to train and save the prediction model first."
        )
    except Exception as exc:
        st.error(f"Prediction module error: {exc}")


# =========================================================
# 9. FOOTER
# =========================================================
st.divider()
st.markdown(
    '<p class="small-note">CO₂ Emissions Intelligence Dashboard · '
    'Powered by Streamlit, Pandas, Plotly and LangGraph</p>',
    unsafe_allow_html=True
)
