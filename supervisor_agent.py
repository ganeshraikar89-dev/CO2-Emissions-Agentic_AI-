
import os

from dotenv import load_dotenv

from langchain_groq import ChatGroq

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

from langgraph.graph import (
    StateGraph,
    START,
    END,
    MessagesState
)

# Import existing agents
from agent import app as analytics_app

from Data_analyst_agent import data_analyst


# --------------------------------------------------
# 1. LOAD ENVIRONMENT VARIABLES
# --------------------------------------------------

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found")


# --------------------------------------------------
# 2. INITIALIZE SUPERVISOR LLM
# --------------------------------------------------

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=api_key
)


# --------------------------------------------------
# 3. DEFINE GRAPH STATE
# --------------------------------------------------

class SupervisorState(MessagesState):
    route: str


# --------------------------------------------------
# 4. SUPERVISOR NODE
# --------------------------------------------------


# --------------------------------------------------
# 4. SUPERVISOR NODE
# --------------------------------------------------

def supervisor(state: SupervisorState):

    question = state["messages"][-1].content

    system_prompt = """
You are a Supervisor Agent for a CO2 Emissions Analytics system.

You have two agents available.

AGENT 1: analytics_agent

Use this agent for predefined overall analytics:

- Total CO2 emissions across all vehicles
- Overall average CO2 emissions
- Overall maximum CO2 emissions
- Overall minimum CO2 emissions
- Emissions by Vehicle Class
- Emissions by Fuel Type
- Emissions by Cylinders
- Emissions by Transmission
- Emission by Fuel Type
- Top 10 highest-emission vehicles
- Highest emission by vehicle model

AGENT 2: data_analyst

Use this agent for manufacturer-level and flexible data analysis:

- Analysis by manufacturer or Make
- Comparing two or more manufacturers
- Average emissions for a particular manufacturer
- Total emissions for a particular manufacturer
- Highest average emissions by manufacturer
- Lowest average emissions by manufacturer
- Top N manufacturers by emissions
- Top N Emission by Transmission
- Analysis by Make and Vehicle Class
- Analysis by Make and Fuel Type
- Analysis by Make and Cylinders
- Comparing manufacturers across multiple categories
- Flexible filtering and grouping
- Analysis involving two or more dataset columns

ROUTING RULES:

1. If the question mentions a manufacturer, brand , Transmissionor Make,
   always select data_analyst.

2. If the question compares manufacturers,
   select data_analyst.

3. If the question asks for emissions grouped by two or more
   columns, select data_analyst.

4. If the question asks for flexible filtering, ranking,
   grouping or category comparisons, select data_analyst.

5. If the question asks for predefined overall statistics
   without manufacturer-specific analysis, select analytics_agent.

EXAMPLES:

Question: What is total CO2 emissions?
Route: analytics_agent

Question: What is overall average CO2 emissions?
Route: analytics_agent

Question: Average CO2 emissions by Fuel Type?
Route: analytics_agent

Question: Average CO2 emissions by Transmission?
Route: data_analyst

Question: What is average CO2 emission for Mercedes-Benz?
Route: data_analyst

Question: Compare BMW and Audi emissions.
Route: data_analyst

Question: Which manufacturer has the highest average emissions?
Route: data_analyst

Question: Show average emissions by Make and Vehicle Class.
Route: data_analyst

Question: Show total emissions by Make and Fuel Type.
Route: data_analyst

Question: Find the top 10 Make and Vehicle Class combinations.
Route: data_analyst

Return ONLY one exact value:

analytics_agent
data_analyst
"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=str(question))
    ])

    route_text = response.content

    if isinstance(route_text, list):
        route_text = "".join(
            item.get("text", "")
            if isinstance(item, dict)
            else str(item)
            for item in route_text
        )

    route_text = route_text.strip().lower()

    print("\nSUPERVISOR DECISION:")
    print(route_text)

    if "data_analyst" in route_text:
        route = "data_analyst"
    elif "analytics_agent" in route_text:
        route = "analytics_agent"
    else:
        raise ValueError(
            f"Supervisor returned an invalid route: {route_text}"
        )

    print("Selected agent:", route)

    return {"route": route}


# --------------------------------------------------
# 5. ANALYTICS AGENT NODE
# --------------------------------------------------

def run_analytics_agent(state: SupervisorState):

    print("\nROUTING TO ANALYTICS AGENT")

    result = analytics_app.invoke({
        "messages": state["messages"]
    })

    final_message = result["messages"][-1]

    return {
        "messages": [final_message]
    }


# --------------------------------------------------
# 6. DATA ANALYST AGENT NODE
# --------------------------------------------------

def run_data_analyst(state: SupervisorState):

    print("\nROUTING TO DATA ANALYST AGENT")

    question = state["messages"][-1].content

    answer = data_analyst(str(question))

    return {
        "messages": [
            AIMessage(content=str(answer))
        ]
    }


# --------------------------------------------------
# 7. ROUTING FUNCTION
# --------------------------------------------------

def route_question(state: SupervisorState):

    return state["route"]


# --------------------------------------------------
# 8. BUILD LANGGRAPH WORKFLOW
# --------------------------------------------------

workflow = StateGraph(SupervisorState)

# Add nodes
workflow.add_node("supervisor", supervisor)

workflow.add_node(
    "analytics_agent",
    run_analytics_agent
)

workflow.add_node(
    "data_analyst",
    run_data_analyst
)

# Starting point
workflow.add_edge(START, "supervisor")

# Conditional routing
workflow.add_conditional_edges(
    "supervisor",
    route_question,
    {
        "analytics_agent": "analytics_agent",
        "data_analyst": "data_analyst"
    }
)

# Both agents finish the workflow
workflow.add_edge("analytics_agent", END)

workflow.add_edge("data_analyst", END)

# Compile graph
supervisor_app = workflow.compile()


# --------------------------------------------------
# 9. TEST SUPERVISOR AGENT
# --------------------------------------------------

if __name__ == "__main__":

    print("\nCO2 SUPERVISOR AGENT")
    print("--------------------")

    while True:

        question = input(
            "\nAsk your question (exit to stop): "
        )

        if question.lower().strip() == "exit":
            print("Supervisor stopped.")
            break

        result = supervisor_app.invoke({
            "messages": [
                HumanMessage(content=question)
            ]
        })

        print("\nFINAL ANSWER:")

        print(
            result["messages"][-1].content
        )
