from dotenv import load_dotenv
import os


groq_api_key = os.getenv("GROQ_API_KEY")


import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

from analytics import (
    total_emissions,
    average_emissions,
    maximum_emissions,
    minimum_emissions,
    emissions_by_vehicle_class,
    emissions_by_fuel_type,
    emissions_by_cylinders,
    top_10_emission_vehicles
)

# Load environment variables
load_dotenv()

# Initialize Groq LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)


# -------------------------------
# DEFINE ANALYTICS TOOLS
# -------------------------------

@tool
def get_total_emissions():
    """Calculate total CO2 emissions across all vehicles."""
    return str(total_emissions())


@tool
def get_average_emissions():
    """Calculate average CO2 emissions across all vehicles."""
    return str(average_emissions())


@tool
def get_maximum_emissions():
    """Find the maximum CO2 emission value."""
    return str(maximum_emissions())


@tool
def get_minimum_emissions():
    """Find the minimum CO2 emission value."""
    return str(minimum_emissions())


@tool
def get_emissions_by_vehicle_class():
    """Calculate average CO2 emissions by vehicle class."""
    return emissions_by_vehicle_class().to_string()


@tool
def get_emissions_by_fuel_type():
    """Calculate average CO2 emissions by fuel type."""
    return emissions_by_fuel_type().to_string()


@tool
def get_emissions_by_cylinders():
    """Calculate average CO2 emissions by number of cylinders."""
    return emissions_by_cylinders().to_string()


@tool
def get_top_10_vehicles():
    """Return the 10 vehicles with the highest CO2 emissions."""
    return top_10_emission_vehicles().to_string(index=False)


# Register tools
tools = [
    get_total_emissions,
    get_average_emissions,
    get_maximum_emissions,
    get_minimum_emissions,
    get_emissions_by_vehicle_class,
    get_emissions_by_fuel_type,
    get_emissions_by_cylinders,
    get_top_10_vehicles
]

# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools)


# -------------------------------
# DEFINE AGENT NODE
# -------------------------------

def agent(state: MessagesState):

    system_prompt = """
    You are a CO2 Emissions Analytics Assistant.

    Answer questions using the available analytical tools.
    Select the appropriate tool based on the user's question.
    Explain results clearly and concisely.
    Emission values are measured in g/km.
    Never invent numerical results.
    """

    response = llm_with_tools.invoke(
        [SystemMessage(content=system_prompt)]
        + state["messages"]
    )

    return {"messages": [response]}


# -------------------------------
# BUILD LANGGRAPH WORKFLOW
# -------------------------------

workflow = StateGraph(MessagesState)

workflow.add_node("agent", agent)
workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "agent")

workflow.add_conditional_edges(
    "agent",
    tools_condition
)

workflow.add_edge("tools", "agent")

app = workflow.compile()


# -------------------------------
# TEST AGENT
# -------------------------------

if __name__ == "__main__":

    print("CO2 EMISSIONS ANALYTICS AGENT")
    print("--------------------------------")

    while True:

        question = input("\nAsk your question (type exit to stop): ")

        if question.lower() == "exit":
            break

        result = app.invoke({
            "messages": [HumanMessage(content=question)]
        })

        print("\nAI RESPONSE:")
        print(result["messages"][-1].content)