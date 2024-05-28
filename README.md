# Building an Intelligent Agent with LangGraph

Welcome to the setup guide for creating an intelligent agent using LangChain and LangGraph. This README will walk you through setting up the project, explain the key components, and guide you in understanding the interactions within the code.

## Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Project Structure](#project-structure)
5. [Setting Up Environment Variables](#setting-up-environment-variables)
6. [Code Overview](#code-overview)
    - [Defining the Agent's State](#defining-the-agents-state)
    - [Running the Agent and Executing Tools](#running-the-agent-and-executing-tools)
    - [Conditional Logic for Workflow](#conditional-logic-for-workflow)
    - [Building the Workflow with LangGraph](#building-the-workflow-with-langgraph)
    - [Running the Application](#running-the-application)
7. [Conclusion](#conclusion)

## Introduction

In this project, we'll build an intelligent agent capable of interacting with language models, executing functions, and managing complex tasks using LangChain and LangGraph. We'll cover the setup, code structure, and how everything works together.

## Prerequisites

Ensure you have the following installed:
- Python 3.7 or higher
- Git (for cloning the repository)

## Installation

1. **Clone the Repository**

   ```sh
   git clone https://github.com/yourusername/yourproject.git
   cd yourproject
   ```

2. **Install Dependencies**

   Install the required Python packages using pip:

   ```sh
   pip install -r requirements.txt
   ```

## Project Structure

Here's the structure of the project:

```
yourproject/
│
├── main.py
├── .env
├── requirements.txt
└── README.md
```

- `main.py`: The main script containing the agent's code.
- `.env`: File to store environment variables.
- `requirements.txt`: List of required Python packages.
- `README.md`: This setup guide.

## Setting Up Environment Variables

Create a `.env` file in the root directory of your project and add your API keys:

```
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
```

## Code Overview

### Defining the Agent's State

We define a custom data structure to hold the agent's state, including the input, chat history, outcomes, and intermediate steps.

```python
from typing import TypedDict, Annotated, List, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    input: str  # The input string
    chat_history: list[BaseMessage]  # The list of previous messages in the conversation
    agent_outcome: Union[AgentAction, AgentFinish, None]  # The outcome of a given call to the agent
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]  # List of actions and corresponding observations
```

### Running the Agent and Executing Tools

We define functions to run the agent and execute tools. These will be used as nodes in our workflow.

```python
from langchain.agents import create_openai_functions_agent
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

# Define the agent
tools = [TavilySearchResults(max_results=1)]
llm = ChatOpenAI(model="gpt-3.5-turbo-1106", streaming=True)
prompt = hub.pull("hwchase17/openai-functions-agent")
agent_runnable = create_openai_functions_agent(llm, tools, prompt)

# Function to run the agent
def run_agent(data):
    agent_outcome = agent_runnable.invoke(data)
    return {"agent_outcome": agent_outcome}

# Function to execute tools
from langgraph.prebuilt.tool_executor import ToolExecutor

tool_executor = ToolExecutor(tools)

def execute_tools(data):
    agent_action = data["agent_outcome"]
    output = tool_executor.invoke(agent_action)
    return {"intermediate_steps": [(agent_action, str(output))]}
```

### Conditional Logic for Workflow

We need a function to determine whether to continue the workflow or end it based on the agent's outcomes.

```python
def should_continue(data):
    if isinstance(data["agent_outcome"], AgentFinish):
        return "end"
    else:
        return "continue"
```

### Building the Workflow with LangGraph

We define our workflow using LangGraph, creating a state graph where nodes represent different steps.

```python
from langgraph.graph import END, StateGraph

# Define a new graph
workflow = StateGraph(AgentState)

# Add nodes to the workflow
workflow.add_node("agent", run_agent)
workflow.add_node("action", execute_tools)

# Set the entry point
workflow.set_entry_point("agent")

# Add conditional edges
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END,
    },
)

# Add a normal edge
workflow.add_edge("action", "agent")

# Compile the workflow into a LangChain Runnable
app = workflow.compile()
```

### Running the Application

Finally, we run the application and stream the outputs.

```python
inputs = {"input": "what is the weather in sf", "chat_history": []}
for s in app.stream(inputs):
    print(list(s.values())[0])
    print("----")
```

## Conclusion

In this guide, we've walked through setting up and building an intelligent agent using LangChain and LangGraph. We've defined the agent's state, created functions to run the agent and execute tools, set up conditional logic, and built a workflow. This structured approach allows you to manage complex tasks and build sophisticated agents efficiently.

By understanding the interactions between these components, you can extend and customize this setup for various applications, leveraging the power of LangChain and LangGraph to create intelligent, automated workflows.
