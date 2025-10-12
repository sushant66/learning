# LangGraph

## Overview

LangGraph is a library for building stateful, multi-actor applications with LLMs. It extends LangChain with the ability to create cyclical graphs, which are essential for building agent runtimes. LangGraph makes it easy to build complex, stateful workflows with multiple actors that can loop, branch, and maintain state.

**Official Documentation**: https://langchain-ai.github.io/langgraph

## Why LangGraph?

Traditional chains in LangChain are acyclic (DAG - Directed Acyclic Graph). LangGraph enables:

- **Cycles and Loops**: Iterate until a condition is met
- **State Management**: Maintain and update state across steps
- **Branching Logic**: Conditional routing between nodes
- **Human-in-the-Loop**: Pause for human input/approval
- **Persistence**: Save and resume workflows
- **Multi-Agent Systems**: Coordinate multiple agents

## Installation

```bash
pip install langgraph
pip install langchain-openai  # For OpenAI models
```

## Core Concepts

### 1. Graph Structure

A LangGraph workflow consists of:

- **Nodes**: Functions that process state (e.g., call LLM, run tool)
- **Edges**: Connect nodes (normal, conditional, or entry points)
- **State**: Shared data structure passed between nodes
- **Checkpoints**: Save state at each step for persistence/replay

### 2. State

State is a typed object (TypedDict or Pydantic) that flows through the graph:

```python
from typing import TypedDict

class GraphState(TypedDict):
    messages: list
    user_input: str
    final_answer: str
    step_count: int
```

### 3. StateGraph

The main class for building graphs:

```python
from langgraph.graph import StateGraph, END

workflow = StateGraph(GraphState)
workflow.add_node("node_name", node_function)
workflow.add_edge("node_name", "next_node")
workflow.set_entry_point("node_name")
app = workflow.compile()
```

## Basic Example

### Simple Sequential Graph

```python
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

# Define state
class State(TypedDict):
    input: str
    output: str

# Define nodes
def process_input(state: State) -> State:
    """First node: process input"""
    processed = state["input"].upper()
    return {"input": state["input"], "output": processed}

def generate_response(state: State) -> State:
    """Second node: generate response"""
    llm = ChatOpenAI()
    response = llm.invoke(f"Respond to: {state['output']}")
    return {"output": response.content}

# Build graph
workflow = StateGraph(State)
workflow.add_node("process", process_input)
workflow.add_node("generate", generate_response)

workflow.set_entry_point("process")
workflow.add_edge("process", "generate")
workflow.add_edge("generate", END)

# Compile and run
app = workflow.compile()
result = app.invoke({"input": "hello"})
print(result)
```

## Key Features

### 1. Conditional Edges

Route to different nodes based on state:

```python
from langgraph.graph import END

class State(TypedDict):
    question: str
    answer: str
    needs_search: bool

def analyze_question(state: State) -> State:
    """Determine if search is needed"""
    llm = ChatOpenAI()
    prompt = f"Does this question need web search? Answer yes/no: {state['question']}"
    response = llm.invoke(prompt)

    needs_search = "yes" in response.content.lower()
    return {"needs_search": needs_search}

def search_web(state: State) -> State:
    """Search the web"""
    # Search logic here
    return {"answer": "Search results..."}

def answer_directly(state: State) -> State:
    """Answer without search"""
    llm = ChatOpenAI()
    response = llm.invoke(state["question"])
    return {"answer": response.content}

# Conditional routing function
def should_search(state: State) -> str:
    if state.get("needs_search"):
        return "search"
    return "answer"

# Build graph
workflow = StateGraph(State)
workflow.add_node("analyze", analyze_question)
workflow.add_node("search", search_web)
workflow.add_node("answer", answer_directly)

workflow.set_entry_point("analyze")

# Add conditional edge
workflow.add_conditional_edges(
    "analyze",
    should_search,
    {
        "search": "search",
        "answer": "answer"
    }
)

workflow.add_edge("search", END)
workflow.add_edge("answer", END)

app = workflow.compile()
```

### 2. Loops and Cycles

Create iterative workflows:

```python
from typing import TypedDict, List

class State(TypedDict):
    task: str
    iterations: int
    max_iterations: int
    result: str

def process_step(state: State) -> State:
    """Process one iteration"""
    llm = ChatOpenAI()
    response = llm.invoke(f"Improve this: {state.get('result', state['task'])}")

    return {
        "result": response.content,
        "iterations": state["iterations"] + 1
    }

def should_continue(state: State) -> str:
    """Decide whether to continue looping"""
    if state["iterations"] >= state["max_iterations"]:
        return "end"
    return "continue"

# Build graph with loop
workflow = StateGraph(State)
workflow.add_node("process", process_step)

workflow.set_entry_point("process")

# Add conditional edge that loops back
workflow.add_conditional_edges(
    "process",
    should_continue,
    {
        "continue": "process",  # Loop back to process
        "end": END
    }
)

app = workflow.compile()

result = app.invoke({
    "task": "Write a short poem",
    "iterations": 0,
    "max_iterations": 3
})
```

### 3. Agent with Tools

Build an agent that uses tools iteratively:

```python
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langgraph.graph import StateGraph, END
import operator

# Define tools
@tool
def search(query: str) -> str:
    """Search for information"""
    return f"Search results for: {query}"

@tool
def calculator(expression: str) -> str:
    """Calculate mathematical expressions"""
    try:
        return str(eval(expression))
    except:
        return "Invalid expression"

tools = [search, calculator]
tool_executor = ToolExecutor(tools)

# Define state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

# Define nodes
def call_model(state: AgentState) -> AgentState:
    """Call the LLM"""
    llm = ChatOpenAI(temperature=0).bind_tools(tools)
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def call_tools(state: AgentState) -> AgentState:
    """Execute tools"""
    messages = state["messages"]
    last_message = messages[-1]

    outputs = []
    for tool_call in last_message.tool_calls:
        tool_result = tool_executor.invoke(
            ToolInvocation(
                tool=tool_call["name"],
                tool_input=tool_call["args"]
            )
        )
        outputs.append({
            "tool_call_id": tool_call["id"],
            "output": tool_result
        })

    return {"messages": outputs}

# Routing function
def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return "end"
    return "continue"

# Build graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", call_tools)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)

workflow.add_edge("tools", "agent")

app = workflow.compile()

# Run agent
result = app.invoke({
    "messages": [HumanMessage(content="What is 25 * 4 + 10?")]
})

for message in result["messages"]:
    print(message)
```

### 4. Human-in-the-Loop

Pause execution for human approval:

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END

class State(TypedDict):
    request: str
    draft: str
    approved: bool

def create_draft(state: State) -> State:
    """Create a draft"""
    llm = ChatOpenAI()
    draft = llm.invoke(f"Create draft for: {state['request']}")
    return {"draft": draft.content}

def finalize(state: State) -> State:
    """Finalize after approval"""
    return {"draft": state["draft"] + "\n[APPROVED]"}

def should_finalize(state: State) -> str:
    if state.get("approved"):
        return "finalize"
    return "wait"

# Build graph with checkpointing
workflow = StateGraph(State)
workflow.add_node("draft", create_draft)
workflow.add_node("finalize", finalize)

workflow.set_entry_point("draft")

workflow.add_conditional_edges(
    "draft",
    should_finalize,
    {
        "finalize": "finalize",
        "wait": END  # Pause here
    }
)

workflow.add_edge("finalize", END)

# Compile with checkpointer for persistence
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)

# Run and pause
thread = {"configurable": {"thread_id": "1"}}
result = app.invoke({"request": "Write a blog post"}, thread)

# Later, approve and continue
result = app.invoke({"approved": True}, thread)
```

### 5. Persistence and Checkpointing

Save and resume workflows:

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# Use SQLite for persistence
checkpointer = SqliteSaver.from_conn_string("checkpoints.db")

app = workflow.compile(checkpointer=checkpointer)

# Run with thread ID
config = {"configurable": {"thread_id": "user-123"}}
result = app.invoke({"input": "hello"}, config)

# Resume from checkpoint
result = app.invoke({"input": "continue"}, config)

# Get state history
history = app.get_state_history(config)
for state in history:
    print(state)
```

### 6. Streaming

Stream intermediate results:

```python
# Stream events
for event in app.stream({"input": "hello"}):
    print(event)

# Stream with updates
for state in app.stream({"input": "hello"}, stream_mode="updates"):
    print(state)

# Stream values (full state at each step)
for state in app.stream({"input": "hello"}, stream_mode="values"):
    print(state)
```

## Multi-Agent Systems

### Supervisor Pattern

One agent coordinates multiple specialized agents:

```python
from typing import TypedDict, Literal

class State(TypedDict):
    messages: list
    next: str

def researcher(state: State) -> State:
    """Research agent"""
    llm = ChatOpenAI()
    response = llm.invoke("Research: " + state["messages"][-1])
    return {"messages": [response.content]}

def writer(state: State) -> State:
    """Writing agent"""
    llm = ChatOpenAI()
    response = llm.invoke("Write based on: " + state["messages"][-1])
    return {"messages": [response.content]}

def supervisor(state: State) -> State:
    """Supervisor decides next agent"""
    llm = ChatOpenAI()
    prompt = f"""Given the conversation, who should act next?
    Options: researcher, writer, finish

    Conversation: {state['messages']}
    """
    response = llm.invoke(prompt)

    next_agent = "finish"
    if "researcher" in response.content.lower():
        next_agent = "researcher"
    elif "writer" in response.content.lower():
        next_agent = "writer"

    return {"next": next_agent}

def route(state: State) -> str:
    return state["next"]

# Build graph
workflow = StateGraph(State)
workflow.add_node("supervisor", supervisor)
workflow.add_node("researcher", researcher)
workflow.add_node("writer", writer)

workflow.set_entry_point("supervisor")

workflow.add_conditional_edges(
    "supervisor",
    route,
    {
        "researcher": "researcher",
        "writer": "writer",
        "finish": END
    }
)

workflow.add_edge("researcher", "supervisor")
workflow.add_edge("writer", "supervisor")

app = workflow.compile()
```

### Handoff Pattern

Agents hand off to each other:

```python
class State(TypedDict):
    task: str
    result: str
    handoff_to: str

def agent_a(state: State) -> State:
    """First agent"""
    # Process and decide to hand off
    return {
        "result": "Agent A processed",
        "handoff_to": "agent_b"
    }

def agent_b(state: State) -> State:
    """Second agent"""
    return {
        "result": state["result"] + " -> Agent B processed",
        "handoff_to": "done"
    }

def router(state: State) -> str:
    return state.get("handoff_to", "done")

workflow = StateGraph(State)
workflow.add_node("a", agent_a)
workflow.add_node("b", agent_b)

workflow.set_entry_point("a")

workflow.add_conditional_edges(
    "a",
    router,
    {"agent_b": "b", "done": END}
)

workflow.add_conditional_edges(
    "b",
    router,
    {"done": END}
)

app = workflow.compile()
```

## Advanced Patterns

### 1. Parallel Execution

Run multiple nodes in parallel:

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List

class State(TypedDict):
    input: str
    result_a: str
    result_b: str
    aggregate: List[str]

def task_a(state: State) -> State:
    return {"result_a": "A done", "aggregate": state.get("aggregate", []) + ["A"]}

def task_b(state: State) -> State:
    return {"result_b": "B done", "aggregate": state.get("aggregate", []) + ["B"]}

def combine(state: State) -> State:
    return {"input": f"{state['result_a']} + {state['result_b']}"}

def route_to_parallel(state: State) -> list[str]:
    # Fan out to both tasks
    return ["a", "b"]

workflow = StateGraph(State)
workflow.add_node("a", task_a)
workflow.add_node("b", task_b)
workflow.add_node("combine", combine)

# Fan out from START to both "a" and "b" in parallel
workflow.add_conditional_edges(START, route_to_parallel, {"a": "a", "b": "b"})
workflow.add_edge("a", "combine")
workflow.add_edge("b", "combine")
workflow.add_edge("combine", END)

app = workflow.compile()
```

### 2. Map-Reduce Pattern

Process items in parallel and combine:

```python
from typing import List

class State(TypedDict):
    items: List[str]
    results: List[str]

def map_node(state: State) -> State:
    """Process each item"""
    results = []
    for item in state["items"]:
        # Process item
        results.append(f"Processed: {item}")
    return {"results": results}

def reduce_node(state: State) -> State:
    """Combine results"""
    combined = " | ".join(state["results"])
    return {"results": [combined]}

workflow = StateGraph(State)
workflow.add_node("map", map_node)
workflow.add_node("reduce", reduce_node)

workflow.set_entry_point("map")
workflow.add_edge("map", "reduce")
workflow.add_edge("reduce", END)

app = workflow.compile()
```

### 3. Subgraphs

Compose graphs within graphs:

```python
# Create a subgraph
def create_subgraph():
    subworkflow = StateGraph(State)
    subworkflow.add_node("sub_a", node_a)
    subworkflow.add_node("sub_b", node_b)
    subworkflow.set_entry_point("sub_a")
    subworkflow.add_edge("sub_a", "sub_b")
    subworkflow.add_edge("sub_b", END)
    return subworkflow.compile()

# Use in main graph
workflow = StateGraph(State)
workflow.add_node("main_a", main_node_a)
workflow.add_node("subgraph", create_subgraph())
workflow.add_node("main_b", main_node_b)

workflow.set_entry_point("main_a")
workflow.add_edge("main_a", "subgraph")
workflow.add_edge("subgraph", "main_b")
workflow.add_edge("main_b", END)

app = workflow.compile()
```

## Visualization

Visualize your graph structure:

```python
from IPython.display import Image, display

# Generate graph visualization
display(Image(app.get_graph().draw_mermaid_png()))

# Or get mermaid syntax
print(app.get_graph().draw_mermaid())
```

## Error Handling

```python
def node_with_error_handling(state: State) -> State:
    try:
        # Node logic
        result = risky_operation()
        return {"result": result}
    except Exception as e:
        return {"error": str(e), "result": "fallback"}

# Or handle at graph level
def should_continue(state: State) -> str:
    if "error" in state:
        return "error_handler"
    return "continue"
```

## Best Practices

1. **State Design**: Keep state minimal and well-typed
2. **Idempotent Nodes**: Nodes should be rerunnable with same input
3. **Use Checkpointing**: Enable persistence for long-running workflows
4. **Streaming**: Stream for better UX in interactive apps
5. **Visualization**: Visualize graphs during development
6. **Error Handling**: Handle errors gracefully in nodes
7. **Testing**: Test nodes independently before integration
8. **Conditional Logic**: Keep routing logic simple and clear

## Common Use Cases

### 1. Research Agent
```python
# Multi-step research with iteration
# search -> analyze -> refine -> search -> final_answer
```

### 2. Code Generation
```python
# generate -> test -> fix -> test -> finalize
```

### 3. Content Creation
```python
# plan -> write -> review -> revise -> publish
```

### 4. Customer Support
```python
# classify -> route -> respond -> escalate_if_needed
```

### 5. Data Pipeline
```python
# extract -> transform -> validate -> load
```

## Integration with LangSmith

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-key"

# All graph executions will be traced
app = workflow.compile()
result = app.invoke({"input": "data"})
```

## Comparison: LangChain vs LangGraph

| Feature | LangChain | LangGraph |
|---------|-----------|-----------|
| Structure | Linear/DAG | Cyclic graphs |
| Loops | No | Yes |
| State | Implicit | Explicit |
| Conditional routing | Limited | Full support |
| Persistence | No | Yes (checkpoints) |
| Human-in-loop | Difficult | Built-in |
| Multi-agent | Complex | Native |
| Use case | Simple chains | Complex workflows |

## Resources

- **Documentation**: https://langchain-ai.github.io/langgraph/
- **GitHub**: https://github.com/langchain-ai/langgraph
- **Examples**: https://github.com/langchain-ai/langgraph/tree/main/examples
- **Tutorials**: https://langchain-ai.github.io/langgraph/tutorials/
