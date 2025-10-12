# LangSmith

## Overview

LangSmith is a platform for debugging, testing, evaluating, and monitoring LLM applications. It provides observability and evaluation tools for the entire LLM application lifecycle, from development to production.

**Official Website**: https://smith.langchain.com

## Why LangSmith?

Building production LLM applications requires:

- **Debugging**: Understand what's happening in complex chains
- **Testing**: Ensure consistent behavior across changes
- **Evaluation**: Measure quality and performance
- **Monitoring**: Track production usage and issues
- **Optimization**: Identify bottlenecks and improve performance

LangSmith solves these problems with a comprehensive platform.

## Key Features

1. **Tracing**: Detailed execution traces of LLM calls and chains
2. **Datasets**: Create and manage test datasets
3. **Evaluation**: Run automated evaluations with custom metrics
4. **Monitoring**: Production monitoring and analytics
5. **Prompt Hub**: Version and share prompts
6. **Annotations**: Human feedback on traces
7. **Playground**: Test and iterate on prompts
8. **Collaboration**: Share traces and datasets with team

## Getting Started

### 1. Sign Up

1. Go to https://smith.langchain.com
2. Sign up for a free account
3. Create an organization and project

### 2. Get API Key

1. Go to Settings → API Keys
2. Create a new API key
3. Save it securely

### 3. Setup Environment

```bash
pip install langsmith langchain
```

```python
import os

# Set environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-api-key"
os.environ["LANGCHAIN_PROJECT"] = "your-project-name"
```

Or use `.env` file:
```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_...
LANGCHAIN_PROJECT=my-project
```

## Tracing

### Automatic Tracing

Once configured, all LangChain operations are automatically traced:

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-key"

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# This will be automatically traced
llm = ChatOpenAI()
prompt = ChatPromptTemplate.from_template("Tell me about {topic}")
chain = prompt | llm

# Execution is traced to LangSmith
result = chain.invoke({"topic": "AI"})
```

### Manual Tracing (Non-LangChain Code)

Trace any Python function:

```python
from langsmith import traceable
from langsmith import Client

client = Client()

@traceable
def my_function(input_text: str) -> str:
    """This function will be traced"""
    # Your logic here
    result = process_text(input_text)
    return result

@traceable
def process_text(text: str) -> str:
    """Nested function also traced"""
    return text.upper()

# Call function - automatically traced
result = my_function("hello")
```

### Nested Tracing

```python
from langsmith import traceable
import openai

@traceable
def retrieve_documents(query: str) -> list:
    """Retrieve relevant documents"""
    # Retrieval logic
    return ["doc1", "doc2"]

@traceable
def generate_answer(query: str, documents: list) -> str:
    """Generate answer from documents"""
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": query}]
    )
    return response.choices[0].message.content

@traceable(run_type="chain")
def rag_pipeline(query: str) -> str:
    """Full RAG pipeline"""
    docs = retrieve_documents(query)
    answer = generate_answer(query, docs)
    return answer

# All three functions traced with hierarchy
result = rag_pipeline("What is LangSmith?")
```

### Adding Metadata and Tags

```python
from langsmith import traceable

@traceable(
    run_type="llm",
    name="custom-name",
    metadata={"version": "1.0", "env": "prod"},
    tags=["production", "rag"]
)
def my_llm_call(prompt: str) -> str:
    # Your code
    return "response"
```

### Context Manager for Tracing

```python
from langsmith import trace

with trace(
    name="my-operation",
    run_type="chain",
    inputs={"query": "test"},
    tags=["experimental"]
) as run:
    result = perform_operation()
    run.end(outputs={"result": result})
```

## Datasets

Datasets are collections of examples for testing and evaluation.

### Creating Datasets

**Via UI**:
1. Go to Datasets in LangSmith
2. Click "Create Dataset"
3. Add examples manually or upload CSV

**Via Python**:

```python
from langsmith import Client

client = Client()

# Create dataset
dataset_name = "qa-examples"
dataset = client.create_dataset(
    dataset_name=dataset_name,
    description="Question answering examples"
)

# Add examples
examples = [
    {
        "inputs": {"question": "What is AI?"},
        "outputs": {"answer": "Artificial Intelligence is..."}
    },
    {
        "inputs": {"question": "What is ML?"},
        "outputs": {"answer": "Machine Learning is..."}
    }
]

for example in examples:
    client.create_example(
        dataset_id=dataset.id,
        inputs=example["inputs"],
        outputs=example["outputs"]
    )
```

### Loading Datasets

```python
# List datasets
datasets = client.list_datasets()

# Read dataset
dataset = client.read_dataset(dataset_name="qa-examples")

# Get examples
examples = client.list_examples(dataset_id=dataset.id)
for example in examples:
    print(example.inputs, example.outputs)
```

### Updating Examples

```python
# Update example
client.update_example(
    example_id=example.id,
    inputs={"question": "Updated question?"},
    outputs={"answer": "Updated answer"}
)

# Delete example
client.delete_example(example_id=example.id)
```

## Evaluation

Evaluate your LLM application against a dataset.

### Basic Evaluation

```python
from langsmith import Client, evaluate
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

client = Client()

# Define your application
def my_app(inputs: dict) -> dict:
    llm = ChatOpenAI()
    prompt = ChatPromptTemplate.from_template(
        "Answer this question: {question}"
    )
    chain = prompt | llm
    response = chain.invoke(inputs)
    return {"answer": response.content}

# Run evaluation
results = evaluate(
    my_app,
    data="qa-examples",  # dataset name
    experiment_prefix="qa-eval",
    metadata={"version": "1.0"}
)

print(results)
```

### Custom Evaluators

**Simple Evaluator**:

```python
def accuracy_evaluator(run, example):
    """Check if answer contains expected text"""
    predicted = run.outputs["answer"]
    expected = example.outputs["answer"]

    score = 1 if expected.lower() in predicted.lower() else 0

    return {
        "key": "accuracy",
        "score": score
    }

results = evaluate(
    my_app,
    data="qa-examples",
    evaluators=[accuracy_evaluator]
)
```

**LLM-as-Judge Evaluator**:

```python
from langsmith.evaluation import LangChainStringEvaluator

# Use built-in evaluators
qa_evaluator = LangChainStringEvaluator(
    "qa",
    config={"llm": ChatOpenAI(model="gpt-4", temperature=0)},
    prepare_data=lambda run, example: {
        "prediction": run.outputs["answer"],
        "reference": example.outputs["answer"],
        "input": example.inputs["question"]
    }
)

results = evaluate(
    my_app,
    data="qa-examples",
    evaluators=[qa_evaluator]
)
```

**Custom LLM Evaluator**:

```python
from langsmith.evaluation import evaluator

@evaluator
def relevance_evaluator(run, example):
    """Evaluate answer relevance using LLM"""
    llm = ChatOpenAI(model="gpt-4", temperature=0)

    prompt = f"""
    Question: {example.inputs['question']}
    Answer: {run.outputs['answer']}
    Expected: {example.outputs['answer']}

    Is the answer relevant and correct? Rate 0-1.
    Return only a number.
    """

    response = llm.invoke(prompt)
    score = float(response.content.strip())

    return {
        "key": "relevance",
        "score": score,
        "comment": "LLM-based relevance score"
    }

results = evaluate(
    my_app,
    data="qa-examples",
    evaluators=[relevance_evaluator]
)
```

### Multiple Evaluators

```python
from langsmith.evaluation import evaluate

results = evaluate(
    my_app,
    data="qa-examples",
    evaluators=[
        accuracy_evaluator,
        relevance_evaluator,
        qa_evaluator
    ],
    experiment_prefix="multi-metric-eval"
)

# Access results
print(f"Accuracy: {results['accuracy']}")
print(f"Relevance: {results['relevance']}")
```

### Comparing Experiments

```python
# Run multiple experiments with different configs
experiments = []

# Experiment 1: GPT-4
def app_v1(inputs):
    llm = ChatOpenAI(model="gpt-4")
    return {"answer": llm.invoke(inputs["question"]).content}

experiments.append(
    evaluate(app_v1, data="qa-examples", experiment_prefix="gpt4")
)

# Experiment 2: GPT-3.5
def app_v2(inputs):
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    return {"answer": llm.invoke(inputs["question"]).content}

experiments.append(
    evaluate(app_v2, data="qa-examples", experiment_prefix="gpt35")
)

# Compare in UI or programmatically
```

## Monitoring

### Production Monitoring

**Setup**:
```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-key"
os.environ["LANGCHAIN_PROJECT"] = "production"

# All production calls are now monitored
```

**Metadata for Filtering**:
```python
from langsmith import traceable

@traceable(
    metadata={
        "user_id": "user-123",
        "environment": "production",
        "version": "2.0"
    },
    tags=["production", "critical"]
)
def production_endpoint(query: str) -> str:
    # Your production code
    return result
```

### Querying Traces

```python
from langsmith import Client
from datetime import datetime, timedelta

client = Client()

# Get recent traces
runs = client.list_runs(
    project_name="production",
    start_time=datetime.now() - timedelta(hours=24),
    filter='eq(status, "error")'  # Only errors
)

for run in runs:
    print(f"Error: {run.name} - {run.error}")
```

### Monitoring Metrics

In the UI, you can view:
- **Latency**: P50, P95, P99 latencies
- **Cost**: Token usage and costs
- **Error Rate**: Failed runs
- **Volume**: Request volume over time
- **User Analytics**: Per-user metrics

### Alerts (Enterprise)

Set up alerts for:
- Error rate thresholds
- Latency spikes
- Cost anomalies
- Custom metrics

## Annotations and Feedback

### Adding Feedback via UI

1. Open any trace in LangSmith
2. Click "Add Annotation"
3. Add score, comment, or correction

### Programmatic Feedback

```python
from langsmith import Client

client = Client()

# Add feedback to a run
client.create_feedback(
    run_id="run-id",
    key="user-rating",
    score=0.9,
    comment="Great response!"
)

# Add correction
client.create_feedback(
    run_id="run-id",
    key="correction",
    correction={"output": "Corrected answer"},
    comment="Should have mentioned X"
)
```

### Collecting User Feedback

```python
from langsmith import traceable, get_current_run_tree

@traceable
def chat_endpoint(message: str) -> str:
    run = get_current_run_tree()
    response = generate_response(message)

    # Store run_id for later feedback
    return {
        "response": response,
        "run_id": str(run.id)
    }

# Later, when user provides feedback
def submit_feedback(run_id: str, rating: int, comment: str):
    client = Client()
    client.create_feedback(
        run_id=run_id,
        key="user-rating",
        score=rating / 5.0,  # Normalize to 0-1
        comment=comment
    )
```

### Querying Feedback

```python
# Get feedback for runs
feedback = client.list_feedback(run_ids=[run.id])

for fb in feedback:
    print(f"{fb.key}: {fb.score} - {fb.comment}")
```

## Prompt Hub

Manage and version prompts centrally.

### Pushing Prompts

```python
from langchain import hub
from langchain.prompts import ChatPromptTemplate

# Create prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{input}")
])

# Push to hub
hub.push("myorg/my-prompt", prompt)
```

### Pulling Prompts

```python
# Pull latest version
prompt = hub.pull("myorg/my-prompt")

# Pull specific version
prompt = hub.pull("myorg/my-prompt:v2")

# Use in chain
chain = prompt | llm
result = chain.invoke({"input": "Hello"})
```

### Versioning

```python
# Push new version
hub.push("myorg/my-prompt", updated_prompt, new_repo=False)

# Tag version
hub.push("myorg/my-prompt:production", prompt)
```

## Playground

The Playground allows you to:
- Test prompts interactively
- Compare different models
- Adjust parameters (temperature, tokens, etc.)
- Save successful prompts to Hub
- Share playground sessions

Access at: https://smith.langchain.com/playground

## Advanced Features

### 1. Custom Run Types

```python
from langsmith import traceable

@traceable(run_type="retriever")
def retrieve_docs(query: str):
    # Retrieval logic
    return docs

@traceable(run_type="reranker")
def rerank_docs(query: str, docs: list):
    # Reranking logic
    return ranked_docs
```

### 2. Sampling Traces

Sample production traces to reduce costs:

```python
import os
import random

# Sample 10% of traces
if random.random() < 0.1:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
else:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
```

### 3. Async Tracing

```python
from langsmith import traceable
import asyncio

@traceable
async def async_function(input: str) -> str:
    result = await some_async_operation(input)
    return result

# Tracing works with async
await async_function("test")
```

### 4. Batch Operations

```python
# Create examples in batch
client.create_examples(
    dataset_id=dataset.id,
    inputs=[{"question": q} for q in questions],
    outputs=[{"answer": a} for a in answers]
)
```

### 5. Filtering and Search

```python
# Advanced filtering
runs = client.list_runs(
    project_name="my-project",
    filter='and(eq(metadata.version, "2.0"), gt(latency, 1000))',
    start_time=datetime.now() - timedelta(days=7)
)
```

## Best Practices

### Development

1. **Use Separate Projects**: Dev, staging, production
2. **Tag Experiments**: Use descriptive tags and metadata
3. **Version Prompts**: Use Prompt Hub for versioning
4. **Create Datasets Early**: Build test cases as you develop
5. **Annotate Traces**: Add comments to interesting traces

### Testing

1. **Build Comprehensive Datasets**: Cover edge cases
2. **Multiple Evaluators**: Use different metrics
3. **Regression Testing**: Run evals on every change
4. **Compare Experiments**: A/B test different approaches
5. **Set Baselines**: Establish baseline metrics early

### Production

1. **Enable Monitoring**: Always trace production
2. **Add Metadata**: Include user_id, session_id, etc.
3. **Collect Feedback**: Implement user feedback loops
4. **Set Alerts**: Monitor for errors and anomalies
5. **Sample if Needed**: Use sampling for high-volume apps
6. **Review Regularly**: Weekly review of production traces

### Evaluation

1. **LLM-as-Judge**: Use GPT-4 for complex evaluations
2. **Human Review**: Periodically review LLM judgments
3. **Multiple Metrics**: Don't rely on single metric
4. **Representative Data**: Ensure dataset represents real use
5. **Continuous Evaluation**: Run evals continuously

## Common Use Cases

### 1. Debugging Complex Chains

```python
# Trace shows:
# - Each step in the chain
# - Inputs/outputs at each step
# - Latency breakdown
# - Token usage per call
# - Errors with full stack traces
```

### 2. A/B Testing

```python
# Compare two versions
results_a = evaluate(app_v1, data="test-set", experiment_prefix="v1")
results_b = evaluate(app_v2, data="test-set", experiment_prefix="v2")

# Analyze differences in UI
```

### 3. Prompt Engineering

```python
# Iterate on prompts
for i, prompt_template in enumerate(prompt_variations):
    evaluate(
        lambda x: (prompt_template | llm).invoke(x),
        data="test-set",
        experiment_prefix=f"prompt-v{i}"
    )
```

### 4. Production Monitoring

```python
# Monitor production issues
@traceable(
    metadata={"env": "prod", "user_id": user_id},
    tags=["production"]
)
def production_handler(request):
    # Automatically monitored
    return response
```

### 5. User Feedback Loop

```python
# Collect and analyze user feedback
feedback_data = client.list_feedback(
    project_name="production",
    feedback_key="user-rating"
)

# Use low-rated runs to improve
```

## Integration Examples

### With LangChain

```python
# Automatic - just set environment variables
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# All LangChain operations are traced
```

### With LangGraph

```python
# Also automatic
from langgraph.graph import StateGraph

workflow = StateGraph(State)
# ... build graph

app = workflow.compile()

# All graph executions are traced with full state history
result = app.invoke({"input": "test"})
```

### With OpenAI SDK

```python
from langsmith import traceable
from langsmith.wrappers import wrap_openai
import openai

# Wrap OpenAI client
client = wrap_openai(openai.OpenAI())

# Now traced automatically
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}]
)
```

### With Custom Code

```python
from langsmith import traceable

@traceable
def my_custom_pipeline(input_data):
    step1 = preprocess(input_data)
    step2 = model_inference(step1)
    step3 = postprocess(step2)
    return step3

# Fully traced
```

## Pricing

Pricing changes over time. Please check https://smith.langchain.com for current plans.

## Resources

- **Documentation**: https://docs.smith.langchain.com
- **Website**: https://smith.langchain.com
- **Cookbook**: https://github.com/langchain-ai/langsmith-cookbook
- **Blog**: https://blog.langchain.dev/tag/langsmith
- **Discord**: https://discord.gg/langchain
