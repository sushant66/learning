# LangChain

## Overview

LangChain is a framework for developing applications powered by large language models (LLMs). It provides a standardized interface for chains, integrations with various tools, and end-to-end chains for common applications.

**Official Documentation**: https://python.langchain.com/docs

## Core Concepts

### 1. Components

LangChain is built around several key components that can be composed together:

#### Models
- **LLMs**: Text completion models (e.g., GPT-3)
- **Chat Models**: Message-based models (e.g., GPT-4, Claude)
- **Text Embedding Models**: Convert text to vector representations

#### Prompts
- **Prompt Templates**: Parameterized prompts for dynamic input
- **Chat Prompt Templates**: Templates specifically for chat models
- **Example Selectors**: Choose examples to include in prompts

#### Chains
- Combine multiple components into a single workflow
- Sequential execution of LLM calls and other operations

#### Retrievers
- Interface for fetching relevant documents
- Used in RAG (Retrieval Augmented Generation) applications

#### Agents
- Use LLMs to decide which actions to take
- Can use tools to interact with external systems

## Installation

```bash
pip install langchain
pip install langchain-openai  # For OpenAI models
pip install langchain-anthropic  # For Claude models
```

## Key Features

### 1. Prompt Templates

Create reusable, parameterized prompts:

```python
from langchain_core.prompts import PromptTemplate

# Simple template
template = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?"
)

prompt = template.format(product="colorful socks")
```

**Chat Prompt Templates**:

```python
from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    ("user", "{user_input}")
])

messages = chat_template.format_messages(user_input="Hello!")
```

### 2. Language Models

#### Chat Models (Most Common)

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# OpenAI
llm = ChatOpenAI(model="gpt-4", temperature=0.7)

# Anthropic Claude
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")

# Invoke the model
response = llm.invoke("Explain quantum computing in simple terms")
print(response.content)
```

#### Streaming Responses

```python
for chunk in llm.stream("Tell me a short story"):
    print(chunk.content, end="", flush=True)
```

### 3. Chains (LCEL - LangChain Expression Language)

Modern way to create chains using the `|` operator:

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Define components
prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
model = ChatOpenAI()
output_parser = StrOutputParser()

# Create chain using LCEL
chain = prompt | model | output_parser

# Execute
result = chain.invoke({"topic": "programming"})
print(result)
```

**Sequential Chains**:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# First chain: Generate story
story_prompt = ChatPromptTemplate.from_template(
    "Write a short story about {topic}"
)
story_chain = story_prompt | ChatOpenAI()

# Second chain: Summarize story
summary_prompt = ChatPromptTemplate.from_template(
    "Summarize this story in one sentence: {story}"
)
summary_chain = summary_prompt | ChatOpenAI() | StrOutputParser()

# Combine chains
full_chain = (
    story_chain
    | (lambda x: {"story": x.content})
    | summary_chain
)

result = full_chain.invoke({"topic": "space exploration"})
```

### 4. Output Parsers

Convert LLM output into structured data:

```python
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class Person(BaseModel):
    name: str = Field(description="Person's name")
    age: int = Field(description="Person's age")
    occupation: str = Field(description="Person's occupation")

parser = PydanticOutputParser(pydantic_object=Person)

prompt = ChatPromptTemplate.from_template(
    "Extract person information from: {text}\n{format_instructions}"
)

chain = (
    prompt.partial(format_instructions=parser.get_format_instructions())
    | ChatOpenAI()
    | parser
)

result = chain.invoke({"text": "John is 30 years old and works as a software engineer"})
# result is a Person object
```

### 5. Memory

Add conversational memory to chains:

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

memory = ConversationBufferMemory()

conversation = ConversationChain(
    llm=ChatOpenAI(),
    memory=memory
)

conversation.predict(input="Hi, my name is Alice")
conversation.predict(input="What's my name?")  # Will remember "Alice"
```

**Memory Types**:
- **ConversationBufferMemory**: Stores all messages
- **ConversationBufferWindowMemory**: Keeps last N messages
- **ConversationSummaryMemory**: Summarizes conversation over time
- **ConversationTokenBufferMemory**: Keeps messages within token limit

### 6. RAG (Retrieval Augmented Generation)

Combine document retrieval with LLM generation:

```python
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Load documents
loader = TextLoader("document.txt")
documents = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)

# Create embeddings and vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)

# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Create RAG chain
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

system_prompt = """Use the following context to answer the question.
If you don't know, say you don't know.

Context: {context}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(ChatOpenAI(), prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Query
response = rag_chain.invoke({"input": "What is the main topic?"})
print(response["answer"])
```

### 7. Document Loaders

Load data from various sources:

```python
# Text files
from langchain_community.document_loaders import TextLoader
loader = TextLoader("file.txt")

# PDFs
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("document.pdf")

# Web pages
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://example.com")

# CSV
from langchain_community.document_loaders import CSVLoader
loader = CSVLoader("data.csv")

# Load documents
documents = loader.load()
```

### 8. Text Splitters

Split documents into manageable chunks:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

chunks = text_splitter.split_documents(documents)
```

**Other splitters**:
- `CharacterTextSplitter`: Split by character count
- `TokenTextSplitter`: Split by token count
- `MarkdownHeaderTextSplitter`: Split markdown by headers
- `RecursiveCharacterTextSplitter`: Smart splitting (recommended)

### 9. Vector Stores

Store and search document embeddings:

```python
from langchain_community.vectorstores import FAISS, Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# FAISS (local)
vectorstore = FAISS.from_documents(documents, embeddings)
vectorstore.save_local("faiss_index")
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Chroma (local/persistent)
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Search
results = vectorstore.similarity_search("query", k=3)
results_with_scores = vectorstore.similarity_search_with_score("query", k=3)
```

### 10. Agents and Tools

Create agents that can use tools to accomplish tasks:

```python
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain import hub

# Define tools
def search_tool(query: str) -> str:
    """Search for information"""
    return f"Results for: {query}"

def calculator(expression: str) -> str:
    """Calculate mathematical expressions"""
    return str(eval(expression))

tools = [
    Tool(
        name="Search",
        func=search_tool,
        description="Useful for searching information"
    ),
    Tool(
        name="Calculator",
        func=calculator,
        description="Useful for mathematical calculations"
    )
]

# Create agent
llm = ChatOpenAI(temperature=0)
prompt = hub.pull("hwchase17/openai-functions-agent")
agent = create_openai_functions_agent(llm, tools, prompt)

# Create executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Run
result = agent_executor.invoke({
    "input": "What is 25 * 4 + 10?"
})
```

**Built-in Tools**:
- **SerpAPI**: Google search
- **Wikipedia**: Wikipedia search
- **ArXiv**: Academic paper search
- **PythonREPL**: Execute Python code
- **Requests**: HTTP requests

### 11. Callbacks

Monitor and debug chain execution:

```python
from langchain.callbacks import StdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler

class MyCallbackHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"LLM started with prompts: {prompts}")

    def on_llm_end(self, response, **kwargs):
        print(f"LLM ended with response: {response}")

chain = prompt | llm

chain.invoke(
    {"topic": "AI"},
    config={"callbacks": [MyCallbackHandler()]}
)
```

### 12. Caching

Cache LLM responses to save costs and time:

```python
from langchain.cache import InMemoryCache, SQLiteCache
from langchain.globals import set_llm_cache

# In-memory cache
set_llm_cache(InMemoryCache())

# SQLite cache (persistent)
set_llm_cache(SQLiteCache(database_path=".langchain.db"))

# Now LLM calls will be cached
llm = ChatOpenAI()
llm.invoke("Tell me a joke")  # Makes API call
llm.invoke("Tell me a joke")  # Returns cached result
```

## Common Use Cases

### 1. Question Answering over Documents

```python
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load and index documents
documents = loader.load()
chunks = text_splitter.split_documents(documents)
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever()

# Create QA chain
qa_chain = create_retrieval_chain(retriever, question_answer_chain)
response = qa_chain.invoke({"input": "What is the main topic?"})
```

### 2. Chatbot with Memory

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

memory = ConversationBufferMemory()
chatbot = ConversationChain(
    llm=ChatOpenAI(),
    memory=memory,
    verbose=True
)

chatbot.predict(input="Hello!")
chatbot.predict(input="What did I just say?")
```

### 3. Data Extraction

```python
from langchain.chains import create_extraction_chain

schema = {
    "properties": {
        "name": {"type": "string"},
        "email": {"type": "string"},
        "phone": {"type": "string"}
    },
    "required": ["name"]
}

chain = create_extraction_chain(schema, ChatOpenAI())
result = chain.invoke("Contact John at john@example.com or 555-1234")
```

### 4. Summarization

```python
from langchain.chains.summarize import load_summarize_chain

chain = load_summarize_chain(
    ChatOpenAI(),
    chain_type="map_reduce"  # or "stuff", "refine"
)

summary = chain.invoke(documents)
```

## Best Practices

1. **Use LCEL**: Modern chain syntax with `|` operator
2. **Streaming**: Enable streaming for better UX in production
3. **Error Handling**: Wrap chains in try-except blocks
4. **Prompt Engineering**: Spend time crafting good prompts
5. **Chunking Strategy**: Choose appropriate chunk size for your use case
6. **Caching**: Use caching to reduce API costs
7. **Callbacks**: Implement callbacks for monitoring in production
8. **Vector Store Selection**: Choose based on scale and requirements

## Environment Variables

```bash
# OpenAI
export OPENAI_API_KEY="your-key"

# Anthropic
export ANTHROPIC_API_KEY="your-key"

# For search tools
export SERPAPI_API_KEY="your-key"
```

## Common Patterns

### Pattern 1: Simple LLM Chain
```python
chain = prompt | llm | output_parser
result = chain.invoke({"input": "data"})
```

### Pattern 2: RAG Pattern
```python
retriever = vectorstore.as_retriever()
chain = create_retrieval_chain(retriever, qa_chain)
```

### Pattern 3: Agent Pattern
```python
agent = create_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)
```

## Integration with LangSmith

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-langsmith-key"

# All chains will now be traced in LangSmith
chain = prompt | llm
chain.invoke({"input": "data"})
```

## Resources

- **Documentation**: https://python.langchain.com
- **GitHub**: https://github.com/langchain-ai/langchain
- **Discord**: https://discord.gg/langchain
- **Blog**: https://blog.langchain.dev
