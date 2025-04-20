# LangGraph Diagrams in the RAG Agentic System

This document illustrates the LangGraph workflows implemented in the system, showing the nodes, edges, and state transitions.

## HyDE (Hypothetical Document Embeddings) Workflow

### State Definition
```python
# State definition for HyDE workflow
class HyDEState(TypedDict):
    query: str
    hypothetical_answer: Optional[str]
    retrieved_documents: Optional[List[Document]]
    results: Optional[str]
    mcp: Optional[ModelContextProtocol]
```

### Node and Edge Diagram
```
┌──────────────┐
│              │
│    START     │
│              │
└──────┬───────┘
       │
       ▼
┌──────────────────────────┐
│                          │
│  generate_hypothetical   │
│                          │
└──────────────┬───────────┘
               │
               ▼
┌──────────────────────────┐
│                          │
│   retrieve_documents     │
│                          │
└──────────────┬───────────┘
               │
               ▼
┌──────────────────────────┐
│                          │
│    generate_answer       │
│                          │
└──────────────┬───────────┘
               │
               ▼
┌──────────────┐
│              │
│     END      │
│              │
└──────────────┘
```

### Node Functions

1. **generate_hypothetical_answer**:
   - Input: Query, MCP context
   - Process: Generates a hypothetical document passage that might answer the query
   - Output: Updated state with hypothetical_answer

2. **retrieve_with_hypothetical**:
   - Input: Hypothetical answer, query, MCP context
   - Process: Uses the hypothetical answer for vector similarity search
   - Output: Updated state with retrieved_documents

3. **generate_final_answer**:
   - Input: Retrieved documents, query, MCP context
   - Process: Creates final answer using retrieved content and MCP guidelines
   - Output: Updated state with results

## Agent Workflow

### State Definition
```python
# State definition for Agent workflow
class AgentState(TypedDict):
    messages: List[BaseMessage]
    context: Dict[str, Any]
```

### Node and Edge Diagram
```
┌──────────────┐
│              │
│    START     │
│              │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│              │
│    agent     │
│              │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────┐
│     has_tool_calls?          │
│                              │
└──────────────────────────────┘
       │                 │
       │                 │
       ▼                 ▼
┌──────────────┐   ┌─────────────┐
│              │   │             │
│    tools     │   │     END     │
│              │   │             │
└──────┬───────┘   └─────────────┘
       │
       │
       │
       └───────────────┐
                       │
                       ▼
                  ┌──────────────┐
                  │              │
                  │    agent     │
                  │              │
                  └──────────────┘
```

### Node Functions

1. **agent_node**:
   - Input: Messages, context dictionary
   - Process: Invokes LLM with messages to generate a response
   - Output: Updated state with new response message

2. **tools_node**:
   - Input: Messages (including tool calls), context dictionary
   - Process: Executes any tool calls found in the most recent message
   - Output: Updated state with tool execution results as new messages

3. **has_tool_calls** (Conditional Function):
   - Input: Current state
   - Process: Checks if the latest message contains tool calls
   - Output: Boolean determining whether to route to tools node or end the workflow

## LangGraph Construction Code

### HyDE Workflow Construction
```python
# Define the LangGraph workflow for HyDE
workflow = StateGraph(HyDEState)

# Add nodes to the graph
workflow.add_node("generate_hypothetical", generate_hypothetical_answer)
workflow.add_node("retrieve_documents", retrieve_with_hypothetical)
workflow.add_node("generate_answer", generate_final_answer)

# Define the edges
workflow.add_edge(START, "generate_hypothetical")
workflow.add_edge("generate_hypothetical", "retrieve_documents")
workflow.add_edge("retrieve_documents", "generate_answer")
workflow.add_edge("generate_answer", END)

# Compile the workflow
return workflow.compile()
```

### Agent Workflow Construction
```python
# Build the LangGraph workflow
workflow = StateGraph(AgentState)

# Add nodes to the graph
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tools_node)

# Add edges to the graph
workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    has_tool_calls,  # Use our custom tool_calls detection function
    {
        True: "tools",   # If tool call detected, go to tools node
        False: END    # If no tool call detected, end the workflow
    }
)
workflow.add_edge("tools", "agent")

# Compile the graph
return workflow.compile()
```

## HyDE Workflow State Transitions

```
┌───────────────────────────────────────────────────────┐
│ Initial State                                         │
├───────────────────────────────────────────────────────┤
│ query: "What are the main benefits of microservices?" │
│ hypothetical_answer: None                             │
│ retrieved_documents: None                             │
│ results: None                                         │
│ mcp: ModelContextProtocol(...)                        │
└───────────────────────────────────────────────────────┘
                           │
                           ▼
┌───────────────────────────────────────────────────────┐
│ After generate_hypothetical                           │
├───────────────────────────────────────────────────────┤
│ query: "What are the main benefits of microservices?" │
│ hypothetical_answer: "Microservices Architecture:     │
│   Benefits and Considerations. Microservices          │
│   architecture represents a structural approach..."    │
│ retrieved_documents: None                             │
│ results: None                                         │
│ mcp: ModelContextProtocol(...)                        │
└───────────────────────────────────────────────────────┘
                           │
                           ▼
┌───────────────────────────────────────────────────────┐
│ After retrieve_documents                              │
├───────────────────────────────────────────────────────┤
│ query: "What are the main benefits of microservices?" │
│ hypothetical_answer: "Microservices Architecture:     │
│   Benefits and Considerations..."                      │
│ retrieved_documents: [Document(...), Document(...),   │
│   Document(...), Document(...), Document(...)]        │
│ results: None                                         │
│ mcp: ModelContextProtocol(...)                        │
└───────────────────────────────────────────────────────┘
                           │
                           ▼
┌───────────────────────────────────────────────────────┐
│ After generate_answer                                 │
├───────────────────────────────────────────────────────┤
│ query: "What are the main benefits of microservices?" │
│ hypothetical_answer: "Microservices Architecture:     │
│   Benefits and Considerations..."                      │
│ retrieved_documents: [Document(...), Document(...),   │
│   Document(...), Document(...), Document(...)]        │
│ results: "The main benefits of microservices are:     │
│   1. Enhanced scalability: Individual services...     │
│   2. Improved fault isolation: Failures in one...     │
│   3. Technology flexibility: Different services..."   │
│ mcp: ModelContextProtocol(...)                        │
└───────────────────────────────────────────────────────┘
```

## Agent Workflow State Transitions

```
┌───────────────────────────────────────────────┐
│ Initial State                                 │
├───────────────────────────────────────────────┤
│ messages: [HumanMessage("Find information     │
│   about the deployment models in the doc")]   │
│ context: {...}                                │
└───────────────────────────────────────────────┘
                      │
                      ▼
┌───────────────────────────────────────────────┐
│ After agent node                              │
├───────────────────────────────────────────────┤
│ messages: [HumanMessage("Find information..."),│
│   AIMessage(content="", tool_calls=[{         │
│     "id": "call_abc123",                      │
│     "function": {                             │
│       "name": "search_document",              │
│       "arguments": "{"query":"deployment      │
│         models"}"                             │
│     }                                         │
│   }])]                                        │
│ context: {...}                                │
└───────────────────────────────────────────────┘
                      │
                      ▼
┌───────────────────────────────────────────────┐
│ After tools node                              │
├───────────────────────────────────────────────┤
│ messages: [HumanMessage("Find information..."),│
│   AIMessage(content="", tool_calls=[...]),    │
│   ToolMessage(content="Found relevant         │
│     information: The document describes       │
│     three deployment models...",              │
│     tool_call_id="call_abc123",               │
│     name="search_document")]                  │
│ context: {...}                                │
└───────────────────────────────────────────────┘
                      │
                      ▼
┌───────────────────────────────────────────────┐
│ After agent node (final)                      │
├───────────────────────────────────────────────┤
│ messages: [HumanMessage("Find information..."),│
│   AIMessage(content="", tool_calls=[...]),    │
│   ToolMessage(content="Found relevant..."),   │
│   AIMessage(content="The document discusses   │
│     three main deployment models:             │
│     1. On-premises deployment: ...            │
│     2. Cloud-based deployment: ...            │
│     3. Hybrid deployment: ...")]              │
│ context: {...}                                │
└───────────────────────────────────────────────┘
``` 