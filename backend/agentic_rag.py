"""
Agentic RAG (Retrieval-Augmented Generation) implementation using LangGraph.
This module extends the basic RAG implementation with agent capabilities.
"""
import os
import json
from typing import Dict, List, Any, TypedDict, Annotated, Sequence, Union, Optional
from dotenv import load_dotenv

# --- LangChain Imports ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI

# --- LangChain Core Imports ---
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    FunctionMessage,
    ToolMessage,
)
from langchain_core.tools import Tool, BaseTool
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- LangGraph Imports ---
from langgraph.graph import StateGraph, START, END

# --- MCP Import ---
from .mcp_builder import ModelContextProtocol, create_default_mcp, extract_document_metadata

# Load environment variables
load_dotenv()

# --- Configuration ---
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_DEVICE = "cpu"
DEFAULT_OPENAI_MODEL = "gpt-4o"
DEFAULT_OLLAMA_MODEL = "llama3"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150

# --- State Definition ---
class AgentState(TypedDict):
    """Represents the state of the agent during execution."""
    messages: List[Any]
    context: Dict[str, Any]

# --- HyDE State Definition ---
class HyDEState(TypedDict):
    """Represents the state for HyDE processing."""
    query: str
    hypothetical_answer: Optional[str]
    retrieved_documents: Optional[List[Any]]
    results: Optional[str]
    mcp: Optional[ModelContextProtocol]  # Added MCP to the HyDE state

# --- Custom Tool Schemas ---
class SearchDocumentInput(BaseModel):
    query: str = Field(description="The query to search for in the document")

class SummarizeContextInput(BaseModel):
    pass  # No inputs needed for summarization

# --- Custom Tools ---
def search_document_tool(context: Dict[str, Any]):
    """Create a search document tool with access to the context"""
    
    def _search_document(query: str) -> str:
        """
        Search the document for relevant information based on the query.
        """
        if not context.get("vector_store"):
            return "No document has been uploaded yet. Please upload a document first."
        
        vector_store = context["vector_store"]
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})  # Increased from 4 to 5 for more context
        docs = retriever.get_relevant_documents(query)
        
        if not docs:
            return "No relevant information found in the document. Please try a different question."
        
        # Format the retrieved documents with more detailed page/metadata info
        context_str = "\n\n".join([
            f"Document Chunk {i+1} [From page {doc.metadata.get('page', 'unknown')}]:\n{doc.page_content}" 
            for i, doc in enumerate(docs)
        ])
        
        # Update the context summaries
        if "current_summaries" not in context:
            context["current_summaries"] = ""
        context["current_summaries"] += f"\n\nRetrieved for query '{query}':\n{context_str}"
        
        return f"Found the following relevant information:\n\n{context_str}"
    
    return Tool.from_function(
        func=_search_document,
        name="search_document",
        description="Search the document for relevant information based on the query",
        args_schema=SearchDocumentInput
    )

def summarize_context_tool(context: Dict[str, Any]):
    """Create a summarize context tool with access to the context"""
    
    def _summarize_context() -> str:
        """
        Summarize the current context.
        """
        if not context.get("current_summaries"):
            return "No context to summarize yet."
        
        summaries = context["current_summaries"]
        return f"Current summaries of the document chunks:\n\n{summaries}"
    
    return Tool.from_function(
        func=_summarize_context,
        name="summarize_context",
        description="Get a summary of the current context that has been gathered so far",
        args_schema=SummarizeContextInput
    )

# --- Helper Functions ---
def load_and_split_pdf(file_path: str) -> list:
    """
    Loads a PDF, extracts text, and splits it into Langchain Document objects.
    """
    print(f"Loading and splitting PDF: {file_path}")
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        if not documents:
            print("Warning: No text could be extracted from the PDF.")
            return []

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv("CHUNK_SIZE", DEFAULT_CHUNK_SIZE)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP)),
            length_function=len,
            add_start_index=True,
        )
        chunks = text_splitter.split_documents(documents)

        print(f"Document split into {len(chunks)} chunks.")
        if not chunks:
            print("Warning: Splitting resulted in zero chunks.")
        return chunks
    except Exception as e:
        print(f"Error loading/splitting PDF {file_path}: {e}")
        raise

def create_embeddings_model():
    """
    Initializes the sentence transformer embedding model using Langchain's wrapper.
    """
    model_name = os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_EMBEDDING_MODEL)
    device = os.getenv("EMBEDDING_DEVICE", DEFAULT_EMBEDDING_DEVICE)
    print(f"Initializing embedding model: {model_name} on device: {device}")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': False}
        )
        print("HuggingFace Embeddings model initialized successfully.")
        return embeddings
    except Exception as e:
        print(f"Error initializing embedding model {model_name}: {e}")
        raise

def create_llm():
    """
    Initializes the Large Language Model using Langchain.
    Prioritizes OpenAI if OPENAI_API_KEY is set, otherwise falls back to Ollama.
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if openai_api_key and openai_api_key.startswith("sk-"):
        # --- Use OpenAI --- 
        model_name = os.getenv("OPENAI_MODEL_NAME", DEFAULT_OPENAI_MODEL)
        print(f"Initializing OpenAI LLM: model={model_name}")
        try:
            llm = ChatOpenAI(openai_api_key=openai_api_key, model_name=model_name, temperature=0)
            # Perform a simple test
            try:
                print("Testing OpenAI API connection...")
                response = llm.invoke("Respond with OK if you are ready.")
                print(f"OpenAI LLM initialized and tested successfully: {response.content}")
                return llm
            except Exception as openai_error:
                print(f"Warning: OpenAI LLM initialization failed: {openai_error}")
                if "401" in str(openai_error):
                    print("Error: Invalid or expired OpenAI API key. Please check your API key.")
                elif "insufficient_quota" in str(openai_error).lower():
                    print("Error: Insufficient quota or credits for OpenAI API.")
                elif "rate_limit" in str(openai_error).lower():
                    print("Error: Rate limited by OpenAI API. Please try again later.")
                else:
                    print(f"Error connecting to OpenAI API: {openai_error}")
                # Continue to Ollama fallback
        except Exception as e:
            print(f"Error initializing OpenAI LLM: {e}")
            # Continue to Ollama fallback
    else:
        print("OPENAI_API_KEY not found or invalid (should start with 'sk-'). Falling back to Ollama LLM.")
        
    # --- Fallback to Ollama --- 
    print("Falling back to Ollama LLM.")
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL)
    ollama_model = os.getenv("OLLAMA_MODEL_NAME", DEFAULT_OLLAMA_MODEL)
    print(f"Initializing Ollama LLM: model={ollama_model}, url={ollama_base_url}")
    try:
        llm = Ollama(base_url=ollama_base_url, model=ollama_model)
        # Perform a simple test invocation
        try:
            response = llm.invoke("Respond with OK if you are ready.")
            print(f"Ollama LLM initialized and tested successfully: {response}")
            return llm
        except Exception as ollama_error:
            print(f"Warning: Ollama LLM initialized but failed test invocation: {ollama_error}")
            print(f"Ensure the Ollama server is running and the model ({ollama_model}) is available/pulled.")
            raise
        return llm
    except Exception as e:
        print(f"Error initializing Ollama LLM: {e}")
        raise

def create_index(doc_chunks: list, embeddings) -> FAISS:
    """
    Creates a FAISS vector store (index) from the document chunks and embeddings.
    """
    if not doc_chunks:
        print("No document chunks provided to create index.")
        return None

    print(f"Creating FAISS index from {len(doc_chunks)} chunks...")
    try:
        vector_store = FAISS.from_documents(doc_chunks, embeddings)
        print("FAISS index (Langchain VectorStore) created successfully.")
        return vector_store
    except Exception as e:
        print(f"Error creating Langchain FAISS index: {e}")
        raise

# --- Custom tools condition function ---
def has_tool_calls(state: AgentState) -> bool:
    """Check if the last message contains tool calls."""
    messages = state["messages"]
    if not messages:
        return False
    
    last_message = messages[-1]
    if not isinstance(last_message, AIMessage):
        return False
    
    # Check if the message has tool calls
    return "tool_calls" in last_message.additional_kwargs and bool(last_message.additional_kwargs["tool_calls"])

# --- Modified HyDE Implementation ---
def setup_hyde_workflow(llm, embeddings, vector_store):
    """
    Sets up the HyDE (Hypothetical Document Embeddings) workflow using LangGraph.
    Now enhanced with Model Context Protocol for improved interactions.
    """
    print("Setting up HyDE workflow with LangGraph and MCP...")
    
    # Define the LangGraph workflow for HyDE
    workflow = StateGraph(HyDEState)
    
    # Define nodes for the workflow
    
    # Node 1: Generate a hypothetical answer to the query with MCP guidance
    def generate_hypothetical_answer(state: HyDEState) -> HyDEState:
        query = state["query"]
        mcp = state.get("mcp", create_default_mcp())
        
        print(f"Generating hypothetical answer for query with MCP: {query}")
        
        # Create a prompt that incorporates MCP
        mcp_text = mcp.to_prompt_string() if mcp else ""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are an expert at generating hypothetical document passages that could contain answers to questions.

            {mcp_text}
            
            Your task is to write a passage that might appear in a document that discusses the topic of the user's query.
            IMPORTANT: Do NOT directly answer the query - instead generate content that would help answer it.
            
            - Write as if you're creating a document excerpt or passage that contains relevant information
            - Include specific details, terminology, and context that would likely appear in such a document
            - Make the passage appear as if it was extracted from a textbook, report, or reference material
            - Focus on providing raw information rather than directly answering the query
            - Use formal, technical language appropriate for a reference document
            - Include related concepts and terminology that would typically surround this information
            
            The hypothetical document passage should be 1-3 paragraphs that provide information relevant to the query, but formatted
            as documentary content rather than a direct answer. This passage will be used for semantic search to find relevant 
            actual document sections."""),
            ("human", "{query}")
        ])
        
        # Generate the hypothetical answer
        response_chain = prompt | llm | StrOutputParser()
        hypothetical_answer = response_chain.invoke({"query": query})
        
        print(f"Generated hypothetical answer of length {len(hypothetical_answer)}")
        
        # Return updated state
        return {
            "query": query,
            "hypothetical_answer": hypothetical_answer,
            "retrieved_documents": None,
            "results": None,
            "mcp": mcp
        }
    
    # Node 2: Retrieve documents using the hypothetical answer
    def retrieve_with_hypothetical(state: HyDEState) -> HyDEState:
        hypothetical_answer = state["hypothetical_answer"]
        query = state["query"]
        mcp = state.get("mcp", create_default_mcp())
        
        if not hypothetical_answer:
            print("No hypothetical answer to use for embedding")
            return state
        
        print("Using hypothetical answer for embedding and retrieval with MCP guidance")
        
        # Embed the hypothetical answer rather than the original query
        # This is the key part of HyDE: using the hypothetical answer as the retrieval query
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        
        # Get documents based on the hypothetical answer
        docs = retriever.get_relevant_documents(hypothetical_answer)
        
        if not docs:
            print("No relevant documents found using hypothetical embedding")
            # Fall back to original query
            docs = retriever.get_relevant_documents(query)
            if not docs:
                print("No documents found with fallback query either")
        
        print(f"Retrieved {len(docs)} documents using hypothetical embedding")
        
        # Extract document metadata to enhance MCP if not already set
        if not mcp.document_context.domain and docs:
            doc_contents = [doc.page_content for doc in docs]
            metadata = extract_document_metadata(doc_contents)
            if "domain" in metadata:
                mcp.document_context.domain = metadata["domain"]
                print(f"Detected document domain: {metadata['domain']}")
        
        return {
            "query": query,
            "hypothetical_answer": hypothetical_answer,
            "retrieved_documents": docs,
            "results": None,
            "mcp": mcp
        }
    
    # Node 3: Generate the final answer based on retrieved documents with MCP guidance
    def generate_final_answer(state: HyDEState) -> HyDEState:
        query = state["query"]
        docs = state["retrieved_documents"]
        mcp = state.get("mcp", create_default_mcp())
        
        if not docs:
            print("No documents to generate answer from")
            return {
                **state,
                "results": "I couldn't find any relevant information on that topic in the document."
            }
        
        # Format retrieved documents
        context_str = "\n\n".join([
            f"Document Chunk {i+1} [From page {doc.metadata.get('page', 'unknown')}]:\n{doc.page_content}" 
            for i, doc in enumerate(docs)
        ])
        
        # Check response format preference
        format_instructions = ""
        response_format = mcp.response_guidelines.format.lower()
        print(f"Using response format in generate_final_answer: {response_format}")
        
        if "bullet" in response_format or "list" in response_format:
            format_instructions = """
            Format your response as a bullet-point list:
            • Use bullet points (•) for each main point or fact
            • Group related information under clear headings
            • Present information in a sequential, logical order
            • Keep each bullet point focused and concise
            """
        elif "comprehensive" in response_format or "detailed" in response_format:
            format_instructions = """
            Provide a comprehensive explanation with:
            • Thorough coverage of all relevant aspects
            • Detailed explanations with supporting examples
            • Well-structured paragraphs with clear transitions
            • In-depth analysis where appropriate
            • Multiple viewpoints or approaches when relevant
            """
        elif "concise" in response_format or "summary" in response_format:
            format_instructions = """
            Provide a concise response with:
            • Direct answers to the question
            • Brief, focused explanations
            • Only the most relevant information
            • Clear, straightforward language
            """
        
        # Create a prompt for generating the final answer with MCP guidance
        mcp_text = mcp.to_prompt_string() if mcp else ""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a helpful research assistant answering questions about documents.
            
            {mcp_text}
            
            Your task is to create a DIRECT and STRUCTURED answer to the user's question based ONLY on the provided document sections.
            
            {format_instructions}
            
            Follow these key instructions:
            1. Start with a clear, direct answer to the question.
            2. Structure your response in a way that's easy to read and understand.
            3. If appropriate for the query type, use bullet points or numbered lists.
            4. Be specific and concise - focus on answering exactly what was asked.
            5. If the document doesn't contain relevant information, say so clearly.
            6. DO NOT make up information or hallucinate facts not in the document.
            7. DO NOT say phrases like "according to the document" or "based on the provided sections".
            8. DO NOT repeat the same information that would be in a hypothetical passage - be direct and to-the-point.
            
            Ensure your response follows the guidelines in the Model Context Protocol above.
            Adapt your detail level, terminology, and style to match the user's background knowledge and preferences."""),
            ("human", f"Document sections:\n\n{context_str}\n\nQuestion: {query}")
        ])
        
        # Generate the final answer
        response_chain = prompt | llm | StrOutputParser()
        final_answer = response_chain.invoke({})
        
        print("Generated final answer based on HyDE retrieval with MCP guidance")
        
        return {
            "query": query,
            "hypothetical_answer": state["hypothetical_answer"],
            "retrieved_documents": docs,
            "results": final_answer,
            "mcp": mcp
        }
    
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

# --- Agent Setup ---
def setup_agent(llm, context: Dict[str, Any]):
    """
    Sets up the LangGraph agent with the necessary tools and workflow.
    """
    # Define the tools available to the agent
    tools = [
        search_document_tool(context),
        summarize_context_tool(context)
    ]
    
    # Create prompt templates for the agent
    system_message = SystemMessage(content="""
    You are a specialized research assistant with access to a document database. 
    Your task is to answer user queries based ONLY on the content of the documents.
    
    IMPORTANT RULES:
    1. Only answer based on the document content, NEVER from your general knowledge.
    2. If the document doesn't contain the information needed, clearly state that.
    3. Always cite specific sections from the document to support your answers.
    4. DO NOT make up information or hallucinate facts.
    5. DO NOT announce when you're going to use tools or describe your process.
    6. DO NOT say phrases like "please hold on" or "let me search".
    7. DO NOT say "according to the document" - just present the information directly.
    8. DO NOT refer to yourself or the fact that you're using tools.
    
    For each query, you should:
    1. IMMEDIATELY use the search_document tool to find relevant information 
    2. Analyze the retrieved information carefully
    3. If needed, use the summarize_context tool to review what you've gathered
    4. Provide comprehensive, accurate answers based solely on the document content
    """)
    
    # Define the state handler functions
    def agent_node(state: AgentState):
        """Process agent actions"""
        # Add context to the messages for tool execution
        messages = state["messages"]
        context_dict = state["context"]
        
        # Add the system message to guide the agent
        full_messages = [system_message] + messages
        
        # Execute the agent and get the response
        response = llm.invoke(full_messages)
        
        # Update the state with the agent's response
        return {"messages": messages + [response], "context": context_dict}
    
    def tools_node(state: AgentState):
        """Execute tools called by the agent"""
        # Get the most recent message
        messages = state["messages"]
        context_dict = state["context"]
        
        most_recent = messages[-1]
        
        # Process any tool calls
        if "tool_calls" in most_recent.additional_kwargs:
            tool_calls = most_recent.additional_kwargs["tool_calls"]
            
            # Execute each tool call
            for tool_call in tool_calls:
                # Extract the tool details
                tool_name = tool_call["function"]["name"]
                tool_args = json.loads(tool_call["function"]["arguments"])
                
                # Find the tool and execute it
                for tool in tools:
                    if tool.name == tool_name:
                        result = tool.invoke(tool_args)
                        
                        # Create a tool message with the result
                        tool_message = ToolMessage(
                            content=str(result),
                            tool_call_id=tool_call["id"],
                            name=tool_name
                        )
                        
                        # Add the tool message to the state
                        messages.append(tool_message)
                        break
        
        # Return the updated state
        return {"messages": messages, "context": context_dict}

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

# --- Update the query_document function to use MCP ---
def query_document(query: str, context: Dict[str, Any], checkpoint=None):
    """
    Process a user query using HyDE-based RAG with Model Context Protocol.
    This implementation first generates a hypothetical answer to the query,
    then uses that to perform retrieval.
    """
    try:
        # Check that we have a valid LLM
        llm = context.get("llm")
        if not llm:
            error_msg = "LLM not initialized. Please initialize models first."
            print(f"Error: {error_msg}")
            raise ValueError(error_msg)
        
        print(f"Processing query with HyDE and MCP: {query}")
        
        # Ensure we have a vector store before continuing
        if not context.get("vector_store"):
            print("No vector store available for retrieval")
            return {"final_answer": "I don't have any document to search from. Please upload a PDF document first.", "hypothetical_answer": None}
            
        # Get embeddings and vector store from context
        embeddings = context.get("embeddings")
        vector_store = context.get("vector_store")
        
        # Set up HyDE workflow (or reuse if already set up)
        if not context.get("hyde_workflow"):
            context["hyde_workflow"] = setup_hyde_workflow(llm, embeddings, vector_store)
        
        hyde_workflow = context["hyde_workflow"]
        
        # Create or get MCP from context
        mcp = context.get("mcp", create_default_mcp())
        
        # Get user preferences from context if available
        user_preferences = context.get("user_preferences", {})
        if user_preferences:
            # Update MCP with user preferences
            mcp.user_context.preferences.update(user_preferences)
            
        # Update MCP for this specific query if needed
        if "technical" in query.lower() or "detailed" in query.lower():
            mcp.user_context.preferences["technical_level"] = "high"
            mcp.user_context.preferences["detail_level"] = "detailed"
        elif "simple" in query.lower() or "beginner" in query.lower():
            mcp.user_context.preferences["technical_level"] = "low"
            mcp.user_context.preferences["detail_level"] = "simplified"
        
        # Execute the HyDE workflow with MCP
        print("Executing HyDE workflow with MCP...")
        hyde_result = hyde_workflow.invoke({
            "query": query,
            "hypothetical_answer": None,
            "retrieved_documents": None,
            "results": None,
            "mcp": mcp
        })
        
        # Get the final answer from the workflow result
        response = hyde_result["results"]
        hypothetical_answer = hyde_result["hypothetical_answer"]
        
        # If the response is empty or not helpful, provide a fallback
        if not response or not response.strip():
            print("Empty response received, providing fallback answer")
            response = "I couldn't find specific information about this in the document. Please try a different question or upload a document with relevant information."
        elif any(phrase in response.lower() for phrase in [
            "i need to examine", 
            "need more information", 
            "i don't have access", 
            "please provide",
            "to answer this question",
            "i would need to see"
        ]):
            print("Detected unhelpful response, providing better answer")
            response = "Based on the document you uploaded, I couldn't find specific information to answer this question. Please try asking something else about the document's content."
        
        print("Response generated successfully using HyDE with MCP")
        return {
            "final_answer": response, 
            "hypothetical_answer": hypothetical_answer
        }
        
    except Exception as e:
        print(f"Error during HyDE query processing with MCP: {e}")
        import traceback
        traceback.print_exc()
        raise 