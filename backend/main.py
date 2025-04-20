from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import time # To measure processing time
import traceback
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import agentic RAG logic and MCP
from . import agentic_rag
from .mcp_builder import ModelContextProtocol, create_default_mcp, create_user_specific_mcp

app = FastAPI(title="Agentic RAG Application API")

# Configure CORS
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global State (In-memory - NOT production-ready) ---
# Stores the state of the loaded models and data
# In a real app, you'd use a database, proper file storage, and maybe background tasks
app_state = {
    "uploaded_file_path": None,
    "vector_store": None, # Will hold the FAISS index/vector store
    "embeddings": None,
    "llm": None,
    "agent": None, # Field to store the LangGraph agent
    "hyde_workflow": None, # Field to store the HyDE workflow
    "mcp": None, # Field to store the Model Context Protocol
    "conversation_history": [], # Store conversation history
    "current_summaries": "", # Store accumulated context
    "upload_dir": os.getenv("UPLOAD_DIR", "uploads"), # Get upload dir from env or default
    "user_preferences": {} # Store user preferences for MCP
}
# --- End Global State ---

def initialize_models():
    """Loads models if they haven't been loaded yet."""
    if app_state["embeddings"] is None:
        print("Initializing embedding model...")
        try:
            app_state["embeddings"] = agentic_rag.create_embeddings_model()
        except Exception as e:
            error_message = f"Failed to initialize embedding model: {e}"
            print(f"Fatal: {error_message}")
            traceback.print_exc()
            # Decide how to handle this - maybe prevent uploads/queries?
            raise HTTPException(status_code=500, detail=error_message)

    if app_state["llm"] is None:
        print("Initializing LLM...")
        try:
            app_state["llm"] = agentic_rag.create_llm()
            if app_state["llm"] is None:
                raise ValueError("LLM initialization returned None")
        except Exception as e:
            error_message = f"Failed to initialize LLM: {e}"
            print(f"Fatal: {error_message}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=error_message)
        
    # Validate the models
    print("Checking models initialization...")
    if app_state["embeddings"] is None:
        raise HTTPException(status_code=500, detail="Embedding model is None after initialization")
    if app_state["llm"] is None:
        raise HTTPException(status_code=500, detail="LLM is None after initialization")
        
    # Initialize default MCP if not already done
    if app_state["mcp"] is None:
        app_state["mcp"] = create_default_mcp()
        print("Initialized default Model Context Protocol")

# Optional: Cleanup uploads on shutdown (won't work with --reload reliably)
# @app.on_event("shutdown")
# def shutdown_event():
#    agentic_rag.cleanup_uploads(app_state["upload_dir"])

class QueryRequest(BaseModel):
    query: str
    
class UserPreferences(BaseModel):
    """User preferences for MCP customization"""
    detail_level: Optional[str] = None  # "simplified", "balanced", "detailed"
    technical_level: Optional[str] = None  # "low", "moderate", "high"
    include_examples: Optional[bool] = None
    format_preference: Optional[str] = None  # "concise", "comprehensive", "bullet_points"
    goal: Optional[str] = None
    background_knowledge: Optional[str] = None  # "novice", "general", "expert"

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Endpoint to upload, process, and index a PDF file.
    """
    start_time = time.time()
    stored_file_path = None
    
    try:
        # Validate file is a PDF
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
        
        # Ensure models are ready
        initialize_models()

        # Clean up previous uploads if any (simple strategy)
        if app_state["uploaded_file_path"]:
            try:
                os.remove(app_state["uploaded_file_path"]) 
                print(f"Removed previous file: {app_state['uploaded_file_path']}")
            except OSError as e:
                print(f"Error removing previous file: {e}")
            
            # Reset state
            app_state["uploaded_file_path"] = None
            app_state["vector_store"] = None # Reset vector store too
            app_state["agent"] = None # Reset agent
            app_state["hyde_workflow"] = None # Reset HyDE workflow
            app_state["conversation_history"] = [] # Reset conversation history
            app_state["current_summaries"] = "" # Reset summaries

        # Create uploads directory if needed
        os.makedirs(app_state["upload_dir"], exist_ok=True)
        stored_file_path = os.path.join(app_state["upload_dir"], file.filename)
        
        # Save the new file
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded.")
        
        with open(stored_file_path, "wb") as buffer:
            buffer.write(content)
        
        app_state["uploaded_file_path"] = stored_file_path
        print(f"File '{file.filename}' saved to {stored_file_path}")

        # Process the PDF: Load, Split, Embed, Index
        print("Processing PDF...")
        doc_chunks = agentic_rag.load_and_split_pdf(stored_file_path)

        if not doc_chunks or len(doc_chunks) == 0:
            print("No chunks generated from PDF, aborting indexing.")
            raise HTTPException(status_code=400, detail="Could not extract text or split the PDF into processable chunks. The PDF might be empty, corrupted, or contain only images without text.")

        # Create vector store index
        app_state["vector_store"] = agentic_rag.create_index(doc_chunks, app_state["embeddings"])

        if not app_state["vector_store"]:
            raise HTTPException(status_code=500, detail="Failed to create vector store index.")

        # Clear any existing agent and HyDE workflow to force re-initialization
        app_state["agent"] = None
        app_state["hyde_workflow"] = None
        
        # Analyze document chunks to update MCP with document information
        from .mcp_builder import extract_document_metadata
        doc_contents = [chunk.page_content for chunk in doc_chunks[:10]]  # Use first 10 chunks for analysis
        metadata = extract_document_metadata(doc_contents)
        
        # Update MCP with document metadata
        if app_state["mcp"] is None:
            app_state["mcp"] = create_default_mcp()
            
        if "domain" in metadata:
            app_state["mcp"].document_context.domain = metadata["domain"]
            print(f"Detected document domain: {metadata['domain']}")

        end_time = time.time()
        processing_time = end_time - start_time
        print(f"PDF processed and indexed successfully in {processing_time:.2f} seconds. Created {len(doc_chunks)} chunks.")

        return {
            "filename": file.filename,
            "message": f"File processed and indexed successfully in {processing_time:.2f} seconds. Created {len(doc_chunks)} text chunks.",
            "chunk_count": len(doc_chunks),
            "detected_domain": app_state["mcp"].document_context.domain
        }

    except HTTPException as http_exc:
        # Clean up failed upload
        if stored_file_path and os.path.exists(stored_file_path):
            try: 
                os.remove(stored_file_path)
                print(f"Removed failed upload: {stored_file_path}")
            except OSError as cleanup_err:
                print(f"Error removing failed upload: {cleanup_err}")
        
        app_state["uploaded_file_path"] = None
        # Re-raise the exception
        raise http_exc
        
    except Exception as e:
        # Handle other unexpected errors
        error_message = f"Error during PDF upload/processing: {e}"
        print(error_message)
        traceback.print_exc()
        
        # Clean up failed upload
        if stored_file_path and os.path.exists(stored_file_path):
            try: 
                os.remove(stored_file_path)
                print(f"Removed failed upload: {stored_file_path}")
            except OSError:
                pass
        
        app_state["uploaded_file_path"] = None
        app_state["vector_store"] = None
        
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.post("/preferences")
async def update_preferences(preferences: UserPreferences):
    """
    Endpoint to update user preferences for MCP customization.
    These preferences will be used to guide the model's responses.
    """
    try:
        # Ensure models are initialized
        initialize_models()
        
        # Update user preferences in app state
        new_preferences = {}
        if preferences.detail_level is not None:
            new_preferences["detail_level"] = preferences.detail_level
        if preferences.technical_level is not None:
            new_preferences["technical_level"] = preferences.technical_level
        if preferences.include_examples is not None:
            new_preferences["include_examples"] = preferences.include_examples
        if preferences.format_preference is not None:
            new_preferences["format_preference"] = preferences.format_preference
            
        # Update app state
        app_state["user_preferences"].update(new_preferences)
        
        # Update MCP with user context if provided
        if app_state["mcp"] is None:
            app_state["mcp"] = create_default_mcp()
            
        # Update user context in MCP
        if preferences.goal is not None:
            app_state["mcp"].user_context.goal = preferences.goal
        if preferences.background_knowledge is not None:
            app_state["mcp"].user_context.background_knowledge = preferences.background_knowledge
            
        # Update preferences in MCP
        app_state["mcp"].user_context.preferences.update(new_preferences)
        
        print(f"Updated user preferences: {new_preferences}")
        
        return {
            "message": "User preferences updated successfully",
            "preferences": app_state["user_preferences"],
            "mcp_user_context": {
                "goal": app_state["mcp"].user_context.goal,
                "background_knowledge": app_state["mcp"].user_context.background_knowledge,
                "preferences": app_state["mcp"].user_context.preferences
            }
        }
        
    except Exception as e:
        error_message = f"Error updating preferences: {e}"
        print(error_message)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_message)

@app.get("/preferences")
async def get_preferences():
    """
    Endpoint to get current user preferences and MCP settings.
    """
    try:
        # Ensure models are initialized
        initialize_models()
        
        # Return current preferences and MCP settings
        return {
            "preferences": app_state["user_preferences"],
            "mcp_settings": {
                "user_context": {
                    "goal": app_state["mcp"].user_context.goal,
                    "background_knowledge": app_state["mcp"].user_context.background_knowledge,
                    "preferences": app_state["mcp"].user_context.preferences
                },
                "document_context": {
                    "document_type": app_state["mcp"].document_context.document_type,
                    "domain": app_state["mcp"].document_context.domain,
                    "key_terminology": app_state["mcp"].document_context.key_terminology
                }
            }
        }
    except Exception as e:
        error_message = f"Error getting preferences: {e}"
        print(error_message)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_message)

@app.post("/query")
async def query_pdf(request: QueryRequest):
    """
    Endpoint to ask a question about the most recently uploaded PDF using the HyDE-based RAG system
    with Model Context Protocol for improved interactions.
    Returns both the hypothetical answer generated by HyDE and the final answer.
    """
    start_time = time.time()
    
    # Check if a PDF has been uploaded
    if not app_state["uploaded_file_path"] or not app_state["vector_store"]:
        raise HTTPException(
            status_code=400, 
            detail="No PDF has been successfully uploaded and indexed yet. Please upload a file first."
        )

    # Ensure models are ready
    try:
        initialize_models()
    except HTTPException as e:
        # If models failed to load *after* upload, report it
        raise HTTPException(status_code=500, detail=f"Model loading error: {e.detail}")

    try:
        # Validate the query
        query_text = request.query.strip() if request.query else ""
        if not query_text:
            raise HTTPException(status_code=400, detail="Query text cannot be empty.")

        print(f"Received query for HyDE processing with MCP: {query_text}")

        # Keep track of the conversation history in the app state
        from langchain_core.messages import HumanMessage
        if not app_state.get("conversation_history"):
            app_state["conversation_history"] = []
        
        # Add the new query to the conversation history
        current_query_message = HumanMessage(content=query_text)
        app_state["conversation_history"].append(current_query_message)

        # Perform the query using HyDE-based retrieval with MCP
        print("Calling query_document function with HyDE and MCP...")
        try:
            # Initialize HyDE workflow if not already done
            if not app_state.get("hyde_workflow") and app_state["vector_store"] is not None:
                print("Initializing HyDE workflow...")
                app_state["hyde_workflow"] = agentic_rag.setup_hyde_workflow(
                    app_state["llm"],
                    app_state["embeddings"],
                    app_state["vector_store"]
                )
                
            # Make sure MCP is initialized
            if app_state["mcp"] is None:
                app_state["mcp"] = create_default_mcp()
                
            # Analyze query to update MCP if needed
            if "explain" in query_text.lower() or "how does" in query_text.lower():
                app_state["mcp"].user_context.goal = "Understand how something works"
                app_state["mcp"].response_guidelines.format = "step-by-step explanation"
            elif "compare" in query_text.lower() or "difference" in query_text.lower():
                app_state["mcp"].user_context.goal = "Compare and contrast concepts"
                app_state["mcp"].response_guidelines.format = "comparison with clear distinctions"
            elif "list" in query_text.lower() or "what are" in query_text.lower():
                app_state["mcp"].response_guidelines.format = "concise list"
            
            # Apply format_preference from user preferences to response_guidelines
            format_preference = app_state["user_preferences"].get("format_preference")
            if format_preference:
                if format_preference == "bullet_points":
                    app_state["mcp"].response_guidelines.format = "bullet point list"
                elif format_preference == "comprehensive":
                    app_state["mcp"].response_guidelines.format = "comprehensive explanation"
                elif format_preference == "concise":
                    app_state["mcp"].response_guidelines.format = "concise summary"
                    
            print(f"Using response format: {app_state['mcp'].response_guidelines.format}")
                
            result = agentic_rag.query_document(
                query=query_text,
                context=app_state,
                checkpoint=None  # Not used in the HyDE approach
            )
            
            # Extract both final and hypothetical answers
            final_answer = result["final_answer"]
            hypothetical_answer = result["hypothetical_answer"]
            
        except Exception as query_error:
            print(f"Error in query_document function: {query_error}")
            traceback.print_exc()
            # Remove the failed query from history
            if app_state["conversation_history"] and app_state["conversation_history"][-1] == current_query_message:
                app_state["conversation_history"].pop()
            
            # Provide a more informative error message
            error_message = str(query_error)
            if "openai" in error_message.lower() and "api" in error_message.lower():
                error_message = "Error connecting to the OpenAI API. Please check your API key or network connection."
            elif "ollama" in error_message.lower():
                error_message = "Error connecting to Ollama. Please ensure the Ollama server is running."
            else:
                error_message = "Failed to process your question. Please try a different question or upload another document."
            
            raise HTTPException(status_code=500, detail=f"Failed to answer query: {error_message}")

        # Check if the final answer is empty or unhelpful
        if not final_answer or not final_answer.strip():
            print("Empty response received, providing fallback answer")
            final_answer = "I couldn't find specific information to answer your question in the document. Please try a more specific question."
        elif any(phrase in final_answer.lower() for phrase in [
            "i need to examine", 
            "need more information", 
            "i don't have access", 
            "please provide",
            "to answer this question",
            "i would need to see"
        ]):
            print("Detected unhelpful response, providing better answer")
            final_answer = "Based on the document you uploaded, I couldn't find specific information to answer this question. Please try asking something else about the document's content."

        # Add the agent's response to the conversation history
        from langchain_core.messages import AIMessage
        app_state["conversation_history"].append(AIMessage(content=final_answer))
        
        end_time = time.time()
        query_time = end_time - start_time
        print(f"Query answered using HyDE with MCP in {query_time:.2f} seconds.")

        # Get current MCP settings for the response
        mcp_info = {
            "user_context": {
                "goal": app_state["mcp"].user_context.goal,
                "background_knowledge": app_state["mcp"].user_context.background_knowledge
            },
            "response_format": app_state["mcp"].response_guidelines.format,
            "document_domain": app_state["mcp"].document_context.domain
        }

        return {
            "answer": final_answer,
            "hypothetical_answer": hypothetical_answer,
            "processing_time": query_time,
            "mcp_info": mcp_info
        }

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        error_message = f"Failed to answer query: {str(e)}"
        print(f"Error during query: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_message)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Agentic RAG API with HyDE and Model Context Protocol. Use /upload to post a PDF, /preferences to customize, and /query to ask questions."}

# To run the backend server (from the root RAG directory):
# poetry install  (if you haven't already)
# cp backend/.env.example backend/.env  (and configure models/API keys if needed)
# poetry run uvicorn backend.main:app --reload --port 8000

if __name__ == "__main__":
    # Recommended to run with uvicorn for development/production
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) # Added reload=True for direct run convenience 