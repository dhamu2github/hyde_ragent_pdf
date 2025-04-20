# PDF Insights: Agentic RAG with HyDE and MCP

A sophisticated document question-answering system that combines Hypothetical Document Embeddings (HyDE) with Model Context Protocol (MCP) to provide accurate, personalized responses to questions about PDF documents.

![PDF Insights Application](./docs/images/app-screenshot.jpeg)

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Key Features](#key-features)
- [Technological Components](#technological-components)
- [System Workflow](#system-workflow)
- [Model Context Protocol (MCP)](#model-context-protocol-mcp)
- [Installation and Setup](#installation-and-setup)
- [Usage Guide](#usage-guide)
- [Development](#development)
- [License](#license)

## Overview

PDF Insights is an advanced document question-answering application built on the Retrieval-Augmented Generation (RAG) paradigm. It extends traditional RAG capabilities with:

1. **Hypothetical Document Embeddings (HyDE)**: Generates hypothetical document passages to improve retrieval relevance
2. **Model Context Protocol (MCP)**: Provides structured guidance to LLMs for personalized, consistent responses
3. **Agentic Capabilities**: Uses LangGraph to implement a workflow-based approach to document processing

The system allows users to upload PDF documents, ask questions about them, and receive accurate answers tailored to their preferences and knowledge level.

## Architecture

```
┌─────────────────────────────────────┐     ┌─────────────────────────────────────┐
│                                     │     │                                     │
│           Frontend (Next.js)        │     │           Backend (FastAPI)         │
│                                     │     │                                     │
│ ┌───────────────┐  ┌──────────────┐ │     │ ┌───────────────┐  ┌──────────────┐ │
│ │ PDF Upload    │  │ Preferences  │ │     │ │ PDF Processing│  │ MCP Builder  │ │
│ │ Component     │  │ Panel        │ │     │ │ & Indexing    │  │              │ │
│ └───────────────┘  └──────────────┘ │     │ └───────────────┘  └──────────────┘ │
│                                     │     │                                     │
│ ┌───────────────┐  ┌──────────────┐ │     │ ┌───────────────┐  ┌──────────────┐ │
│ │ Query Input   │  │ Answer       │ │     │ │ HyDE Workflow │  │ Vector Store │ │
│ │ & History     │  │ Display      │ │     │ │ (LangGraph)   │  │ (FAISS)      │ │
│ └───────────────┘  └──────────────┘ │     │ └───────────────┘  └──────────────┘ │
│                                     │     │                                     │
└─────────────────────────────────────┘     └─────────────────────────────────────┘
                  │                                           │
                  │        HTTP/REST API Calls               │
                  └───────────────────────────────────────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────────┐
                    │        External Services         │
                    │                                  │
                    │  ┌───────────────┐  ┌─────────┐ │
                    │  │ OpenAI API    │  │ Ollama  │ │
                    │  │ (GPT-4o)      │  │ (Local) │ │
                    │  └───────────────┘  └─────────┘ │
                    │                                  │
                    └─────────────────────────────────┘
```

## Key Features

- **PDF Processing**: Upload, parse, and index PDF documents for semantic search
- **HyDE-Enhanced RAG**: Improved retrieval through hypothetical document passages
- **Personalized Responses**: Customize detail level, technical complexity, and format
- **Domain Adaptation**: Automatic document domain detection and terminology adaptation
- **Responsive UI**: Modern interface with dark/light mode and responsive design
- **Ethical Guidelines**: Built-in constraints to ensure responsible AI responses

## Technological Components

### Frontend
- **Next.js & React**: Building the user interface
- **Tailwind CSS**: Styling and UI components
- **TypeScript**: Type-safe frontend development

### Backend
- **FastAPI**: High-performance API framework
- **LangChain**: LLM orchestration and prompting
- **LangGraph**: Workflow management for RAG processes
- **FAISS**: Vector storage and similarity search
- **PyPDF**: PDF parsing and text extraction
- **Pydantic**: Data validation and settings management

### Machine Learning
- **Embeddings**: Hugging Face sentence transformers
- **LLM**: OpenAI GPT-4o (with Ollama fallback)
- **Vector Search**: FAISS for efficient similarity searching
- **Chunking**: Recursive character text splitter with metadata preservation

## System Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Document Processing Flow                           │
└─────────────────────────────────────────────────────────────────────────────┘
   │
   ▼
┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐
│ Upload PDF │ -> │ Parse Text │ -> │ Split into │ -> │ Create     │
│            │    │            │    │ Chunks     │    │ Embeddings │
└────────────┘    └────────────┘    └────────────┘    └────────────┘
                                                          │
┌─────────────────────────────────────────────────────┐   │
│                 Query Processing Flow               │   │
└─────────────────────────────────────────────────────┘   │
   │                                                      ▼
   ▼                                                ┌────────────┐
┌────────────┐    ┌────────────────────────┐       │ FAISS      │
│ User Query │ -> │ MCP Context Generation │       │ Vector     │
│            │    │ (Preferences + Query)  │       │ Store      │
└────────────┘    └────────────────────────┘       └────────────┘
   │                       │                             ▲
   │                       ▼                             │
   │               ┌────────────────────────┐           │
   │               │ Hypothetical Document  │ ──────────┘
   │               │ Passage Generation     │   Vector Search
   │               └────────────────────────┘
   │                       │
   │                       ▼
   │               ┌────────────────────────┐
   │               │ Retrieval of Relevant  │
   │               │ Document Chunks        │
   │               └────────────────────────┘
   │                       │
   ▼                       ▼
┌────────────────────────────────────────┐
│ Final Answer Generation                │
│ (Query + Retrieved Chunks + MCP)       │
└────────────────────────────────────────┘
   │
   ▼
┌────────────┐
│ Response   │
│ to User    │
└────────────┘
```

## Model Context Protocol (MCP)

The Model Context Protocol is a structured approach to providing context and guidance to LLMs. In this application, MCP consists of:

### 1. User Context
- **Goal**: The user's primary objective (finding information, understanding concepts, etc.)
- **Background Knowledge**: User's familiarity with the domain (novice, general, expert)
- **Preferences**: Desired detail level, technical complexity, and format 

### 2. Response Guidelines
- **Format**: Structure of the response (concise, comprehensive, bullet points)
- **Style**: Tone and approach to use (informative, step-by-step, comparison)
- **Constraints**: Limitations on what can be included in responses

### 3. Document Context
- **Document Type**: Kind of document being queried (automatically detected)
- **Domain**: Subject area the document covers (automatically detected)
- **Key Terminology**: Important domain-specific terms (extracted from document)

### 4. Ethical Guidelines
- Rules ensuring AI responses are accurate, helpful, and responsible

See [MCP Diagram](./docs/mcp-diagram.txt) for a visual representation of the MCP architecture.

For a detailed comparison between traditional RAG and our MCP-enhanced approach, see [RAG vs MCP-RAG](./docs/rag-vs-mcp-rag.md).

## Hypothetical Document Embeddings (HyDE)

This system uses Hypothetical Document Embeddings (HyDE) to improve retrieval accuracy by bridging the semantic gap between user queries and document content. Learn more about this approach in [HyDE Explained](./docs/hyde-explained.md).

## Installation and Setup

### Prerequisites
- Python 3.9+
- Node.js 18+
- npm 9+

### Backend Setup
```bash
# Navigate to backend directory
cd backend

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy .env.example to .env and edit as needed
cp .env.example .env

# Start the backend server
uvicorn backend.main:app --reload --port 8000
```

### Frontend Setup
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start the frontend server
npm run dev
```

### LLM Configuration
The system can use either:
- OpenAI API (requires API key in .env file)
- Ollama (requires local Ollama installation with llama3 model)

## Usage Guide

1. **Upload a Document**: Use the upload panel to select and process a PDF
2. **Set Preferences**: Configure MCP preferences (optional)
3. **Ask Questions**: Enter questions about the document content
4. **View Results**: See both the hypothetical document passage and final answer
5. **Toggle Views**: Show/hide components as needed

## Development

### Project Structure
```
.
├── backend/
│   ├── agentic_rag.py     # RAG and HyDE implementation
│   ├── mcp_builder.py     # Model Context Protocol implementation 
│   ├── main.py            # FastAPI routes and endpoints
│   └── requirements.txt   # Python dependencies
│
├── frontend/
│   ├── src/               # React/Next.js source code 
│   ├── public/            # Static assets
│   └── package.json       # JavaScript dependencies
│
└── docs/                  # Documentation and assets
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Acknowledgments

- LangChain for RAG tools and components
- LangGraph for workflow orchestration
- The HyDE paper authors for the hypothetical document embeddings approach
