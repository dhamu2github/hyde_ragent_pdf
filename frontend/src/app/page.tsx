"use client";

import { useState, FormEvent, ChangeEvent, useEffect } from "react";

// Define types for MCP preferences
interface MCPPreferences {
  detail_level?: string;
  technical_level?: string;
  include_examples?: boolean;
  format_preference?: string;
  goal?: string;
  background_knowledge?: string;
}

interface MCPInfo {
  user_context: {
    goal: string;
    background_knowledge: string;
  };
  response_format: string;
  document_domain: string;
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [fileName, setFileName] = useState<string>("");
  const [uploading, setUploading] = useState<boolean>(false);
  const [uploadSuccess, setUploadSuccess] = useState<boolean>(false);
  const [uploadMessage, setUploadMessage] = useState<string>("");
  const [query, setQuery] = useState<string>("");
  const [answer, setAnswer] = useState<string>("");
  const [hypotheticalAnswer, setHypotheticalAnswer] = useState<string>("");
  const [isQuerying, setIsQuerying] = useState<boolean>(false);
  const [error, setError] = useState<string>("");
  const [history, setHistory] = useState<Array<{
    query: string;
    answer: string;
    hypotheticalAnswer: string | null;
    mcpInfo?: MCPInfo;
  }>>([]);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [showTooltip, setShowTooltip] = useState<boolean>(false);
  const [queryTime, setQueryTime] = useState<number | null>(null);
  const [backendStatus, setBackendStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  const [showHypothetical, setShowHypothetical] = useState<boolean>(true);
  const [showPreferences, setShowPreferences] = useState<boolean>(false);
  const [preferences, setPreferences] = useState<MCPPreferences>({
    detail_level: "balanced",
    technical_level: "moderate",
    include_examples: true,
    format_preference: "concise",
    background_knowledge: "general"
  });
  const [documentDomain, setDocumentDomain] = useState<string | null>(null);

  // Check if backend is available
  useEffect(() => {
    const checkBackend = async () => {
      try {
        const response = await fetch("http://localhost:8000/", { method: "GET" });
        if (response.ok) {
          setBackendStatus('online');
          // Get current preferences
          fetchPreferences();
        } else {
          setBackendStatus('offline');
        }
      } catch (err) {
        setBackendStatus('offline');
      }
    };
    
    checkBackend();
  }, []);

  // Fetch current preferences
  const fetchPreferences = async () => {
    try {
      const response = await fetch("http://localhost:8000/preferences");
      if (response.ok) {
        const data = await response.json();
        if (data.preferences) {
          setPreferences(prev => ({...prev, ...data.preferences}));
        }
        if (data.mcp_settings?.document_context?.domain) {
          setDocumentDomain(data.mcp_settings.document_context.domain);
        }
      }
    } catch (err) {
      console.error("Error fetching preferences:", err);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile.type !== "application/pdf") {
        setError("Please select a PDF file");
        return;
      }
      setFile(droppedFile);
      setFileName(droppedFile.name);
      setError("");
    }
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const selectedFile = e.target.files[0];
      if (selectedFile.type !== "application/pdf") {
        setError("Please select a PDF file");
        setFile(null);
        return;
      }
      setFile(selectedFile);
      setFileName(selectedFile.name);
      setError("");
    }
  };

  const handlePreferenceChange = (name: string, value: any) => {
    setPreferences(prev => ({...prev, [name]: value}));
  };

  const savePreferences = async () => {
    try {
      const response = await fetch("http://localhost:8000/preferences", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(preferences),
      });

      if (response.ok) {
        const result = await response.json();
        console.log("Preferences updated:", result);
        setShowPreferences(false);
      } else {
        const error = await response.json();
        setError(error.detail || "Failed to update preferences");
      }
    } catch (err) {
      setError(`Error updating preferences: ${err instanceof Error ? err.message : String(err)}`);
    }
  };

  const handleUpload = async (e: FormEvent) => {
    e.preventDefault();
    if (!file) {
      setError("Please select a file first");
      return;
    }

    setUploading(true);
    setUploadSuccess(false);
    setUploadMessage("");
    setError("");
    setHistory([]);
    setAnswer("");
    setHypotheticalAnswer("");
    setDocumentDomain(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://localhost:8000/upload", {
        method: "POST",
        body: formData,
      });

      const result = await response.json();

      if (response.ok) {
        setUploadSuccess(true);
        setUploadMessage(result.message || "File uploaded successfully");
        
        // Update document domain if detected
        if (result.detected_domain) {
          setDocumentDomain(result.detected_domain);
        }
        
        // Refresh preferences after upload
        fetchPreferences();
      } else {
        setError(result.detail || "Failed to upload file");
      }
    } catch (err) {
      setError(`Error uploading file: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setUploading(false);
    }
  };

  const handleQuery = async (e: FormEvent) => {
    e.preventDefault();
    if (!query.trim()) {
      setError("Please enter a query");
      return;
    }

    if (!uploadSuccess) {
      setError("Please upload a PDF first");
      return;
    }

    setIsQuerying(true);
    setError("");
    const startTime = Date.now();

    try {
      const response = await fetch("http://localhost:8000/query", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query }),
      });

      const result = await response.json();

      if (response.ok) {
        setAnswer(result.answer);
        setHypotheticalAnswer(result.hypothetical_answer || "");
        setHistory(prev => [...prev, { 
          query: query, 
          answer: result.answer,
          hypotheticalAnswer: result.hypothetical_answer || null,
          mcpInfo: result.mcp_info
        }]);
        setQueryTime(Date.now() - startTime);
        setQuery(""); // Clear the input for next question
      } else {
        setError(result.detail || "Failed to get answer");
      }
    } catch (err) {
      setError(`Error querying: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setIsQuerying(false);
    }
  };

  const toggleHypothetical = () => {
    setShowHypothetical(!showHypothetical);
  };

  const togglePreferences = () => {
    setShowPreferences(!showPreferences);
  };

  return (
    <main className="min-h-screen bg-gradient-to-b from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800 p-4 md:p-8">
      <div className="max-w-5xl mx-auto">
        {/* Header */}
        <header className="flex flex-col md:flex-row justify-between items-center mb-6 bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-700 p-4 rounded-lg shadow-lg text-white">
          <div>
            <h1 className="text-2xl md:text-3xl font-bold text-white leading-tight">
              PDF Insights
            </h1>
            <p className="text-blue-100 font-medium text-sm md:text-base -mt-0.5">
              Ask & Discover with Agentic-RAG <span className="bg-purple-500 bg-opacity-40 px-2 py-0.5 rounded-md text-white text-xs ml-1">HyDE</span>
              <span className="bg-blue-500 bg-opacity-40 px-2 py-0.5 rounded-md text-white text-xs ml-1">MCP</span>
            </p>
          </div>
          <div className="flex items-center space-x-2 mt-2 md:mt-0 bg-black bg-opacity-20 px-3 py-1 rounded-full">
            <span className={`inline-block w-2.5 h-2.5 rounded-full ${
              backendStatus === 'online' ? 'bg-green-400' : 
              backendStatus === 'offline' ? 'bg-red-400' : 'bg-yellow-400'
            }`}></span>
            <span className="text-xs text-yellow-100 font-medium">
              {backendStatus === 'online' ? 'API Connected' : 
               backendStatus === 'offline' ? 'API Disconnected' : 'Checking API...'}
            </span>
          </div>
        </header>
        
        {error && (
          <div className="bg-red-100 dark:bg-red-900/30 border border-red-400 dark:border-red-800 text-red-700 dark:text-red-400 px-4 py-3 rounded-lg mb-6 flex items-center">
            <span className="mr-2">⚠️</span>
            <p>{error}</p>
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Left Column - Upload Section */}
          <div className="md:col-span-1 bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4 flex items-center">
              <span className="mr-2">📄</span>
              Document Upload
            </h2>
            
            <form onSubmit={handleUpload} className="space-y-4">
              <div 
                className={`border-2 ${isDragging ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20' : 'border-dashed border-gray-300 dark:border-gray-600'} 
                rounded-lg p-8 text-center cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-all duration-200`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => document.getElementById('file-upload')?.click()}
              >
                {fileName ? (
                  <div className="flex flex-col items-center">
                    <span className="text-4xl text-blue-500 mb-2">📑</span>
                    <p className="text-gray-800 dark:text-gray-200 font-medium">{fileName}</p>
                    <p className="text-gray-500 dark:text-gray-400 text-xs mt-1">Click to change file</p>
                  </div>
                ) : (
                  <div className="flex flex-col items-center">
                    <span className="text-4xl text-gray-400 mb-3">📤</span>
                    <p className="text-gray-500 dark:text-gray-400">Drop your PDF here or click to browse</p>
                  </div>
                )}
                <input
                  id="file-upload"
                  type="file"
                  accept=".pdf"
                  onChange={handleFileChange}
                  className="hidden"
                />
              </div>
              
              <button
                type="submit"
                disabled={!file || uploading}
                className="w-full bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white font-medium px-4 py-2.5 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 flex justify-center items-center"
              >
                {uploading ? (
                  <>
                    <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Processing...
                  </>
                ) : (
                  'Upload & Process'
                )}
              </button>
            </form>
            
            {uploadSuccess && (
              <div className="mt-4 p-4 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 text-green-700 dark:text-green-400 rounded-lg">
                <p className="font-medium">✓ Document Ready</p>
                <p className="text-sm mt-1">{uploadMessage}</p>
                {documentDomain && (
                  <p className="text-sm mt-1">
                    <span className="font-medium">Detected Domain:</span> {documentDomain}
                  </p>
                )}
              </div>
            )}

            {uploadSuccess && (
              <>
                <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg">
                  <h3 className="font-medium text-blue-800 dark:text-blue-400 mb-2">Example Questions</h3>
                  <ul className="text-sm space-y-2">
                    <li className="cursor-pointer hover:text-blue-600 dark:hover:text-blue-400" 
                        onClick={() => setQuery("What are the main topics covered in this document?")}>
                      What are the main topics covered?
                    </li>
                    <li className="cursor-pointer hover:text-blue-600 dark:hover:text-blue-400" 
                        onClick={() => setQuery("Can you summarize this document in a few sentences?")}>
                      Summarize this document
                    </li>
                    <li className="cursor-pointer hover:text-blue-600 dark:hover:text-blue-400" 
                        onClick={() => setQuery("What are the key findings or conclusions?")}>
                      Key findings or conclusions
                    </li>
                  </ul>
                </div>
                
                <div className="mt-4">
                  <button 
                    onClick={togglePreferences}
                    className="w-full flex items-center justify-center text-sm text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 py-2 rounded-lg transition-colors"
                  >
                    <span className="mr-1">⚙️</span> {showPreferences ? "Hide" : "Show"} MCP Preferences
                  </button>
                </div>
                
                {showPreferences && (
                  <div className="mt-4 p-4 bg-indigo-50 dark:bg-indigo-900/20 border border-indigo-200 dark:border-indigo-800 rounded-lg">
                    <h3 className="font-medium text-indigo-800 dark:text-indigo-400 mb-3">MCP Preferences</h3>
                    
                    <div className="space-y-3">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Background Knowledge</label>
                        <select 
                          value={preferences.background_knowledge || "general"}
                          onChange={(e) => handlePreferenceChange("background_knowledge", e.target.value)}
                          className="w-full p-2 text-sm border border-gray-300 dark:border-gray-600 dark:bg-gray-700 rounded-md"
                        >
                          <option value="novice">Novice - Explain simply</option>
                          <option value="general">General - Average understanding</option>
                          <option value="expert">Expert - Use technical terms</option>
                        </select>
                      </div>
                      
                      <div>
                        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Detail Level</label>
                        <select 
                          value={preferences.detail_level || "balanced"}
                          onChange={(e) => handlePreferenceChange("detail_level", e.target.value)}
                          className="w-full p-2 text-sm border border-gray-300 dark:border-gray-600 dark:bg-gray-700 rounded-md"
                        >
                          <option value="simplified">Simplified - Brief overview</option>
                          <option value="balanced">Balanced - Moderate details</option>
                          <option value="detailed">Detailed - Comprehensive information</option>
                        </select>
                      </div>
                      
                      <div>
                        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Format Preference</label>
                        <select 
                          value={preferences.format_preference || "concise"}
                          onChange={(e) => handlePreferenceChange("format_preference", e.target.value)}
                          className="w-full p-2 text-sm border border-gray-300 dark:border-gray-600 dark:bg-gray-700 rounded-md"
                        >
                          <option value="concise">Concise - Straight to the point</option>
                          <option value="comprehensive">Comprehensive - Full explanations</option>
                          <option value="bullet_points">Bullet Points - List format</option>
                        </select>
                      </div>
                      
                      <div className="flex items-center">
                        <input
                          type="checkbox"
                          id="include-examples"
                          checked={preferences.include_examples ?? true}
                          onChange={(e) => handlePreferenceChange("include_examples", e.target.checked)}
                          className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                        />
                        <label htmlFor="include-examples" className="ml-2 block text-sm text-gray-700 dark:text-gray-300">
                          Include examples in answers
                        </label>
                      </div>
                      
                      <button
                        onClick={savePreferences}
                        className="w-full bg-indigo-600 hover:bg-indigo-700 text-white text-sm font-medium px-4 py-2 rounded-md mt-2"
                      >
                        Save Preferences
                      </button>
                    </div>
                  </div>
                )}
              </>
            )}
          </div>

          {/* Right Column - Query Section */}
          <div className="md:col-span-2 bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md flex flex-col">
            <h2 className="text-xl font-semibold mb-4 flex items-center">
              <span className="mr-2">🔍</span>
              Ask Questions
            </h2>
            
            <form onSubmit={handleQuery} className="mb-4">
              <div className="relative">
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder={uploadSuccess ? "Ask anything about the document..." : "Upload a document first..."}
                  className="w-full p-3 pr-12 border border-gray-300 dark:border-gray-600 dark:bg-gray-700 dark:text-white rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                  disabled={!uploadSuccess || isQuerying}
                />
                <button
                  type="submit"
                  disabled={!uploadSuccess || !query || isQuerying}
                  className="absolute right-2 top-2 bg-blue-600 text-white p-1.5 rounded-lg disabled:bg-gray-400"
                  title="Ask Question"
                >
                  {isQuerying ? (
                    <svg className="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                  ) : (
                    <span>🔍</span>
                  )}
                </button>
              </div>
            </form>

            {history.length > 0 && (
              <div className="mb-4 flex justify-end space-x-4">
                <button 
                  onClick={toggleHypothetical}
                  className="text-sm text-blue-600 dark:text-blue-400 hover:underline flex items-center"
                >
                  {showHypothetical ? "Hide" : "Show"} HyDE Document Passages
                  <span className="ml-1">{showHypothetical ? "👁️‍🗨️" : "👁️"}</span>
                </button>
              </div>
            )}

            <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg flex-grow overflow-y-auto min-h-[400px] p-4">
              {answer ? (
                <div className="space-y-6">
                  {history.map((item, index) => (
                    <div key={index} className="space-y-2">
                      <div className="bg-blue-50 dark:bg-blue-900/20 p-3 rounded-lg">
                        <p className="font-medium text-blue-800 dark:text-blue-400">Your Question:</p>
                        <p>{item.query}</p>
                      </div>
                      
                      {showHypothetical && item.hypotheticalAnswer && (
                        <div className="bg-purple-50 dark:bg-purple-900/20 p-3 rounded-lg border border-purple-100 dark:border-purple-800">
                          <p className="font-medium text-purple-800 dark:text-purple-400 flex items-center justify-between">
                            <span className="flex items-center">
                              <span className="mr-1">🧠</span> HyDE Document Passage
                            </span>
                            <span className="text-xs bg-purple-200 dark:bg-purple-800 px-2 py-0.5 rounded-md text-purple-800 dark:text-purple-200">
                              Generated for Retrieval
                            </span>
                          </p>
                          <p className="text-xs text-purple-600 dark:text-purple-300 italic mb-2">
                            This is a hypothetical document excerpt generated to find relevant content - not the actual answer
                          </p>
                          <div className="prose dark:prose-invert max-w-none text-purple-700 dark:text-purple-300 text-sm p-2 bg-white/30 dark:bg-black/10 rounded border border-purple-100 dark:border-purple-800/40">
                            {item.hypotheticalAnswer.split('\n').map((paragraph, i) => (
                              <p key={i} className={i > 0 ? "mt-2" : ""}>{paragraph}</p>
                            ))}
                          </div>
                        </div>
                      )}
                      
                      <div className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow-sm border border-gray-100 dark:border-gray-700">
                        <p className="font-medium text-gray-800 dark:text-gray-300 mb-2 flex items-center justify-between">
                          <span className="flex items-center">
                            <span className="mr-1">📝</span> Final Answer
                          </span>
                          <span className="text-xs bg-blue-100 dark:bg-blue-900 px-2 py-0.5 rounded-md text-blue-800 dark:text-blue-200">
                            Based on Document Content
                          </span>
                        </p>
                        <div className="prose dark:prose-invert max-w-none p-3 bg-gray-50 dark:bg-gray-700/50 rounded border border-gray-200 dark:border-gray-600/40">
                          {item.answer.split('\n').map((paragraph, i) => (
                            paragraph.trim().startsWith("•") || paragraph.trim().startsWith("-") || /^\d+\./.test(paragraph.trim()) ? (
                              <div key={i} className="ml-4" dangerouslySetInnerHTML={{ __html: paragraph.replace(/^\s*[-•]\s*(.+)$/, '• $1') }} />
                            ) : (
                              <p key={i} className={i > 0 ? "mt-3" : ""}>{paragraph}</p>
                            )
                          ))}
                        </div>
                      </div>
                      
                      {/* MCP Info */}
                      {item.mcpInfo && (
                        <div className="bg-indigo-50 dark:bg-indigo-900/10 p-2 rounded-lg text-xs">
                          <p className="font-medium text-indigo-800 dark:text-indigo-400 flex items-center">
                            <span className="mr-1">⚙️</span> MCP Context:
                          </p>
                          <div className="grid grid-cols-2 gap-2 mt-1 text-gray-600 dark:text-gray-400">
                            <div>
                              <span className="font-medium">Goal:</span> {item.mcpInfo.user_context.goal}
                            </div>
                            <div>
                              <span className="font-medium">Knowledge Level:</span> {item.mcpInfo.user_context.background_knowledge}
                            </div>
                            <div>
                              <span className="font-medium">Format:</span> {item.mcpInfo.response_format}
                            </div>
                            <div>
                              <span className="font-medium">Domain:</span> {item.mcpInfo.document_domain || "Not detected"}
                            </div>
                          </div>
                        </div>
                      )}
                      
                      {index === history.length - 1 && queryTime && (
                        <p className="text-xs text-right text-gray-500 dark:text-gray-400">
                          Response time: {(queryTime / 1000).toFixed(2)}s
                        </p>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <div className="h-full flex flex-col items-center justify-center text-gray-400 dark:text-gray-500">
                  {uploadSuccess ? (
                    <>
                      <span className="text-5xl mb-4 opacity-40">🔍</span>
                      <p>Ask a question about your document</p>
                    </>
                  ) : (
                    <>
                      <span className="text-5xl mb-4 opacity-40">📄</span>
                      <p>Upload a document to start questioning</p>
                    </>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>

        <footer className="mt-8 text-center text-sm text-gray-500 dark:text-gray-400">
          <p>PDF RAG Question Answering System powered by LangChain and FastAPI with HyDE and MCP</p>
          <div 
            className="inline-flex items-center mt-2 cursor-pointer hover:text-blue-600 dark:hover:text-blue-400"
            onMouseEnter={() => setShowTooltip(true)}
            onMouseLeave={() => setShowTooltip(false)}
          >
            <span className="mr-1">ℹ️</span> How it works
            {showTooltip && (
              <div className="absolute bottom-12 bg-white dark:bg-gray-800 p-3 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700 max-w-xs text-left">
                <p className="font-medium mb-1">HyDE-enhanced RAG with MCP</p>
                <ol className="list-decimal list-inside text-xs space-y-1">
                  <li>Your PDF is processed and indexed</li>
                  <li>Your preferences shape how the system responds (MCP)</li>
                  <li>Your question generates a <span className="text-purple-600 dark:text-purple-400">hypothetical document passage</span></li>
                  <li>This passage is used for semantic retrieval (not your original query)</li>
                  <li>The retrieved actual document sections are used to create your answer</li>
                  <li>The final answer is structured based on your MCP preferences</li>
                </ol>
              </div>
            )}
          </div>
        </footer>
      </div>
    </main>
  );
}
