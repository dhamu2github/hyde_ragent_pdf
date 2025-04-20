"""
Model Context Protocol (MCP) Builder

This module implements the Model Context Protocol pattern for enhancing LLM interactions
by providing structured context and guidelines for model behavior.
"""
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

class UserContext(BaseModel):
    """User context information for MCP"""
    goal: str = Field(
        default="Find accurate information from the document",
        description="The user's primary goal or objective for the query"
    )
    background_knowledge: str = Field(
        default="general",
        description="The user's level of background knowledge (novice, general, expert)"
    )
    preferences: Dict[str, Any] = Field(
        default_factory=lambda: {
            "detail_level": "balanced",
            "technical_level": "moderate",
            "include_examples": True,
            "format_preference": "concise"
        },
        description="User preferences for response content and formatting"
    )

class ResponseGuidelines(BaseModel):
    """Guidelines for the model's response"""
    format: str = Field(
        default="clear and direct response",
        description="Preferred output format (bullet points, prose, etc.)"
    )
    style: str = Field(
        default="informative and professional",
        description="Tone and style for the response"
    )
    constraints: List[str] = Field(
        default_factory=lambda: [
            "Only include information found in the document",
            "Do not make up information",
            "Clearly indicate uncertainty when appropriate"
        ],
        description="Constraints on the model's response"
    )

class DocumentContext(BaseModel):
    """Context about the document being queried"""
    document_type: Optional[str] = Field(
        default=None,
        description="Type of document (academic paper, manual, report, etc.)"
    )
    domain: Optional[str] = Field(
        default=None,
        description="Domain or field the document belongs to"
    )
    key_terminology: List[str] = Field(
        default_factory=list,
        description="Important domain-specific terminology from the document"
    )
    
class EthicalGuidelines(BaseModel):
    """Ethical guidelines for the model's behavior"""
    guidelines: List[str] = Field(
        default_factory=lambda: [
            "Prioritize accuracy over completeness",
            "Do not generate harmful, illegal, unethical or deceptive content",
            "Respect privacy and confidentiality of information",
            "Provide balanced information without bias"
        ],
        description="Ethical guidelines for the model to follow"
    )

class ModelContextProtocol(BaseModel):
    """Complete Model Context Protocol object"""
    user_context: UserContext = Field(default_factory=UserContext)
    response_guidelines: ResponseGuidelines = Field(default_factory=ResponseGuidelines)
    document_context: DocumentContext = Field(default_factory=DocumentContext)
    ethical_guidelines: EthicalGuidelines = Field(default_factory=EthicalGuidelines)
    
    def to_prompt_string(self) -> str:
        """Convert the MCP to a formatted string for inclusion in prompts"""
        mcp_str = "# MODEL CONTEXT PROTOCOL\n\n"
        
        # User Context
        mcp_str += "## USER CONTEXT\n"
        mcp_str += f"- Goal: {self.user_context.goal}\n"
        mcp_str += f"- Background Knowledge: {self.user_context.background_knowledge}\n"
        mcp_str += "- Preferences:\n"
        for k, v in self.user_context.preferences.items():
            mcp_str += f"  - {k.replace('_', ' ').title()}: {v}\n"
        
        # Response Guidelines
        mcp_str += "\n## RESPONSE GUIDELINES\n"
        mcp_str += f"- Format: {self.response_guidelines.format}\n"
        mcp_str += f"- Style: {self.response_guidelines.style}\n"
        mcp_str += "- Constraints:\n"
        for constraint in self.response_guidelines.constraints:
            mcp_str += f"  - {constraint}\n"
        
        # Document Context (if available)
        if any([self.document_context.document_type, self.document_context.domain, self.document_context.key_terminology]):
            mcp_str += "\n## DOCUMENT CONTEXT\n"
            if self.document_context.document_type:
                mcp_str += f"- Document Type: {self.document_context.document_type}\n"
            if self.document_context.domain:
                mcp_str += f"- Domain: {self.document_context.domain}\n"
            if self.document_context.key_terminology:
                mcp_str += "- Key Terminology:\n"
                for term in self.document_context.key_terminology:
                    mcp_str += f"  - {term}\n"
        
        # Ethical Guidelines
        mcp_str += "\n## ETHICAL GUIDELINES\n"
        for guideline in self.ethical_guidelines.guidelines:
            mcp_str += f"- {guideline}\n"
            
        return mcp_str

def create_default_mcp() -> ModelContextProtocol:
    """Create a default MCP object"""
    return ModelContextProtocol()

def create_document_specific_mcp(
    document_type: Optional[str] = None,
    domain: Optional[str] = None,
    key_terminology: Optional[List[str]] = None
) -> ModelContextProtocol:
    """Create an MCP with document-specific information"""
    mcp = create_default_mcp()
    
    if document_type:
        mcp.document_context.document_type = document_type
    if domain:
        mcp.document_context.domain = domain
    if key_terminology:
        mcp.document_context.key_terminology = key_terminology
        
    return mcp

def create_user_specific_mcp(
    goal: Optional[str] = None,
    background_knowledge: Optional[str] = None,
    preferences: Optional[Dict[str, Any]] = None
) -> ModelContextProtocol:
    """Create an MCP with user-specific information"""
    mcp = create_default_mcp()
    
    if goal:
        mcp.user_context.goal = goal
    if background_knowledge:
        mcp.user_context.background_knowledge = background_knowledge
    if preferences:
        # Update, don't replace completely
        mcp.user_context.preferences.update(preferences)
        
    return mcp

def extract_document_metadata(doc_contents: List[str]) -> Dict[str, Any]:
    """
    Extract potential document metadata from content
    This is a simplified implementation - in a real system, this would
    use NLP techniques to better extract document type, domain, etc.
    """
    # Simple implementation - just count some keywords
    domain_keywords = {
        "medical": ["patient", "diagnosis", "treatment", "clinical", "disease", "health"],
        "technical": ["system", "implementation", "architecture", "software", "hardware", "code"],
        "legal": ["law", "regulation", "compliance", "legal", "rights", "court"],
        "financial": ["market", "investment", "financial", "stock", "asset", "portfolio"],
        "academic": ["research", "study", "methodology", "findings", "literature", "hypothesis"]
    }
    
    # Count occurrences of domain keywords
    domain_counts = {domain: 0 for domain in domain_keywords}
    combined_text = " ".join(doc_contents).lower()
    
    for domain, keywords in domain_keywords.items():
        for keyword in keywords:
            domain_counts[domain] += combined_text.count(keyword)
    
    # Find the domain with the most keyword occurrences
    detected_domain = max(domain_counts.items(), key=lambda x: x[1])
    
    # Only return the domain if it has a significant number of occurrences
    result = {}
    if detected_domain[1] > 5:
        result["domain"] = detected_domain[0]
    
    return result 