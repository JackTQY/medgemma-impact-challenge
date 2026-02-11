"""
RAG / Vector DB lookup logic for medical guidelines.
Agents call this to verify claims against a knowledge base (e.g. HIPAA-compliant endpoint).
"""

from typing import List, Dict, Any


def lookup_guidelines(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Look up relevant guideline snippets for a clinical query.
    TODO: Connect to Vertex AI / vector store; return list of {text, score, source}.
    """
    # Stub: return empty until RAG pipeline is implemented
    return []
