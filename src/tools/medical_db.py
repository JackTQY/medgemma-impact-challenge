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
    if not (query or "").strip():
        return []
    # Stub: return one mock hit when query present so auditor flow is testable
    return [
        {
            "text": "Guideline: assess cardiovascular risk in hypertension; consider ACE-i/ARB for diabetic kidney disease.",
            "score": 0.85,
            "source": "mock_guideline",
        }
    ][:top_k]
