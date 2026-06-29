from fastapi import APIRouter

from indexing.vector_store import get_vector_store
from generation.llm import get_llm


router = APIRouter()


@router.get("/health")
def health_check():
    """
    Health check endpoint.
    """

    status = {
        "status": "healthy",
        "vector_store": "unknown",
        "llm": "unknown",
    }

    # Check Vector Store
    try:
        get_vector_store()
        status["vector_store"] = "loaded"
    except Exception as e:
        status["vector_store"] = f"error: {str(e)}"

    # Check LLM
    try:
        get_llm()
        status["llm"] = "loaded"
    except Exception as e:
        status["llm"] = f"error: {str(e)}"

    return status