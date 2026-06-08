from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL

_client = Groq(api_key=GROQ_API_KEY)


def generate_response(query, retrieved_chunks):
    """
    Generate a grounded answer from retrieved housing corpus chunks.
    Answers only from provided context — will not draw on outside knowledge.
    Returns a plain string.
    """
    if not retrieved_chunks:
        return (
            "I couldn't find anything relevant in the housing corpus. "
            "Try rephrasing your question or being more specific."
        )

    context_parts = []
    for i, chunk in enumerate(retrieved_chunks):
        context_parts.append(f"[{i+1}] ({chunk['source']})\n{chunk['text']}")

    context = "\n\n".join(context_parts)

    response = _client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant for NYU students looking for off-campus housing in New York City. "
                    "Answer using only the context provided below. "
                    "When the answer is present in the context, state it directly and confidently — do not hedge or say you cannot answer. "
                    "If the answer is genuinely not in the context, say so clearly — do not guess or draw on outside knowledge. "
                    "Always state which numbered source your answer comes from."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}",
            },
        ],
    )

    return response.choices[0].message.content
