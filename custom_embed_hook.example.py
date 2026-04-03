# custom_embed_hook.example.py
#
# Copy this file, edit it, then point EMBED_HOOK_FILE at the result:
#
#   EMBED_HOOK_FILE=/custom_hooks/my_hook.py
#
# Mount it into the AI container (docker-compose.override.yml):
#
#   services:
#     ai:
#       volumes:
#         - ./my_hook.py:/custom_hooks/my_hook.py:ro
#
# The function signature below MUST be preserved exactly.
# The hook is called once per chunk, concurrently across all chunks in a
# document, so it is safe to perform async I/O (e.g. LiteLLM calls) here.

import litellm


async def format_chunk_for_embedding(chunk: str, meta, config) -> str:
    """
    Example hook: ask an LLM to summarise the chunk in one sentence, then
    prepend the summary (and standard context) before embedding.

    ``meta`` exposes:  meta.title, meta.correspondent, meta.document_date
    ``config`` exposes: config.effective_metadata_model, config.metadata_api_base, …
    """
    # --- Uncomment to enable LLM summarisation ---
    # response = await litellm.acompletion(
    #     model=config.effective_metadata_model,
    #     messages=[{
    #         "role": "user",
    #         "content": f"Summarise the following document excerpt in one sentence:\n\n{chunk}",
    #     }],
    #     max_tokens=64,
    #     **({"api_base": config.metadata_api_base} if config.metadata_api_base else {}),
    # )
    # summary = response.choices[0].message.content.strip()
    # context = (
    #     f"Summary: {summary}\n"
    #     f"Title: {meta.title or 'Unknown'}\n"
    #     f"Sender: {meta.correspondent or 'Unknown'}\n"
    #     f"Date: {meta.document_date or 'Unknown'}\n"
    #     f"---\n"
    # )
    # return f"{context}{chunk}"

    # Default fallback (identical to the built-in situated embedding header):
    context = (
        f"Title: {meta.title or 'Unknown'}\n"
        f"Sender: {meta.correspondent or 'Unknown'}\n"
        f"Date: {meta.document_date or 'Unknown'}\n"
        f"---\n"
    )
    return f"{context}{chunk}"
