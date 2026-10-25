"""Library processing with AI."""

from typing import Any

import orjson
from loguru import logger

from src.adapters.ai import AIProvider
from src.adapters import ProviderAdapter


# Prompt templates for AI processing
SNIPPET_PROMPT_TEMPLATE = """Analyze and split this file into useful code pieces.

File: {file_path}
Content: {content}

CRITICAL: You must return EXACTLY this JSON array format (no other text):

[
  {{
    "title": "Function name or section title",
    "description": "What this code does",
    "source": "{file_path}#snippet_1",
    "language": "python",
    "code": "actual code content here"
  }}
]

Rules:
- Return ONLY the JSON array, no markdown blocks, no explanations
- Split by functions, classes, or logical sections
- Each snippet must be complete
- If file is too simple, return empty array []
- Number snippets in order: #snippet_1, #snippet_2, etc.
- Always use English for title and description fields
"""

SYSTEM_PROMPT = """You are a JSON API. You output ONLY valid JSON arrays.

FORBIDDEN: explanations, markdown blocks, text before/after JSON
REQUIRED: Start with [ and end with ]
REQUIRED: Use English for all title and description fields
EXAMPLE OUTPUT: [{"title": "example function", "description": "test implementation",
"source": "file#snippet_1", "language": "python", "code": "def test(): pass"}]
"""


class Processor:
    """Library processor - handles AI library processing."""

    def __init__(self, ai_provider: AIProvider) -> None:
        """Initialize with AI provider.

        Args:
            ai_provider: AI provider for embeddings and chat.
        """
        self.ai = ai_provider

    async def process(self, files: dict[str, str]) -> list[dict[str, Any]]:
        """Process files and create embeddings.

        Args:
            files: Dict mapping file paths to content.

        Returns:
            List of snippets with embeddings.

        Raises:
            Exception: If AI processing fails.
        """
        snippets = []

        for file_path, file_content in files.items():
            if not file_content.strip():
                continue

            for snippet in await self._generate_snippets(file_content, file_path):
                # Convert snippet to text for embedding
                embedding_text = "\n".join(f"{k}: {v}" for k, v in snippet.items() if v)
                embedding_result = await self.ai.embedding(embedding_text)

                # Add vector and token count for search
                snippet["vector"] = embedding_result.embedding
                snippet["tokens"] = embedding_result.embedding_tokens
                snippets.append(snippet)

        return snippets

    async def generate_embedding(self, provider: ProviderAdapter) -> list[float]:
        """Create embedding for provider content.

        Args:
            provider: Content provider with title and description.

        Returns:
            List of embedding values.

        Raises:
            Exception: If embedding generation fails.
        """
        embedding_result = await self.ai.embedding(
            f"{provider.title} {provider.description}"
        )
        return embedding_result.embedding

    async def _generate_snippets(
        self, content: str, file_path: str
    ) -> list[dict[str, str | int | list[float]]]:
        """Create code pieces from content using AI.

        Args:
            content: File content to process.
            file_path: Path to the source file.

        Returns:
            Code snippets with metadata.

        Raises:
            orjson.JSONDecodeError: If AI response is not valid JSON.
        """
        prompt = SNIPPET_PROMPT_TEMPLATE.format(file_path=file_path, content=content)
        response = await self.ai.chat_completion(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )

        try:
            return orjson.loads(response)
        except orjson.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response for {file_path}: {e}")
            return []
