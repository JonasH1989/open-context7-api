"""Library storage operations."""

import uuid
from datetime import datetime
from typing import Any

from loguru import logger
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance
from qdrant_client.models import FieldCondition
from qdrant_client.models import Filter
from qdrant_client.models import MatchValue
from qdrant_client.models import PointStruct
from qdrant_client.models import VectorParams

from src.adapters import ProviderAdapter
from src.core.config import settings
from src.core.constants import DEFAULT_LIBRARY_TAG
from src.core.constants import QDRANT_LIBRARIES_COLLECTION_NAME
from src.core.enums import LibraryStatus
from src.core.enums import LibraryType
from src.core.errors import ResourceNotFoundError
from src.schemas.responses import LibraryDetail
from src.schemas.responses import LibrarySearchItem


class Storage:
    """Library storage manager.

    Handles all database operations for library storage using Qdrant vector database.
    Manages collections, document storage, and search operations.
    """

    def __init__(self, qdrant_client: AsyncQdrantClient | None = None) -> None:
        """Initialize storage client.

        Args:
            qdrant_client: Optional Qdrant client instance. If None, creates a new one.
        """
        self.qdrant = qdrant_client or AsyncQdrantClient(url=settings.QDRANT_URL)

    async def _ensure_index_collection_exists(self) -> None:
        """Ensure the libraries index collection exists."""
        if not await self.qdrant.collection_exists(QDRANT_LIBRARIES_COLLECTION_NAME):
            await self.qdrant.create_collection(
                collection_name=QDRANT_LIBRARIES_COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=settings.EMBEDDING_DIMENSION,
                    distance=Distance.COSINE,
                ),
            )

    async def initialize(
        self,
        library_id: str,
        title: str,
        description: str,
        library_vector: list[float],
        org: str,
        project: str,
        git_info: dict[str, Any] | None = None,
        library_type: LibraryType = LibraryType.GIT,
    ) -> None:
        """Initialize library storage.

        Args:
            library_id: Library unique ID.
            title: Library title.
            description: Library description.
            library_vector: Vector embedding for the library.
            org: Organization name.
            project: Project name.
            git_info: Optional Git repository info dict.
            library_type: Type of library creation (git or api).
        """
        await self._ensure_index_collection_exists()

        payload = {
            "id": library_id,
            "title": title,
            "description": description,
            "org": org,
            "project": project,
            "state": LibraryStatus.PROCESSING,
            "last_update_date": datetime.now().isoformat(),
            "tags": [],
            "library_type": library_type.value,
        }

        if git_info:
            payload.update(git_info)

        await self.qdrant.upsert(
            QDRANT_LIBRARIES_COLLECTION_NAME,
            [PointStruct(id=library_id, vector=library_vector, payload=payload)],
        )

        await self.qdrant.create_collection(
            collection_name=library_id,
            vectors_config=VectorParams(
                size=settings.EMBEDDING_DIMENSION, distance=Distance.COSINE
            ),
        )
        logger.info(f"{library_id}: Collections initialized")

    async def save_snippets(
        self, snippets: list[dict[str, Any]], library_id: str
    ) -> None:
        """Save code snippets to library collection.

        Stores processed code snippets with embeddings in the vector database.
        Each snippet gets a unique ID and metadata.

        Args:
            snippets: List of processed code snippets with vectors.
            library_id: Unique ID of the library collection.

        Raises:
            Exception: If storage operation fails.
        """
        if not snippets:
            logger.info(f"{library_id}: No snippets to store")
            return

        try:
            snippet_points = [
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=snippet["vector"],
                    payload={
                        "title": snippet["title"],
                        "description": snippet["description"],
                        "source": snippet["source"],
                        "language": snippet["language"],
                        "code": snippet["code"],
                        "tokens": snippet["tokens"],
                        "tag": snippet.get("tag", DEFAULT_LIBRARY_TAG),
                        "created_at": snippet.get(
                            "created_at", datetime.now().isoformat()
                        ),
                    },
                )
                for snippet in snippets
            ]
            await self.qdrant.upsert(library_id, snippet_points)
            logger.info(f"{library_id}: Stored {len(snippets)} snippets")
        except Exception as e:
            logger.error(f"{library_id}: Failed to store snippets: {e}")
            raise

    async def complete(self, provider: ProviderAdapter, total_tokens: int) -> None:
        """Complete library processing.

        Updates library status to finalized and stores processing statistics.
        Records commit ID for repository sources.

        Args:
            provider: Library provider that was processed.
            total_tokens: Total token count of processed library.

        Raises:
            Exception: If completion update fails.
        """
        commit_id = await provider.get_latest_commit_id()

        payload = {
            "state": LibraryStatus.FINALIZED,
            "last_update_date": datetime.now().isoformat(),
            "total_tokens": total_tokens,
        }

        # Add commit ID for repository sources
        if commit_id:
            payload["last_commit_id"] = commit_id

        await self.qdrant.set_payload(
            collection_name=QDRANT_LIBRARIES_COLLECTION_NAME,
            points=[provider.id],
            payload=payload,
        )

    async def cleanup_failed(self, provider: ProviderAdapter, error_message: str) -> None:
        """Clean up after failed library processing.

        Removes partial data and marks library as failed.
        Logs error details for debugging.

        Args:
            provider: Library provider that failed processing.
            error_message: Error details and traceback.
        """
        try:
            await self.qdrant.delete_collection(provider.id)
        except Exception as e:
            logger.debug(f"Failed to delete collection {provider.id}: {e}")

        try:
            await self.qdrant.set_payload(
                collection_name=QDRANT_LIBRARIES_COLLECTION_NAME,
                points=[provider.id],
                payload={
                    "state": LibraryStatus.FAILED,
                    "updated_at": datetime.now().isoformat(),
                    "error_message": error_message,
                },
            )
        except Exception as e:
            logger.debug(f"Failed to set failed status: {e}")

        logger.error(f"{provider.id}: Processing failed: {error_message}")

    async def query(
        self,
        library_id: str,
        query_vector: list[float],
        limit: int = 20,
        tag: str | None = None,
    ) -> list[dict[str, Any]]:
        """Query documents in a specific library using vector similarity.

        Performs semantic search in the library collection using query vector.
        Can filter by specific tag if provided.

        Args:
            library_id: Unique ID of library to query in.
            query_vector: Vector embedding of search query.
            limit: Maximum number of results (default: 20).
            tag: Optional tag filter.

        Returns:
            List of matching documents with similarity scores, ordered by relevance
            (highest score first).
        """
        tag_filter = (
            {"must": [{"key": "tag", "match": {"value": tag}}]} if tag else None
        )

        results = await self.qdrant.query_points(
            collection_name=library_id,
            query=query_vector,
            query_filter=tag_filter,
            limit=limit,
            with_payload=True,
        )

        return [{**point.payload, "score": point.score} for point in results.points]

    async def search(
        self,
        query_vector: list[float] | None = None,
        limit: int = 35,
        offset: int = 0,
    ) -> list[LibrarySearchItem]:
        """Search or list all libraries from index collection.

        Performs semantic search when query_vector is provided, otherwise lists all.
        Follows Context7 API behavior: fixed 35 results for search, pagination for list.

        Args:
            query_vector: Vector embedding of search query (None = list all).
            limit: Maximum number of results (default: 35 like Context7).
            offset: Skip this many results for pagination (ignored for vector search).

        Returns:
            List of library search items.
        """
        await self._ensure_index_collection_exists()

        if query_vector is None:
            # List all libraries using scroll - supports pagination
            scroll_result = await self.qdrant.scroll(
                collection_name=QDRANT_LIBRARIES_COLLECTION_NAME,
                limit=limit,
                offset=offset,
                with_payload=True,
            )
            return [LibrarySearchItem(**point.payload) for point in scroll_result[0]]
        else:
            # Semantic search for libraries - only uses limit, ignores offset
            results = await self.qdrant.query_points(
                collection_name=QDRANT_LIBRARIES_COLLECTION_NAME,
                query=query_vector,
                limit=limit,
                with_payload=True,
            )
            return [
                LibrarySearchItem(**point.payload, score=point.score)
                for point in results.points
            ]

    async def get_by_id(self, library_id: str) -> LibraryDetail:
        """Get library details by ID.

        Retrieves library metadata from the index collection.

        Args:
            library_id: Unique ID of the library.

        Returns:
            Library details object with metadata.

        Raises:
            ResourceNotFoundError: If library does not exist.
        """
        await self._ensure_index_collection_exists()

        index_result = await self.qdrant.retrieve(
            QDRANT_LIBRARIES_COLLECTION_NAME, [library_id]
        )

        if not index_result or not index_result[0].payload:
            raise ResourceNotFoundError("Library not found")

        return LibraryDetail(library_id=library_id, **index_result[0].payload)

    async def remove_tag(self, library_id: str, tag: str) -> None:
        """Remove all data for a specific library tag.

        Deletes all documents matching the tag filter from library collection.
        Used for rebuilding or cleaning up old tags.

        Args:
            library_id: Unique ID of the library.
            tag: Tag name to remove.

        Raises:
            Exception: If deletion operation fails.
        """
        try:
            await self.qdrant.delete(
                collection_name=library_id,
                points_selector=Filter(
                    must=[FieldCondition(key="tag", match=MatchValue(value=tag))]
                ),
            )
            logger.info(f"{library_id}: Cleared all data for tag={tag}")

        except Exception as e:
            logger.error(f"Failed to clear tag {tag} from {library_id}: {e}")
            raise

    @staticmethod
    async def _build_payload(
        provider: ProviderAdapter,
    ) -> dict[str, Any]:
        """Build storage payload for provider.

        Creates metadata payload by combining common storage fields
        with provider-specific information.

        Args:
            provider: Library provider to build payload for.

        Returns:
            Complete payload dictionary with all metadata.
        """
        # Build payload with common storage fields
        payload = {
            "id": provider.id,
            "title": provider.title,
            "description": provider.description,
            "state": LibraryStatus.PROCESSING,
            "last_update_date": datetime.now().isoformat(),
            "tags": [],
            **await provider.build_payload(),
        }

        return payload

    async def add_tag_to_index(self, library_id: str, tag: str) -> None:
        """Add tag to index collection.

        This method gets the current index record, adds the new tag to the
        tags list if it doesn't exist, sorts the list with latest first,
        and updates the index collection.

        Args:
            library_id: Library ID to update.
            tag: Tag name to add to list.

        Raises:
            Exception: If adding tag to index fails.
        """
        try:
            # Get current index record
            index_result = await self.qdrant.retrieve(
                collection_name=QDRANT_LIBRARIES_COLLECTION_NAME, ids=[library_id]
            )

            if index_result:
                current_payload = index_result[0].payload
                current_tags = current_payload.get("tags", [])

                if tag not in current_tags:
                    current_tags.append(tag)
                    current_tags.sort(reverse=True)  # Sort with latest first

                    # Update tags list using set_payload
                    await self.qdrant.set_payload(
                        collection_name=QDRANT_LIBRARIES_COLLECTION_NAME,
                        payload={"tags": current_tags},
                        points=[library_id],
                    )

        except Exception as e:
            logger.error(f"Failed to add tag {tag} to index for {library_id}: {e}")
            raise

    async def complete_library(self, library_id: str, total_tokens: int) -> None:
        """Complete library processing without depending on provider.

        Updates library status to finalized and stores processing statistics.

        Args:
            library_id: Library ID that was processed.
            total_tokens: Total token count of processed library.

        Raises:
            Exception: If completion update fails.
        """
        payload = {
            "state": LibraryStatus.FINALIZED,
            "last_update_date": datetime.now().isoformat(),
            "total_tokens": total_tokens,
        }

        await self.qdrant.set_payload(
            collection_name=QDRANT_LIBRARIES_COLLECTION_NAME,
            points=[library_id],
            payload=payload,
        )

    async def cleanup_failed_library(self, library_id: str, error_message: str) -> None:
        """Clean up after failed library processing without depending on provider.

        Removes partial data and marks library as failed.
        Logs error details for debugging.

        Args:
            library_id: Library ID that failed processing.
            error_message: Error details and traceback.
        """
        try:
            await self.qdrant.delete_collection(library_id)
        except Exception as e:
            logger.debug(f"Failed to delete collection {library_id}: {e}")

        try:
            await self.qdrant.set_payload(
                collection_name=QDRANT_LIBRARIES_COLLECTION_NAME,
                points=[library_id],
                payload={
                    "state": LibraryStatus.FAILED,
                    "updated_at": datetime.now().isoformat(),
                    "error_message": error_message,
                },
            )
        except Exception as e:
            logger.debug(f"Failed to set failed status: {e}")

        logger.error(f"{library_id}: Processing failed: {error_message}")
