"""Library service for document lifecycle management."""

import traceback
from datetime import datetime
from typing import Any

from loguru import logger
from qdrant_client import AsyncQdrantClient

from src.adapters.ai import AIProvider
from src.adapters import ProviderAdapter
from src.adapters import adapter_from_library
from src.core.constants import DEFAULT_LIBRARY_TAG
from src.core.constants import QDRANT_LIBRARIES_COLLECTION_NAME
from src.core.enums import LibraryStatus
from src.core.enums import LibraryType
from src.core.errors import ResourceNotFoundError
from src.core.errors import ValidationError
from src.schemas.internal import TokenFilterResult
from src.schemas.responses import DocumentResponse
from src.schemas.responses import LibraryDetail
from src.schemas.responses import LibrarySearchItem
from src.services.processor import Processor
from src.services.storage import Storage


class LibraryService:
    """Library service - manages document library lifecycle."""

    def __init__(
        self, ai_provider: AIProvider, qdrant_client: AsyncQdrantClient | None = None
    ) -> None:
        """Init service with AI provider.

        Args:
            ai_provider: AI provider for embeddings and processing.
            qdrant_client: Optional Qdrant client instance.
        """
        self.ai = ai_provider
        self.storage = Storage(qdrant_client)
        self.processor = Processor(ai_provider)

    async def create(
        self,
        library_id: str,
        title: str,
        description: str,
        files: dict[str, str],
        org: str,
        project: str,
    ) -> None:
        """Create new library from files.

        Process files to build a searchable library.

        Args:
            library_id: Unique ID for the library.
            title: Library title.
            description: Library description.
            files: Dict mapping file paths to content.
            org: Organization name.
            project: Project name.

        Raises:
            Exception: If processing fails.
        """
        logger.info(f"{library_id}: Creating library '{title}'")

        try:
            # Generate library-level embedding
            library_embedding = await self.ai.embedding(f"{title} {description}")

            # Initialize storage with org/project info
            await self.storage.initialize(
                library_id,
                title,
                description,
                library_embedding.embedding,
                org,
                project,
                library_type=LibraryType.API,
            )

            # Process files
            logger.info(f"{library_id}: Found {len(files)} files")
            snippets = await self.processor.process(files)

            # Save to vector database
            await self.storage.save_snippets(snippets, library_id)

            # Complete with statistics
            total_tokens = sum(s.get("tokens", 0) for s in snippets)
            await self.storage.complete_library(library_id, total_tokens)

            logger.info(
                f"{library_id}: Library creation completed - {len(snippets)} snippets, "
                f"{total_tokens} tokens"
            )

        except Exception as e:
            error_traceback = traceback.format_exc()
            error_message = f"{str(e)}\n\nTraceback:\n{error_traceback}"
            await self.storage.cleanup_failed_library(library_id, error_message)
            logger.error(f"Library {library_id} creation failed: {e}")
            logger.debug(f"Full traceback: {error_traceback}")
            raise

    async def create_from_git(self, provider: ProviderAdapter) -> None:
        """Create library from GitLab repository.

        Get files from GitLab and build library.

        Args:
            provider: GitLab adapter instance.

        Raises:
            ValidationError: If cannot access repository.
            Exception: If processing fails.
        """
        logger.info(f"{provider.id}: Creating library '{provider.title}' from Git")

        try:
            # Generate library-level embedding
            library_embedding = await self.ai.embedding(
                f"{provider.title} {provider.description}"
            )

            # Get Git info from provider
            git_info = await provider.build_payload()

            # Initialize storage with Git info
            await self.storage.initialize(
                provider.id,
                provider.title,
                provider.description,
                library_embedding.embedding,
                provider.org,
                provider.project,
                git_info,
                LibraryType.GIT,
            )

            # Collect and process files
            files = await provider.collect_files()
            logger.info(f"{provider.id}: Found {len(files)} files")
            snippets = await self.processor.process(files)

            # Save to vector database
            await self.storage.save_snippets(snippets, provider.id)

            # Complete with statistics
            total_tokens = sum(s.get("tokens", 0) for s in snippets)
            await self.storage.complete(provider, total_tokens)

            logger.info(
                f"{provider.id}: Created {len(snippets)} snippets, "
                f"{total_tokens} tokens"
            )

        except Exception as e:
            error_traceback = traceback.format_exc()
            error_message = f"{str(e)}\n\nTraceback:\n{error_traceback}"
            await self.storage.cleanup_failed_library(provider.id, error_message)
            logger.error(f"Library {provider.id} creation failed: {e}")
            logger.debug(f"Full traceback: {error_traceback}")
            raise

    async def query(
        self,
        library_id: str,
        topic: str,
        tokens: int = 10000,
        tag: str = DEFAULT_LIBRARY_TAG,
    ) -> list[DocumentResponse]:
        """Search library with AI semantic search.

        Use vector search to find docs, filter by token limit.

        Args:
            library_id: Library ID to search.
            topic: Search topic or question.
            tokens: Max tokens to return (default: 10000).
            tag: Library tag to search (default: latest).

        Returns:
            List of matching docs with scores.

        Raises:
            ValidationError: If tag not found.
        """
        search_description = f"'{topic}' in {library_id}"
        if tag != DEFAULT_LIBRARY_TAG:
            search_description += f" (tag: {tag})"
        logger.info(f"Searching {search_description}")

        # Validate tag exists if not using default
        if tag != DEFAULT_LIBRARY_TAG:
            library_detail = await self.storage.get_by_id(library_id)
            if tag not in library_detail.tags:
                library_name = f"/{library_detail.org}/{library_detail.project}"
                raise ValidationError(
                    f"Tag '{tag}' not found for library {library_name}"
                )

        try:
            # Generate embedding and perform search
            embedding_response = await self.ai.embedding(topic)
            results = await self.storage.query(
                library_id, embedding_response.embedding, tag=tag
            )

            # Filter results by token limit
            filter_result = self._apply_token_limit(results, tokens)

            logger.info(
                f"Found {len(filter_result.documents)} documents "
                f"({filter_result.total_tokens} tokens) for {search_description}"
            )

            # Convert to DocumentResponse objects
            return [
                DocumentResponse(
                    title=doc.get("title", ""),
                    description=doc.get("description", ""),
                    source=doc.get("source", ""),
                    language=doc.get("language", "text"),
                    code=doc.get("code", ""),
                    tokens=doc.get("tokens", 0),
                    score=doc.get("score"),
                )
                for doc in filter_result.documents
            ]

        except Exception as e:
            logger.error(f"Search failed for {search_description}: {e}")
            return []

    async def exists(self, library_id: str) -> bool:
        """Check if library exists in the database.

        Args:
            library_id: Unique ID of the library to check.

        Returns:
            True if library exists, False otherwise.
        """
        return await self.storage.qdrant.collection_exists(library_id)

    async def is_processing(self, library_id: str) -> bool:
        """Check if library is currently being processed.

        Args:
            library_id: Unique ID of the library to check.

        Returns:
            True if library is being processed, False otherwise.
        """
        try:
            library = await self.storage.get_by_id(library_id)
            return library.status == LibraryStatus.PROCESSING
        except ResourceNotFoundError:
            return False

    async def get_by_id(self, library_id: str) -> LibraryDetail:
        """Get library details by ID.

        Args:
            library_id: Unique ID of the library to get.

        Returns:
            Library details object.

        Raises:
            ResourceNotFoundError: If library does not exist.
        """
        return await self.storage.get_by_id(library_id)

    async def get_tags(self, library_id: str) -> list[str]:
        """Get tags for repository library.

        Args:
            library_id: Repository library ID.

        Returns:
            List of tag names.

        Raises:
            ValidationError: If not a repo or no access.
        """
        library = await self.storage.get_by_id(library_id)

        # Check if it's a Git repository
        if not library.repo_url:
            raise ValidationError(
                "Cannot get tags: This library was not created from a Git repository"
            )

        provider = adapter_from_library(library)

        if not await provider.validate_access():
            raise ValidationError(f"Cannot access repository '{provider.name}'")

        return await provider.get_tags()

    async def precheck_add_tag(self, library_id: str, tag: str) -> None:
        """Check if tag can be added to library.

        Verify library exists, is a repo, tag is new, and has access.

        Args:
            library_id: Library ID to add tag to.
            tag: Tag name (e.g., "v1.0", "0.115.13").

        Raises:
            ResourceNotFoundError: If library not found.
            ValidationError: If tag cannot be added.
        """
        # Get library and validate
        library_detail = await self.storage.get_by_id(library_id)

        # Check if tag already exists
        if tag in library_detail.tags:
            raise ValidationError(f"Tag {tag} already exists")

        # Check if already processing
        if await self.is_processing(library_id):
            raise ValidationError("Library is currently being processed")

        # Validate repository access
        provider = adapter_from_library(library_detail, tag)
        if not await provider.validate_access():
            raise ValidationError(f"Cannot access repository '{provider.name}'")

        # Check if tag exists in the repository
        available_tags = await provider.get_tags()
        if tag not in available_tags:
            raise ValidationError(
                f"Tag '{tag}' does not exist in repository '{provider.name}'"
            )

    async def add_tag(self, library_id: str, tag: str) -> None:
        """Add tag to library.

        Process files for new tag. Call after precheck_add_tag validation.

        Args:
            library_id: Library ID to add tag to.
            tag: Tag name (e.g., "v1.0", "0.115.13").
        """
        logger.info(f"{library_id}: Adding tag {tag}")

        # Get library and create provider
        library_detail = await self.storage.get_by_id(library_id)
        provider = adapter_from_library(library_detail, tag)

        # Process and store tag files
        files = await provider.collect_files()
        snippets = await self.processor.process(files)

        # Set correct tag for all snippets
        for snippet in snippets:
            snippet["tag"] = tag

        await self.storage.save_snippets(snippets, library_id)

        # Add tag to index
        await self.storage.add_tag_to_index(library_id, tag)

        logger.info(
            f"{library_id}: Tag {tag} added successfully - {len(snippets)} documents"
        )

    async def precheck_rebuild(self, library_id: str) -> None:
        """Check if library can and should be rebuilt.

        Check library is a repo with changes since last build.

        Args:
            library_id: Library ID to check.

        Raises:
            ValidationError: If library cannot or should not be rebuilt.
        """
        library = await self.storage.get_by_id(library_id)

        # Check if it's a Git repository
        if not library.repo_url:
            raise ValidationError(
                "Cannot rebuild: This library was not created from a Git repository"
            )

        # Check if already processing
        if await self.is_processing(library_id):
            raise ValidationError("Library is currently being processed")

        # Create GitLab provider
        provider = adapter_from_library(library)

        # Validate access
        if not await provider.validate_access():
            raise ValidationError(f"Cannot access repository '{provider.name}'")

        # Compare commit IDs
        current_commit = await provider.get_latest_commit_id()

        if current_commit == library.last_commit_id:
            raise ValidationError(
                "No changes detected since last build. Rebuild not needed."
            )

    async def rebuild(self, library_id: str) -> None:
        """Rebuild library from original source.

        Rebuilds only the latest tag of repository library.
        Clears existing data and processes library again.

        Args:
            library_id: Unique ID of the library to rebuild.

        Raises:
            ValidationError: If rebuild requirements are not met.
            Exception: If rebuild process fails.
        """
        logger.info(f"{library_id}: Starting rebuild process")

        try:
            # Immediately set status to PROCESSING to prevent concurrent rebuilds
            await self.storage.qdrant.set_payload(
                collection_name=QDRANT_LIBRARIES_COLLECTION_NAME,
                points=[library_id],
                payload={"state": LibraryStatus.PROCESSING},
            )

            # Clear latest tag data
            await self.storage.remove_tag(library_id, DEFAULT_LIBRARY_TAG)
            logger.info(f"{library_id}: Cleared latest tag data")

            # Create GitLab provider
            library = await self.storage.get_by_id(library_id)
            provider = adapter_from_library(library)

            await self._do_processing(provider)

            logger.info(f"{library_id}: Rebuild completed successfully")

        except Exception as e:
            # Set status to FAILED on error
            await self.storage.qdrant.set_payload(
                collection_name=QDRANT_LIBRARIES_COLLECTION_NAME,
                points=[library_id],
                payload={
                    "state": LibraryStatus.FAILED,
                    "error_message": str(e),
                    "last_update_date": datetime.now().isoformat(),
                },
            )
            logger.error(f"Library {library_id} rebuild failed: {e}")
            raise

    async def _do_processing(self, provider: ProviderAdapter) -> None:
        """Execute library processing workflow.

        Legacy method for backward compatibility. New code should use create() directly.

        Args:
            provider: Library provider instance.

        Raises:
            Exception: If processing workflow fails.
        """
        # Collect files from provider
        files = await provider.collect_files()
        logger.info(f"{provider.id}: Found {len(files)} files")

        # Process content with AI
        snippets = await self.processor.process(files)

        # Save to vector database
        await self.storage.save_snippets(snippets, provider.id)

        # Complete with statistics
        total_tokens = sum(s.get("tokens", 0) for s in snippets)
        await self.storage.complete(provider, total_tokens)

        logger.info(
            f"{provider.id}: Processing completed - {len(snippets)} snippets, "
            f"{total_tokens} tokens"
        )

    @staticmethod
    def _apply_token_limit(
        documents: list[dict[str, Any]], token_limit: int
    ) -> TokenFilterResult:
        """Filter docs by total token limit.

        Args:
            documents: List of docs to filter.
            token_limit: Max total tokens allowed.

        Returns:
            Filter result with docs and total tokens.
        """
        filtered_docs = []
        total_tokens = 0

        for doc in documents:
            doc_tokens = doc.get("tokens", 0)
            if total_tokens + doc_tokens <= token_limit:
                filtered_docs.append(doc)
                total_tokens += doc_tokens
            else:
                break

        return TokenFilterResult(documents=filtered_docs, total_tokens=total_tokens)

    async def search(
        self,
        query: str | None = None,
        limit: int = 35,
        offset: int = 0,
    ) -> list[LibrarySearchItem]:
        """Search libraries (compatible with Context7 API).

        Args:
            query: Search term for libraries (None = list all).
            limit: Maximum results (default: 35 like Context7).
            offset: Skip results (only for listing, ignored for search).

        Returns:
            List of library data dictionaries.
        """
        try:
            query_vector = None
            if query:
                embedding = await self.ai.embedding(query)
                query_vector = embedding.embedding

            return await self.storage.search(
                query_vector=query_vector, limit=limit, offset=offset
            )
        except Exception as e:
            logger.error(f"Failed to search libraries: {e}")
            return []
