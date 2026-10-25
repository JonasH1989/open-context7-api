"""Library API endpoints."""

from typing import Annotated

from fastapi import APIRouter
from fastapi import BackgroundTasks
from fastapi import Depends
from fastapi import Query
from fastapi import Request
from fastapi import Response

from src.adapters import create_adapter
from src.core.constants import DEFAULT_LIBRARY_TAG
from src.core.errors import ResourceAlreadyExistsError
from src.core.errors import ValidationError
from src.deps import get_library_id
from src.deps import get_service
from src.schemas.base import BaseResponse
from src.schemas.requests import ContentRequest
from src.schemas.requests import RepositoryRequest
from src.schemas.responses import LibraryDetail
from src.services.library import LibraryService
from src.utils.response import success_response


router = APIRouter()


@router.get("/search")
async def search(
    library_service: Annotated[LibraryService, Depends(get_service)],
    query: str | None = Query(None, description="Search query"),
    limit: int = Query(10, ge=1, le=100, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
):
    """## Search Libraries.

    Find libraries in your collection or list all available libraries.
    Uses Context7 API format.

    ### Query Parameters
    - **query**: Search text to find libraries (empty = show all)
    - **limit**: Max results to return (default: 35)
    - **offset**: Skip results for pagination (search ignores this)

    ### Returns
    JSON list with library info and metadata.
    """
    libraries = await library_service.search(query=query, limit=limit, offset=offset)

    results = []
    for lib in libraries:
        result = {
            "id": f"/{lib.org}/{lib.project}",
            "title": lib.title,
            "description": lib.description,
            "branch": lib.branch,
            "lastUpdateDate": lib.last_update_date,
            "state": lib.state,
            "totalTokens": lib.total_tokens,
            "versions": lib.tags,
            "libraryType": lib.library_type,
        }
        if lib.score is not None:
            result["trustScore"] = lib.score
        results.append(result)

    return {"results": results}


@router.post("/")
async def create(
    request: Request,
    request_data: RepositoryRequest,
    library_service: Annotated[LibraryService, Depends(get_service)],
    background_tasks: BackgroundTasks,
):
    """## Create Library from Repository.

    Convert your GitLab repository into searchable documentation.

    ### Request Body
    ```json
    {
        "repoUrl": "https://gitlab.company.com/myorg/myproject",
        "accessToken": "glpat-xxxxxxxxxxxxxxxxxxxx"
    }
    ```

    ### Returns
    Success status with tracking info.

    ### Success Response
    ```json
    {
        "data": {},
        "message": "Operation successful",
        "createdAt": "2025-08-06T10:30:00Z",
        "requestId": "xyz789-uvw012"
    }
    ```

    ### Error Responses
    - **400**: Bad request or no access
    - **409**: Library already exists

    ```json
    {
        "error": "ValidationError",
        "message": "Cannot access repository '/myorg/myproject'."
    }
    ```
    """
    # Create the adapter matching the repository host (GitHub or GitLab)
    provider = create_adapter(
        request_data.repo_url, request_data.access_token
    )

    # Validate access
    if not await provider.validate_access():
        raise ValidationError(f"Cannot access repository '{provider.name}'")

    # Check if library already exists
    if await library_service.exists(provider.id):
        raise ResourceAlreadyExistsError(f"Library '{provider.name}' already exists")

    # Start processing in background using new unified method
    background_tasks.add_task(library_service.create_from_git, provider)

    return success_response(request=request)


# Specific sub-path routes must be defined before the general route pattern


@router.post("/{org}/{project}/content")
async def create_from_content(
    org: str,
    project: str,
    request: Request,
    request_data: ContentRequest,
    library_service: Annotated[LibraryService, Depends(get_service)],
    library_id: Annotated[str, Depends(get_library_id)],
    background_tasks: BackgroundTasks,
):
    r"""## Create Library from Content.

    Create library from files sent by other systems.

    ### Parameters
    - **org**: Organization name (e.g., "myorg")
    - **project**: Project name (e.g., "myproject")

    ### Request Body
    ```json
    {
        "title": "My Library",
        "description": "Library description",
        "files": {
            "README.md": "# My Project\\nThis is a readme...",
            "src/main.py": "def main():\\n    print('hello')"
        }
    }
    ```

    ### Returns
    Success status showing creation started.

    ### Success Response
    ```json
    {
        "data": {},
        "message": "Library creation started",
        "createdAt": "2025-08-06T10:30:00Z",
        "requestId": "xyz789-uvw012"
    }
    ```

    ### Error Responses
    - **400**: Invalid request data
    - **409**: Library already exists
    """
    # Check if library already exists
    if await library_service.exists(library_id):
        raise ResourceAlreadyExistsError("Library already exists")

    # Start processing in background using unified method
    background_tasks.add_task(
        library_service.create,
        library_id,
        request_data.title,
        request_data.description,
        request_data.files,
        org,
        project,
    )

    return success_response(
        request=request,
        message="Library creation started",
    )


@router.get("/{org}/{project}/tags")
async def get_tags(
    request: Request,
    library_service: Annotated[LibraryService, Depends(get_service)],
    library_id: Annotated[str, Depends(get_library_id)],
) -> BaseResponse[list[str]]:
    """## Get Repository Tags.

    Show all tags for a GitLab repository.
    Only works with repo-based libraries.

    ### Parameters
    - **org**: Organization name
    - **project**: Project name

    ### Returns
    Tag list sorted by newest first.

    ### Error Responses
    - **404**: Library not found
    - **400**: Not a repo or no access
    """
    tags = await library_service.get_tags(library_id)
    return success_response(data=tags, request=request)


@router.post("/{org}/{project}/rebuild")
async def rebuild(
    request: Request,
    library_service: Annotated[LibraryService, Depends(get_service)],
    background_tasks: BackgroundTasks,
    library_id: Annotated[str, Depends(get_library_id)],
):
    """## Rebuild Library.

    Rebuild library by processing source data again.
    This will refresh all documents and search data.

    ### Parameters
    - **org**: Organization name (e.g., "mycompany")
    - **project**: Project name (e.g., "myproject")

    ### Returns
    HTTP 200 status when rebuild starts.

    ### Error Responses
    - **404**: Library not found
    - **400**: No changes or not a repo
    - **409**: Library is being processed
    """
    # Precheck rebuild conditions
    await library_service.precheck_rebuild(library_id)

    # Start rebuild in background
    background_tasks.add_task(library_service.rebuild, library_id)

    return success_response(request=request)


@router.post("/{org}/{project}/tags/{tag}")
async def add_tag(
    tag: str,
    request: Request,
    library_service: Annotated[LibraryService, Depends(get_service)],
    library_id: Annotated[str, Depends(get_library_id)],
    background_tasks: BackgroundTasks,
):
    """## Add Library Tag.

    Create a new tag for the library.
    Uses token saved when library was created.

    ### Parameters
    - **tag**: Tag name (e.g., "v1.0", "0.115.13")

    ### Returns
    Success status when tag creation starts.

    ### Error Responses
    - **404**: Library not found
    - **400**: Bad tag name or tag exists

    ### Note
    Uses the saved token from library creation.
    To change token, update the database.
    """
    # Validate before starting background task
    await library_service.precheck_add_tag(library_id, tag)

    # Start tag creation in background
    background_tasks.add_task(library_service.add_tag, library_id, tag)

    return success_response(request=request)


@router.get("/{org}/{project}/meta")
async def get_library_meta(
    request: Request,
    library_service: Annotated[LibraryService, Depends(get_service)],
    library_id: Annotated[str, Depends(get_library_id)],
) -> BaseResponse[LibraryDetail]:
    """## Get Library Metadata.

    Get detailed library metadata including status, repository info, and configuration.

    ### Parameters
    - **org**: Organization name (e.g., "myorg")
    - **project**: Project name (e.g., "myproject")

    ### Returns
    Complete library metadata and status information.

    ### Success Response
    ```json
    {
        "retcode": 0,
        "data": {
            "library_id": "abc123def456...",
            "status": "completed",
            "repo_url": "https://gitlab.company.com/myorg/myproject",
            "access_token": null,
            "org": "myorg",
            "project": "myproject",
            "branch": "main",
            "last_commit_id": "1a2b3c4d5e6f...",
            "tags": ["latest", "v1.0", "v2.0"]
        },
        "message": "Operation successful",
        "createdAt": "2025-09-09T12:00:00Z",
        "requestId": "uuid-1234-5678"
    }
    ```

    ### Error Responses
    - **retcode 1002**: Library not found

    ### Usage Examples
    ```bash
    # Get library metadata
    GET /api/v1/library/myorg/myproject/meta

    # Check processing status
    GET /api/v1/library/company/docs/meta
    ```
    """
    library_detail = await library_service.get_by_id(library_id)
    return success_response(data=library_detail, request=request)


# Handle both /{org}/{project} and /{org}/{project}/{tag}
@router.get("/{org}/{project}/{tag}")
@router.get("/{org}/{project}")
async def query(
    library_service: Annotated[LibraryService, Depends(get_service)],
    library_id: Annotated[str, Depends(get_library_id)],
    response: Response,
    tag: str = DEFAULT_LIBRARY_TAG,
    topic: str | None = Query(None, description="Search topic (empty = all docs)"),
    tokens: int = Query(
        10000, ge=100, le=50000, description="Max tokens for AI context"
    ),
):
    """## Get Library Documents.

    Get documents from a library with search and filtering.
    Uses Context7 API format.

    ### Parameters
    - **org**: Organization name (e.g., "fastapi")
    - **project**: Project name (e.g., "fastapi")
    - **tag**: Library tag (e.g., "0.115.13", "v1.0") - optional
    - **topic**: Search topic or question (empty = all documents)
    - **tokens**: Max tokens to return (100-50000)

    ### Usage Examples
    ```bash
    # Get all documents (latest tag)
    GET /fastapi/fastapi

    # Get documents for specific tag
    GET /fastapi/fastapi/0.115.13

    # Search for specific content
    GET /fastapi/fastapi?topic=authentication

    # Search in specific tag
    GET /fastapi/fastapi/v1.0?topic=header&tokens=100
    ```

    ### Returns
    Plain text with document snippets in Context7 format.
    """
    # AI-focused search with token filtering
    documents = await library_service.query(
        library_id=library_id,
        topic=topic or "comprehensive documentation overview",
        tokens=tokens,
        tag=tag,
    )

    # Always return plain text format matching Context7 API
    response.headers["Content-Type"] = "text/plain"

    if not documents:
        return Response(content="", media_type="text/plain")

    # Format each document as Context7-style snippet
    parts = []
    for i, doc in enumerate(documents):
        if i > 0:
            parts.append("----------------------------------------")

        parts.extend(
            [
                "",
                f"TITLE: {doc.title}",
                f"DESCRIPTION: {doc.description}",
                f"SOURCE: {doc.source}",
                "",
                f"LANGUAGE: {doc.language}",
                "CODE:",
                "```",
                doc.code,
                "```",
                "",
            ]
        )

    formatted_text = "\n".join(parts)
    return Response(content=formatted_text, media_type="text/plain")
