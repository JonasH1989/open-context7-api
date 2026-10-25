"""External service adapters."""

from urllib.parse import urlparse

from src.adapters.github import GitHubAdapter
from src.adapters.gitlab import GitLabAdapter
from src.core.errors import ValidationError

ProviderAdapter = GitLabAdapter | GitHubAdapter

GITHUB_HOSTS = ("github.com", "www.github.com")


def is_github_url(url: str) -> bool:
    """Check whether a repository URL points to GitHub.

    Args:
        url: Repository URL.

    Returns:
        True if the host is GitHub, False otherwise.
    """
    host = (urlparse(url).netloc or "").lower()
    return host in GITHUB_HOSTS or host.endswith(".github.com")


def parse_github_repo_url(url: str) -> tuple[str, str]:
    """Parse a GitHub URL into owner and repository.

    Takes the first two path segments, so deeper GitHub URLs
    (for example /owner/repo/tree/main) resolve to the repository.

    Args:
        url: GitHub repository URL.

    Returns:
        Tuple of (owner, repository).

    Raises:
        ValueError: If the URL does not contain owner and repository.
    """
    parsed = urlparse(url)
    parts = [p for p in parsed.path.strip("/").split("/") if p]
    if len(parts) < 2:
        raise ValueError("Invalid repository URL format")

    org, project = parts[0], parts[1].removesuffix(".git")
    if not org or not project:
        raise ValueError("Organization and project names cannot be empty")
    return org, project


def create_adapter(url: str, token: str) -> ProviderAdapter:
    """Create the repository adapter matching the given URL.

    GitHub URLs get the GitHub adapter, everything else the GitLab adapter.

    Args:
        url: Repository URL.
        token: Access token for the repository host.

    Returns:
        Adapter instance for the repository.

    Raises:
        ValidationError: If the URL cannot be parsed.
    """
    if is_github_url(url):
        org, project = parse_github_repo_url(url)
        return GitHubAdapter(url, token, org, project)

    from src.utils.common import parse_repo_url

    try:
        org, project = parse_repo_url(url)
    except ValueError as e:
        raise ValidationError(str(e)) from e
    return GitLabAdapter(url, token, org, project)


def adapter_from_library(library: Any, tag: str | None = None) -> ProviderAdapter:
    """Create the adapter matching the library's repository URL.

    Args:
        library: Library details with repo information.
        tag: Optional tag to set on the adapter.

    Returns:
        Adapter instance for the library.

    Raises:
        ValidationError: If required repo info is missing.
    """
    if not all([library.repo_url, library.access_token]):
        raise ValidationError("Missing required repository information")

    adapter = create_adapter(library.repo_url, library.access_token)
    if tag:
        adapter.tag = tag
    return adapter
