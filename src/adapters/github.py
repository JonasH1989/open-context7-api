"""GitHub adapter for repository access and library processing."""

import shutil
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.parse import urlparse

import git
import httpx
from loguru import logger
from starlette.concurrency import run_in_threadpool

from src.core.config import settings
from src.core.constants import DEFAULT_LIBRARY_TAG
from src.core.constants import GITHUB_API_TIMEOUT
from src.core.errors import ValidationError
from src.utils.common import md5_hash

GITHUB_API_BASE = "https://api.github.com"


class GitHubAdapter:
    """GitHub repository adapter for library extraction."""

    def __init__(self, url: str, token: str, org: str, project: str):
        """Initialize GitHub adapter for repository.

        Args:
            url: GitHub repository URL.
            token: GitHub token (PAT) for API and clone access.
            org: Owner (user or organization).
            project: Repository name.
        """
        self.url = url.rstrip("/")
        self.token = token
        self.org = org
        self.project = project
        self.branch = settings.GIT_DEFAULT_BRANCH
        self.tag = DEFAULT_LIBRARY_TAG
        self._project_info: dict[str, str] = {}

    @property
    def name(self) -> str:
        """Get readable name for this repository."""
        return f"/{self.org}/{self.project}"

    @property
    def id(self) -> str:
        """Get unique identifier for this repository."""
        return md5_hash(self.name)

    @property
    def title(self) -> str:
        """Get repository title."""
        return self._project_info.get("title", "Unknown")

    @property
    def description(self) -> str:
        """Get repository description."""
        return self._project_info.get("description", "No description")

    def _headers(self) -> dict[str, str]:
        """Build GitHub API request headers."""
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "open-context7-api",
        }

    async def validate_access(self) -> bool:
        """Validate GitHub repository access.

        Returns:
            True if access is valid, False otherwise.
        """
        return await run_in_threadpool(self._validate_access)

    def _validate_access(self) -> bool:
        """GitHub access validation and repo info cache."""
        try:
            response = httpx.get(
                f"{GITHUB_API_BASE}/repos/{self.org}/{self.project}",
                headers=self._headers(),
                timeout=GITHUB_API_TIMEOUT,
                follow_redirects=True,
            )
            if response.status_code != 200:
                logger.error(
                    f"Cannot access GitHub /{self.org}/{self.project}: "
                    f"HTTP {response.status_code}"
                )
                return False

            data = response.json()
            self._project_info = {
                "title": data.get("name") or self.project,
                "description": data.get("description")
                or f"Documentation for {self.name}",
            }
            default_branch = data.get("default_branch")
            if default_branch:
                self.branch = default_branch
            return True
        except Exception as e:
            logger.error(f"Cannot access GitHub /{self.org}/{self.project}: {e}")
            return False

    async def collect_files(self) -> dict[str, str]:
        """Collect all Markdown files from GitHub repo.

        Returns:
            Dict with file URLs as keys and contents as values.
        """
        return await run_in_threadpool(self._collect_files)

    def _collect_files(self) -> dict[str, str]:
        """Clone the repository and extract all markdown files."""
        if not self._project_info:
            self._validate_access()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / f"{self.org}_{self.project}"
            self._clone_repository(str(temp_path))
            return self._build_file_mapping(str(temp_path))

    def _clone_repository(self, local_path: str) -> None:
        """Clone GitHub repo to local directory.

        Args:
            local_path: Local directory path to clone to.

        Raises:
            Exception: If clone fails.
        """
        parsed = urlparse(self.url)
        token = quote(self.token, safe="")
        clone_url = (
            f"{parsed.scheme}://x-access-token:{token}@"
            f"{parsed.netloc}/{self.org}/{self.project}.git"
        )

        if Path(local_path).exists():
            shutil.rmtree(local_path)

        if self.tag != DEFAULT_LIBRARY_TAG:
            repo = git.Repo.clone_from(clone_url, local_path)
            repo.git.checkout(self.tag)
            logger.info(f"Checked out tag {self.tag}")
        else:
            git.Repo.clone_from(clone_url, local_path, depth=1)

    def _build_file_mapping(self, local_path: str) -> dict[str, str]:
        """Build file mapping from local directory.

        Args:
            local_path: Local directory path to scan.

        Returns:
            Dict with GitHub blob URLs as keys and contents as values.
        """
        files: dict[str, str] = {}
        path_obj = Path(local_path)
        ref_name = self.tag if self.tag != DEFAULT_LIBRARY_TAG else self.branch

        for md_file in path_obj.rglob("*.md"):
            try:
                relative_path = md_file.relative_to(path_obj)
                content = md_file.read_text(encoding="utf-8")
                file_url = f"{self.url}/blob/{ref_name}/{relative_path}"
                files[file_url] = content
            except Exception as e:
                logger.warning(f"Failed to read {md_file}: {e}")

        return files

    async def build_payload(self) -> dict[str, Any]:
        """Build payload for storage based on GitHub repo info.

        Returns:
            Provider-specific payload data including repo metadata.
        """
        return {
            "repo_url": self.url,
            "org": self.org,
            "project": self.project,
            "branch": self.branch,
            "tag": self.tag,
            "access_token": self.token,
            "last_commit_id": await self.get_latest_commit_id(),
        }

    async def get_latest_commit_id(self) -> str | None:
        """Get latest commit ID from repo.

        Returns:
            Latest commit SHA string, or None if it can't be fetched.
        """
        return await run_in_threadpool(self._get_latest_commit_id)

    def _get_latest_commit_id(self) -> str | None:
        """Fetch latest commit ID from GitHub API."""
        try:
            response = httpx.get(
                f"{GITHUB_API_BASE}/repos/{self.org}/{self.project}/commits",
                headers=self._headers(),
                params={"sha": self.branch, "per_page": 1},
                timeout=GITHUB_API_TIMEOUT,
                follow_redirects=True,
            )
            if response.status_code != 200:
                logger.error(
                    f"Failed to get latest commit for GitHub {self.name}: "
                    f"HTTP {response.status_code}"
                )
                return None
            commits = response.json()
            if commits:
                return commits[0].get("sha")
            return None
        except Exception as e:
            logger.error(f"Failed to get latest commit for GitHub {self.name}: {e}")
            return None

    async def get_tags(self) -> list[str]:
        """Get all tags for GitHub repo.

        Returns:
            List of tag names sorted in reverse order.
        """
        return await run_in_threadpool(self._get_tags)

    def _get_tags(self) -> list[str]:
        """Fetch all tags from GitHub API, sorted newest first."""
        try:
            response = httpx.get(
                f"{GITHUB_API_BASE}/repos/{self.org}/{self.project}/tags",
                headers=self._headers(),
                params={"per_page": 100},
                timeout=GITHUB_API_TIMEOUT,
                follow_redirects=True,
            )
            if response.status_code != 200:
                logger.error(
                    f"Failed to get tags for GitHub {self.name}: "
                    f"HTTP {response.status_code}"
                )
                return []
            tags = [tag.get("name", "") for tag in response.json()]
            return sorted(tags, reverse=True)
        except Exception as e:
            logger.error(f"Failed to get tags for GitHub {self.name}: {e}")
            return []

    @classmethod
    def from_library(
        cls,
        library: Any,  # LibraryDetail - avoid circular import
        tag: str | None = None,
    ) -> "GitHubAdapter":
        """Create GitHub adapter from library.

        Args:
            library: Library details with repo information.
            tag: Optional tag to set on the adapter.

        Returns:
            GitHubAdapter configured for the specified library and tag.

        Raises:
            ValidationError: If required repo info is missing.
        """
        if not all(
            [library.repo_url, library.access_token, library.org, library.project]
        ):
            raise ValidationError("Missing required repository information")

        adapter = cls(
            url=library.repo_url,
            token=library.access_token,
            org=library.org,
            project=library.project,
        )
        if tag:
            adapter.tag = tag
        return adapter
