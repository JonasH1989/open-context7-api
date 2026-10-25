"""Application constants."""

# Application
APP_NAME = "open-context7-api"

# Qdrant collection name for library document index
QDRANT_LIBRARIES_COLLECTION_NAME = "libraries"

# Default tag for library versions when no specific tag is provided
DEFAULT_LIBRARY_TAG = "latest"

# GitLab/GitHub API timeout in seconds
GITLAB_API_TIMEOUT = 10
GITHUB_API_TIMEOUT = 10

# Return codes
RETCODE_SUCCESS = 0
RETCODE_VALIDATION_ERROR = 1001
RETCODE_RESOURCE_NOT_FOUND = 1002
RETCODE_RESOURCE_ALREADY_EXISTS = 1003
RETCODE_CONFIGURATION_ERROR = 2001
RETCODE_CHAT_SERVICE_ERROR = 2002
RETCODE_EMBEDDING_SERVICE_ERROR = 2003
RETCODE_INTERNAL_ERROR = 9999
