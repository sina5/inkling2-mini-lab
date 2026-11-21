"""Project exceptions."""


class InklabError(Exception):
    """Base class for errors in this project."""


class ServerError(InklabError):
    """The llama.cpp server failed to start or returned an error."""


class DatasetError(InklabError):
    """A dataset file is missing or malformed."""
