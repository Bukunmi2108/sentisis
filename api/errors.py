class SentisisError(Exception):
    """A failure with a caller-facing code, an HTTP status, and whether retrying could help."""

    def __init__(
        self, code: str, message: str, *, status_code: int = 500, retryable: bool = False
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.retryable = retryable
