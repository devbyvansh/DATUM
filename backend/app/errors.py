from fastapi import HTTPException, status


class DatumError(Exception):
    def __init__(self, message: str, code: str = "processing_error") -> None:
        super().__init__(message)
        self.message = message
        self.code = code


def not_found(resource: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"code": "not_found", "message": f"{resource} was not found."},
    )
