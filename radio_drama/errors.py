from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SourceLocation:
    source: str | None
    line: int | None
    column: int | None

    def format(self) -> str:
        if self.source is None and self.line is None and self.column is None:
            return "<unknown>"
        source = self.source or "<string>"
        if self.line is None:
            return source
        if self.column is None:
            return f"{source}:{self.line}"
        return f"{source}:{self.line}:{self.column}"


class DocumentError(ValueError):
    def __init__(self, message: str, *, node=None, location: SourceLocation | None = None) -> None:
        self.message = message
        self.node = node
        self.location = location or getattr(node, "location", None)
        super().__init__(self.__str__())

    def __str__(self) -> str:
        node_label = ""
        if self.node is not None and getattr(self.node, "display_name", None):
            node_label = f" {self.node.display_name}"
        if self.location is None:
            return f"{self.message}{node_label}"
        return f"{self.location.format()}: {self.message}{node_label}"
