from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter


class CopyrightFilter(BaseFilter):
    """Filter out documents that do not contain CreativeCommons copyright information"""

    name = "🛡️ Copyright Filter"

    def __init__(self):
        super().__init__()

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        """Filter out documents that do not contain CreativeCommons copyright information

        Args:
          doc: Document: input Document

        Returns:
          bool | tuple[bool, str]: whether to keep the document or not, and the reason if not

        """
        if "license_parse_error" in doc.metadata and doc.metadata["license_parse_error"]:
            return False, "license_parse_error"

        if "license_abbr" not in doc.metadata or not doc.metadata["license_abbr"]:
            return False, "no_explicit_license"

        return True
