from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter


class LicenseFilter(BaseFilter):
    """Filter out documents that do not contain CreativeCommons license information"""

    name = "🛡️ License Filter"

    def __init__(self):
        super().__init__()

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        """Filter out documents that do not contain CreativeCommons copyright information

        Args:
          doc: Document: input Document

        Returns:
          bool | tuple[bool, str]: whether to keep the document or not, and the reason if not

        """
        if doc.metadata.get("license_parse_error", False):
            return False, "license_parse_error"

        if not doc.metadata.get("license_abbr", False):
            return False, "no_explicit_license"

        return True
