from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter


class EmptyTextFilter(BaseFilter):
    """Filter out documents that have no text, i.e. empty stripped text property"""

    name = "🗅 Empty Text Filter"

    def __init__(self):
        super().__init__()

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        """Filter out documents that have no text

        Args:
          doc: Document: input Document

        Returns:
          bool | tuple[bool, str]: whether to keep the document or not, and the reason if not

        """
        if not doc.text.strip():
            return False, "empty_text_property"

        return True


class CCTextFilter(BaseFilter):
    """Filter out documents that do not contain 'creativecommons.org' in their text"""

    name = "©️ creativecommons.org Text Filter"

    def __init__(self):
        super().__init__()

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        """Filter out documents that do not contain 'creativecommons.org' in their text

        Args:
          doc: Document: input Document

        Returns:
          bool | tuple[bool, str]: whether to keep the document or not, and the reason if not

        """
        if "creativecommons.org" not in doc.text.lower():
            return False, "'creativecommons.org' missing"

        return True
