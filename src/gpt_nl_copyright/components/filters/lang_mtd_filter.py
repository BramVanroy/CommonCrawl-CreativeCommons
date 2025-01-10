from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter


class LanguageMetadataFilter(BaseFilter):
    """Filter out documents that are not written in the expected language"""

    name = "💬 Language Metadata Filter"

    def __init__(self, lang: str):
        super().__init__()
        self.lang = lang

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        """Filter out documents that are not written in the expected language

        Args:
          doc: Document: input Document

        Returns:
          bool | tuple[bool, str]: whether to keep the document or not, and the reason if not

        """
        if "language" in doc.metadata and doc.metadata["language"] != self.lang:
            return False, "incorrect_language"

        if "language" not in doc.metadata:
            return False, "no_language_provided"

        return True
