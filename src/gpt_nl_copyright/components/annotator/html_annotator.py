import re

from datatrove.data import Document

from gpt_nl_copyright.components.annotator.base import BaseAnnotator


class HtmlCopier(BaseAnnotator):
    name = "✂️ HTML Copier"

    _requires_dependencies = []

    def __init__(self):
        super().__init__()

    def annotate(self, doc: Document) -> Document:
        """Copy the HTML content to the metadata because the Trafiltura extractor will
        change the `text` to only contain extracted text, not the HTML. We copy the HTML
        content to the metadata so that it can be used later in the pipeline - but we do not
        process the HTML here yet. This is done AFTER filtering, so that we don't process
        documents that will be filtered out later on anyway.

        Args:
            doc: Document: the document to annotate

        Returns:
            Document: the annotated document
        """
        doc.metadata["html"] = re.sub(r">\s+<", "><", doc.text)
        return doc
