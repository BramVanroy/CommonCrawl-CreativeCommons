from datatrove.data import Document

from c5.components.annotators.base import BaseAnnotator


class HtmlCopier(BaseAnnotator):
    name = "🗐 HTML Copier"

    def __init__(self):
        super().__init__()

    def annotate(self, doc: Document) -> Document:
        doc.metadata["html"] = doc.text

        return doc
