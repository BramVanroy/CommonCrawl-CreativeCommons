import re

from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep


class ColumnFilter(PipelineStep):
    name = "✂️ Column Filter"
    type = "🖊️ - ANNOTA"

    _requires_dependencies = []

    def __init__(self, output_text: bool = True, output_html: bool = False):
        super().__init__()
        self.output_text = output_text
        self.output_html = output_html

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        """Filter out columns from the documents
        Args:
          data: DocumentsPipeline:
          rank: int:  (Default value = 0)
          world_size: int:  (Default value = 1)

        Returns:

        """
        for doc in data:
            if not self.output_html:
                doc.metadata.pop("html", None)

            if not self.output_text:
                delattr(doc, "text")

            yield doc
