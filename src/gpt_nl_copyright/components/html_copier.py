import re

from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep


class HtmlCopier(PipelineStep):
    name = "✂️ HTML Copier"
    type = "🖊️ - ANNOTA"

    _requires_dependencies = []

    def __init__(self):
        super().__init__()

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        """Iterates through each document and adds the html content (`text`) to the metadata because because
        the Trafiltura extractor will change the `text` to only contain extracted text, not the HTML.
        We copy the HTML content to the metadata so that it can be used later in the pipeline - but we do not
        process the HTML here yet. This is done AFTER filtering, so that we don't process documents that will
        be filtered out later on anyway.

        Args:
          data: DocumentsPipeline:
          rank: int:  (Default value = 0)
          world_size: int:  (Default value = 1)

        Returns:

        """
        for doc in data:
            # Compress the HTML a bit by removing whitespace between tags
            doc.metadata["html"] = re.sub(r">\s+<", "><", doc.text)
            yield doc
