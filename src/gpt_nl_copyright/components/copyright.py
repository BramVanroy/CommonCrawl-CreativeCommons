from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep

from gpt_nl_copyright.copyright_finder import get_license_from_html


class CopyrightAnnotator(PipelineStep):
    name = "©️ Copyright Annotator"
    type = "🖊️ - ANNOTA"

    _requires_dependencies = ["bs4"]

    def __init__(self):
        super().__init__()

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        """Iterates through each document and finds

        Args:
          data: DocumentsPipeline:
          rank: int:  (Default value = 0)
          world_size: int:  (Default value = 1)

        Returns:

        """
        for doc in data:
            html = doc.metadata["html"]
            license_type, license_path = get_license_from_html(html)
            doc.metadata["license_type"] = license_type
            doc.metadata["license_path"] = license_path
            doc.metadata.pop("html")
            yield doc
