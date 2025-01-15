from abc import abstractmethod

from datatrove.data import Document, DocumentsPipeline
from datatrove.pipeline.base import PipelineStep
from datatrove.utils.typeshelper import StatHints


class BaseAnnotator(PipelineStep):
    type = "🖊️ - ANNOTA"

    @abstractmethod
    def __init__(self):
        super().__init__()

    @abstractmethod
    def annotate(self, doc: Document) -> Document:
        """Annotate the document with additional or changed information in the metadata.
        This method should be overridden by subclasses.

        Args:
            doc: Document: the document to annotate

        Returns:
            Document: the annotated document
        """
        pass

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        """Iterates through each document and calls the `annotate` method

        Args:
          data: DocumentsPipeline:
          rank: int:  (Default value = 0)
          world_size: int:  (Default value = 1)

        Returns:

        """
        for doc in data:
            self.stat_update(StatHints.total)
            with self.track_time():
                yield self.annotate(doc)
