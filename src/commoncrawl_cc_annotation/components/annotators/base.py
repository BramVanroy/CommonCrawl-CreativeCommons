from abc import abstractmethod
from typing import Iterator

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


class BaseBatchAnnotator(BaseAnnotator):
    @abstractmethod
    def __init__(self, batch_size: int = 2):
        super().__init__()
        self.batch_size = batch_size

    @abstractmethod
    def annotate(self, doc: list[Document]) -> Iterator[Document]:
        """Batch annotate batch_size documents with additional or changed information in the metadata.
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
        batch = []
        for doc in data:
            self.stat_update(StatHints.total)
            batch.append(doc)

            if len(batch) == self.batch_size:
                with self.track_time():
                    yield from self.annotate(batch)
                batch = []

        if batch:
            with self.track_time():
                yield from self.annotate(batch)

