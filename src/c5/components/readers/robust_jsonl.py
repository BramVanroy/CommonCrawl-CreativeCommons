import random
from pathlib import Path
from typing import Callable, Literal

from datatrove.data import DocumentsPipeline
from datatrove.io import DataFileLike, DataFolderLike, get_shard_from_paths_file
from datatrove.pipeline.readers import JsonlReader
from datatrove.utils.logging import logger


class RobustJsonlReader(JsonlReader):
    """Read data from JSONL files.
        Will read each line as a separate document.

    Args:
        data_folder: a str, tuple or DataFolder object representing a path/filesystem
        paths_file: optionally provide a file with one path per line (without the `data_folder` prefix) to read.
        compression: the compression to use (default: "infer")
        limit: limit the number of documents to read. Useful for debugging
        skip: skip the first n rows
        file_progress: show progress bar for files
        doc_progress: show progress bar for documents
        adapter: function to adapt the data dict from the source to a Document.
            Takes as input: (self, data: dict, path: str, id_in_file: int | str)
                self allows access to self.text_key and self.id_key
            Returns: a dict with at least a "text" and "id" keys
        text_key: the key containing the text data (default: "text").
        id_key: the key containing the id for each sample (default: "id").
        default_metadata: a dictionary with any data that should be added to all samples' metadata
        recursive: whether to search files recursively. Ignored if paths_file is provided
        glob_pattern: pattern that all files must match exactly to be included (relative to data_folder). Ignored if paths_file is provided
        shuffle_files: shuffle the files within the returned shard. Mostly used for data viz. purposes, do not use with dedup blocks
    """

    name = "🐿 Jsonl"
    _requires_dependencies = ["orjson"]

    def __init__(
        self,
        data_folder: DataFolderLike,
        paths_file: DataFileLike | None = None,
        compression: Literal["infer", "gzip", "zstd"] | None = "infer",
        limit: int = -1,
        skip: int = 0,
        file_progress: bool = False,
        doc_progress: bool = False,
        adapter: Callable = None,
        text_key: str = "text",
        id_key: str = "id",
        default_metadata: dict = None,
        recursive: bool = True,
        glob_pattern: str | None = None,
        shuffle_files: bool = False,
    ):
        self.pdir = Path(data_folder)

        super().__init__(
            data_folder=data_folder,
            paths_file=paths_file,
            compression=compression,
            limit=limit,
            skip=skip,
            file_progress=file_progress,
            doc_progress=doc_progress,
            adapter=adapter,
            text_key=text_key,
            id_key=id_key,
            default_metadata=default_metadata,
            recursive=recursive,
            glob_pattern=glob_pattern,
            shuffle_files=shuffle_files,
        )

    def run(self, data: DocumentsPipeline = None, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        """
        Will get this rank's shard and sequentially read each file in the shard, yielding Document.
        Args:
            data: any existing data from previous pipeline stages
            rank: rank of the current task
            world_size: total number of tasks

        Returns:

        """

        if data:
            yield from data

        # Clean up if the starting directory was empty
        if not self.pdir.exists() or len(list(self.pdir.iterdir())) == 0:
            return

        files_shard = (
            self.data_folder.get_shard(rank, world_size, recursive=self.recursive, glob_pattern=self.glob_pattern)
            if not self.paths_file
            else list(get_shard_from_paths_file(self.paths_file, rank, world_size))
        )
        if files_shard is None:
            raise RuntimeError(f"No files found on {self.data_folder.path}!")
        elif len(files_shard) == 0:
            # otherwise just a warning
            logger.warning(f"No files found on {self.data_folder.path} for {rank=}")

        if self.shuffle_files:
            random.shuffle(files_shard)
        for doc in self.read_files_shard(files_shard):
            self.update_doc_stats(doc)
            yield doc
