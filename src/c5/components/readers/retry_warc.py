import time
from typing import Callable, Literal

from datatrove.io import DataFileLike, DataFolderLike
from datatrove.pipeline.readers.warc import WarcReader, process_record
from datatrove.utils.logging import logger


class RetryWarcReader(WarcReader):
    """Read data from WARC files.
        Will read each record as a separate document.

        Adapted: will retry reading a file with exponential backoff if an error occurs.

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

    name = "🕷 Warc"
    _requires_dependencies = ["warcio", ("cchardet", "faust-cchardet"), ("magic", "python-magic")]

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
        max_num_retries: int = 1000,
    ):
        super().__init__(
            data_folder,
            paths_file,
            compression,
            limit,
            skip,
            file_progress,
            doc_progress,
            adapter,
            text_key,
            id_key,
            default_metadata,
            recursive,
            glob_pattern,
            shuffle_files,
        )
        self.max_num_retries = max_num_retries

    def read_file(self, filepath: str):
        from warcio.archiveiterator import ArchiveIterator

        last_emitted_index = -1
        num_retries = self.max_num_retries
        while True:
            try:
                with self.data_folder.open(filepath, "rb", compression=self.compression) as f:
                    start_index = last_emitted_index + 1
                    if start_index > 0:
                        logger.info(f"Resuming {filepath} from record index {start_index}")
                    for ri, record in enumerate(ArchiveIterator(f)):
                        if ri < start_index:
                            continue

                        with self.track_time():
                            extracted_data = process_record(record)
                            if not extracted_data:
                                continue
                            document = self.get_document_from_dict(extracted_data, filepath, ri)
                            if not document:
                                continue
                        yield document
                        last_emitted_index = ri
                        num_retries = self.max_num_retries

            except Exception as exc:
                num_retries -= 1
                logger.warning(f"Error reading {filepath} at record {last_emitted_index + 1}: {exc}")

                if not num_retries:
                    logger.error(f"Max retries reached for {filepath}...")
                    raise exc
                else:
                    logger.info(f"Retrying {filepath} in 2 seconds... ({num_retries} retries left)")
                    # Common Crawl says to wait at least 1 second between requests: https://status.commoncrawl.org/
                    time.sleep(2)
            else:
                break
        logger.info(f"Finished reading {filepath}")
