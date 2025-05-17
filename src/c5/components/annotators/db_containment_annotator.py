from collections import defaultdict
from typing import Iterator

from datatrove.data import Document

from ...utils import extract_uuid
from .base import BaseBatchAnnotator


class FWDBContainmentAnnotator(BaseBatchAnnotator):
    name = "🗄️ FW Database Containment Annotator"

    _requires_dependencies = ["duckdb"]

    def __init__(
        self,
        fw_duckdb_path: str | None,
        fw2_duckdb_templ_path: str | None,
        added_key: str,
        ignore_duckdb_for: list[str] | None = None,
        fw_eng_tag: str = "eng_Latn",  # Depends on the language detector
        batch_size: int = 128,
        overwrite_with_none: bool = False,
    ):
        super().__init__(batch_size=batch_size)

        if not overwrite_with_none:
            if not fw2_duckdb_templ_path or "{language}" not in fw2_duckdb_templ_path:
                raise ValueError("The fw2_duckdb_templ_path must contain the placeholder '{language}'")
        self.fw_duckdb_path = fw_duckdb_path
        self.fw2_duckdb_templ_path = fw2_duckdb_templ_path
        self.added_key = added_key
        self.ignore_duckdb_for = ignore_duckdb_for or []
        self.fw_eng_tag = fw_eng_tag

        self.cons = {}
        self.overwrite_with_none = overwrite_with_none

    def annotate(self, docs: list[Document]) -> Iterator[Document]:
        if self.overwrite_with_none:
            for doc in docs:
                doc.metadata[self.added_key] = None
                yield doc
            return
        else:
            import duckdb

            # Group documents by language to avoid re-opening the same database multiple times
            grouped_docs = defaultdict(list)
            for doc in docs:
                language = doc.metadata["language"]
                language_script = doc.metadata["language_script"]
                full_lang = f"{language}_{language_script}"
                grouped_docs[full_lang].append(doc)

            for full_lang, doc_group in grouped_docs.items():
                # Set the added key to None for languages that should be ignored
                if full_lang in self.ignore_duckdb_for:
                    for doc in doc_group:
                        doc.metadata[self.added_key] = None
                        yield doc
                    continue

                # Open connection to DuckDB database for this language
                if full_lang not in self.cons:
                    if full_lang == self.fw_eng_tag:
                        duckdb_path = self.fw_duckdb_path
                    else:
                        duckdb_path = self.fw2_duckdb_templ_path.format(language=full_lang)
                    con = duckdb.connect(duckdb_path, read_only=True)
                    self.cons[full_lang] = con

                con = self.cons[full_lang]

                # Batch query: check multiple UUIDs at once
                # FineWeb (English) only has the `id` column to test for existence
                if full_lang == self.fw_eng_tag:
                    uuids = [extract_uuid(doc.id) for doc in doc_group]
                    placeholders = ", ".join(["(?)"] * len(uuids))
                    query = f"""
                        SELECT CASE WHEN d.id IS NOT NULL THEN 1 ELSE 0 END AS exists
                        FROM (VALUES {placeholders}) AS v(id)
                        LEFT JOIN data d
                        ON v.id = d.id;
                    """
                    results = con.execute(query, uuids).fetchall()
                else:
                    uuids = [(doc.metadata["dump"], extract_uuid(doc.id)) for doc in doc_group]
                    placeholders = ", ".join(["(?, ?)"] * len(uuids))
                    query = f"""
                        SELECT CASE WHEN d.id IS NOT NULL THEN 1 ELSE 0 END AS exists
                        FROM (VALUES {placeholders}) AS v(dump, id)
                        LEFT JOIN data d
                        ON v.dump = d.dump AND v.id = d.id;
                    """
                    results = con.execute(query, [value for pair in uuids for value in pair]).fetchall()

                results = [bool(r[0]) for r in results]

                # Update documents with results
                for doc, exists in zip(doc_group, results):
                    doc.metadata[self.added_key] = exists
                    yield doc

    def __del__(self):
        for con in self.cons.values():
            con.close()