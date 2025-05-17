from typing import Iterator

from datatrove.data import Document

from c5.components.annotators.base import BaseBatchAnnotator
from c5.utils import extract_uuid


class FWSingleDBContainmentAnnotator(BaseBatchAnnotator):
    name = "🗄️ FW Database Containment Annotator"

    _requires_dependencies = ["duckdb"]

    def __init__(
        self,
        duckdb_path: str,
        is_fw2: bool,
        added_key: str,
        batch_size: int = 512,
        overwrite_with_none: bool = False,
    ):
        super().__init__(batch_size=batch_size)

        self.duckdb_path = duckdb_path
        self.is_fw2 = is_fw2
        self.added_key = added_key

        self._con = None
        self.overwrite_with_none = overwrite_with_none

    @property
    def con(self):
        if self._con is None:
            import duckdb

            self._con = duckdb.connect(self.duckdb_path, read_only=True)
        return self._con

    def annotate(self, docs: list[Document]) -> Iterator[Document]:
        if self.overwrite_with_none:
            for doc in docs:
                doc.metadata[self.added_key] = None
                yield doc
            return
        else:
            # Batch query: check multiple UUIDs at once
            # FineWeb (English) only has the `id` column to test for existence
            if not self.is_fw2:
                uuids = [extract_uuid(doc.id) for doc in docs]
                placeholders = ", ".join(["(?)"] * len(uuids))
                query = f"""
                    SELECT CASE WHEN d.id IS NOT NULL THEN 1 ELSE 0 END AS exists
                    FROM (VALUES {placeholders}) AS v(id)
                    LEFT JOIN data d
                    ON v.id = d.id;
                """
                results = self.con.execute(query, uuids).fetchall()
            else:
                uuids = [(doc.metadata["dump"], extract_uuid(doc.id)) for doc in docs]
                placeholders = ", ".join(["(?, ?)"] * len(uuids))
                query = f"""
                    SELECT CASE WHEN d.id IS NOT NULL THEN 1 ELSE 0 END AS exists
                    FROM (VALUES {placeholders}) AS v(dump, id)
                    LEFT JOIN data d
                    ON v.dump = d.dump AND v.id = d.id;
                """
                results = self.con.execute(query, [value for pair in uuids for value in pair]).fetchall()

            results = [bool(r[0]) for r in results]

            # Update documents with results
            for doc, exists in zip(docs, results):
                doc.metadata[self.added_key] = exists
                yield doc

    def __del__(self):
        if self._con is not None:
            self._con.close()
            self._con = None