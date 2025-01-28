from datatrove.data import Document

from ...utils import extract_uuid
from .base import BaseAnnotator


class DatabaseContainmentAnnotator(BaseAnnotator):
    name = "🗄️ Database Containment Annotator"

    _requires_dependencies = ["duckdb"]

    def __init__(self, duckdb_templ_path: str, added_key: str, ignore_duckdb_for: list[str]):
        super().__init__()

        if not duckdb_templ_path or "{language}" not in duckdb_templ_path:
            raise ValueError("The duckdb_templ_path must contain the placeholder '{language}'")
        self.duckdb_template = duckdb_templ_path
        self.added_key = added_key
        self.ignore_duckdb_for = ignore_duckdb_for
        self.cons = {}

    def annotate(self, doc: Document) -> Document:
        import duckdb

        language = doc.metadata["language"]
        language_script = doc.metadata["language_script"]
        full_lang = f"{language}_{language_script}"
        if full_lang in self.ignore_duckdb_for:
            doc.metadata[self.added_key] = None
            return doc

        if full_lang not in self.cons:
            duckdb_path = self.duckdb_template.format(language=full_lang)
            con = duckdb.connect(duckdb_path, read_only=True)
            self.cons[full_lang] = con

        con = self.cons[full_lang]
        uuid = extract_uuid(doc.id)
        dump = doc.metadata["dump"]
        query = "SELECT EXISTS (SELECT 1 FROM data WHERE dump = ? AND id = ?)"
        exists = con.execute(query, (dump, uuid)).fetchone()[0]
        doc.metadata[self.added_key] = bool(exists)
        return doc

    def __del__(self):
        for con in self.cons.values():
            con.close()
