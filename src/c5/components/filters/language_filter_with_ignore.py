from datatrove.pipeline.filters.language_filter import LanguageFilter
from typing import Literal

from datatrove.data import Document
from datatrove.pipeline.writers.disk_base import DiskWriter


class LanguageFilterWithIgnore(LanguageFilter):
    def __init__(       
        self,     
        languages: list[str] | str | None = None,
        ignore_languages: list[str] | str | None = None,
        language_threshold: float = 0.65,
        exclusion_writer: DiskWriter = None,
        backend: Literal["ft176", "glotlid"] = "ft176",
        label_only: bool = False,
        keep_top_pairs_threshold: float = -1,
    ):
        """
        See  datatrove.pipeline.filters.language_filter.LanguageFilter
        filters if the predicted language is not among given language or if the language score is below language
        language_threshold

        ADDED: `ignore_languages`. It has precedence over `label_only` and `languages`. If a language is in this list,
        the document is removed regardless of the language score. 

        Args:
            languages: list of languages to keep. None for all
            ignore_languages: list of languages to ignore. If the language is in this list, the document is removed
                regardless of the language score. 
            language_threshold: language_threshold minimum score to accept a document
            exclusion_writer:
            label_only: if True, only the language label is added to the metadata and no documents are removed
            keep_top_pairs_threshold: keep a list of all language pairs with at least this score. -1 to disable
        """
        super().__init__(
            languages=languages,
            language_threshold=language_threshold,
            exclusion_writer=exclusion_writer,
            backend=backend,
            label_only=label_only,
            keep_top_pairs_threshold=keep_top_pairs_threshold,
        )
        if isinstance(ignore_languages, str):
            ignore_languages = [ignore_languages]
        elif ignore_languages is None:
            ignore_languages = []
        self.ignore_languages = ignore_languages

    def filter(self, doc: Document) -> bool:
        """Args:
            doc: document

        Returns:
            is_filter
        """
        best_lang_pair, lang_pairs = self.model.predict(doc)
        lang, lang_score = best_lang_pair
        if self.backend == "glotlid":
            lang, script = lang.split("_")
            doc.metadata["language_script"] = script
        doc.metadata["language"] = lang
        doc.metadata["language_score"] = lang_score
        if self.keep_top_pairs_threshold != -1:
            for key, value in lang_pairs.items():
                if value > self.keep_top_pairs_threshold:
                    doc.metadata[f"top_language_{key}_score"] = value

        for ignore_lang in self.ignore_languages:
            # If using glotlid: at this point `lang` is the language code, not include the script
            # the script is in `doc.metadata["language_script"]`
            if lang == ignore_lang:
                return False, f"language in ignore list: {ignore_lang}" 
        
        return (
            self.label_only
            or (self.languages and any(score > self.language_threshold for score in lang_pairs.values()))
            or (self.languages is None and lang_score > self.language_threshold)
        )