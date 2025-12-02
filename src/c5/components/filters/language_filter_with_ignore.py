from collections import defaultdict

from datatrove.data import Document
from datatrove.pipeline.filters.language_filter import LanguageFilter
from datatrove.pipeline.writers.disk_base import DiskWriter


class LanguageFilterWithIgnore(LanguageFilter):
    def __init__(
        self,
        languages: list[str] | str | None = None,
        ignore_undetermined: bool = True,
        language_threshold: float | dict = 0.65,
        exclusion_writer: DiskWriter = None,
        label_only: bool = False,
    ):
        """
        See  datatrove.pipeline.filters.language_filter.LanguageFilter
        filters if the predicted language is not among given language or if the language score is below language
        language_threshold

        Args:
            languages: list of languages to keep. None for all
            ignore_undetermined: if True, ignore undetermined and non-linguistic languages (`und_` and `zxx_`)
            language_threshold: language_threshold minimum score to accept a document. Can be a float or a dict with
                language as key and threshold as value.
            exclusion_writer:
            label_only: if True, only the language label is added to the metadata and no documents are removed
            keep_top_pairs_threshold: keep a list of all language pairs with at least this score. -1 to disable
        """
        super().__init__(
            languages=languages,
            language_threshold=language_threshold,
            exclusion_writer=exclusion_writer,
            backend="glotlid",
            label_only=label_only,
            keep_top_pairs_threshold=-1,
        )
        self.ignore_undetermined = ignore_undetermined
        self.undetermined_langs = ("und", "zxx")

        if self.languages is not None:
            self.languages = set(self.languages)

        # If language_threshold is a float, convert it to a dict with default value
        # so any language will have the same threshold
        if isinstance(self.language_threshold, float):
            self.language_threshold = defaultdict(lambda: self.language_threshold)
        elif not isinstance(self.language_threshold, dict):
            raise ValueError("language_threshold must be a float or a dict")

        print("LANGUAGE FILTER")
        print("LANGUAGES", self.languages)
        print("IGNORE UND/ZXX", self.ignore_undetermined)
        print("LANGUAGE THRESHOLD", self.language_threshold)
        print("LABEL ONLY", self.label_only)

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        """Args:
            doc: document

        Returns:
            is_filter
        """
        _, lang_pairs = self.model.predict(doc)
        # Sort by score
        lang_pairs = dict(sorted(lang_pairs.items(), key=lambda item: item[1], reverse=True))
        # Highest score first
        for lang, score in lang_pairs.items():
            if score > self.language_threshold[lang]:
                best_lang_pair = (lang, score)
                break
        else:
            if not self.label_only:
                return False, "no_language_above_its_threshold"
            else:
                # If no language matches its threshold, we can still return the best one
                # which is the first one in the sorted dictionary
                best_lang_pair = next(iter(lang_pairs.items()))

        lang, lang_score = best_lang_pair

        if not self.label_only and self.languages is not None and lang not in self.languages:
            return False, f"{lang}_not_in_requested_languages"

        lang, script = lang.split("_")

        if not self.label_only and self.ignore_undetermined and lang in self.undetermined_langs:
            return False, "undetermined_or_non_linguistic_language"

        doc.metadata["language_script"] = script
        doc.metadata["language"] = lang
        doc.metadata["language_score"] = lang_score

        return True
