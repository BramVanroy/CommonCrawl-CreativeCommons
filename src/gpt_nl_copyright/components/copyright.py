from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep

from gpt_nl_copyright.copyright_finder import find_cc_licenses_in_html


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
            # List of tuples (license_abbr, license_version, location_found)
            potential_licenses = find_cc_licenses_in_html(html)
            # Licenses are sorted by the best match
             # Order of preference based on where the license was found: meta_tag, json-ld, link_tag, a_tag
            extracted_license = potential_licenses[0]
            license_abbr = None
            license_version = None
            if extracted_license is not None:                
                license_abbr, license_version, _ = extracted_license
                
            doc.metadata["extracted_license_abbr"] = license_abbr
            doc.metadata["extracted_license_version"] = license_version
            doc.metadata["potential_licenses"] = potential_licenses
            # Remove the added HTML mtd
            doc.metadata.pop("html")
            yield doc
