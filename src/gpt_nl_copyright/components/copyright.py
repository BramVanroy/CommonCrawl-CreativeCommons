from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep

from gpt_nl_copyright.copyright_finder import ParserException, find_cc_licenses_in_html


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
            
            license_abbr = None
            license_version = None
            license_location = None
            potential_licenses = None
            license_parse_error = None
            license_disagreement = None

            # List of tuples (license_abbr, license_version, location_found)
            try:
                potential_licenses = find_cc_licenses_in_html(html)
            except Exception:
                license_parse_error = True
            else:                
                license_parse_error = False
                if potential_licenses:
                    # Licenses are sorted by the best match
                    # Order of preference based on where the license was found: meta_tag, json-ld, link_tag, a_tag              
                    extracted_license = potential_licenses[0]              
                    license_abbr, license_version, license_location = extracted_license
                    # If not all licenses have the same abbreviation, there is a disagreement
                    license_disagreement = len(set(lic[0] for lic in potential_licenses)) > 1

            doc.metadata["license_abbr"] = license_abbr
            doc.metadata["license_version"] = license_version
            doc.metadata["license_location"] = license_location
            doc.metadata["potential_licenses"] = potential_licenses
            doc.metadata["license_parse_error"] = license_parse_error
            doc.metadata["license_disagreement"] = license_disagreement

            # Remove the added HTML mtd
            doc.metadata.pop("html")
            yield doc
