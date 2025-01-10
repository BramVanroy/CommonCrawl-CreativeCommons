import json
import re
import warnings
from typing import Literal
from urllib.parse import unquote

from datatrove.data import Document

from gpt_nl_copyright.components.annotator.base import BaseAnnotator


class CopyrightAnnotator(BaseAnnotator):
    name = "©️ Copyright Annotator"

    _requires_dependencies = [("bs4", "beautifulsoup4")]

    def __init__(self):
        super().__init__()

    def annotate(self, doc: Document) -> Document:
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
                license_disagreement = len({lic[0] for lic in potential_licenses}) > 1

        doc.metadata["license_abbr"] = license_abbr
        doc.metadata["license_version"] = license_version
        doc.metadata["license_location"] = license_location
        doc.metadata["potential_licenses"] = potential_licenses
        doc.metadata["license_parse_error"] = license_parse_error
        doc.metadata["license_disagreement"] = license_disagreement

        return doc


CC_ABBR_TO_LICENSE = {
    "by",
    "by-sa",
    "by-nd",
    "by-nc",
    "by-nc-sa",
    "by-nc-nd",
    "zero",
    "certification",
    "mark",
}


# Add typing for location type
location_type = Literal["meta_tag", "a_tag", "link_tag", "json-ld"]
abbr_type = Literal[
    "cc-unknown", "by", "by-sa", "by-nd", "by-nc", "by-nc-sa", "by-nc-nd", "zero", "certification", "mark"
]


def parse_cc_license_url(license_url: str) -> tuple[abbr_type | None, str | None]:
    """
    Given a URL that might be from creativecommons.org,
    try to parse out the license type and version.
    Returns a string like 'CC BY-NC-ND 4.0' or None if not recognized.
    """
    # Normalize to lowercase for easier parsing
    url_lower = unquote(license_url).lower()

    if "creativecommons.org" not in url_lower:
        return None, None

    # Typical CC license URLs look like:
    #   https://creativecommons.org/licenses/by-nc-nd/4.0/
    # or
    #   https://creativecommons.org/publicdomain/zero/1.0/
    match = re.search(r"creativecommons\.org/(?:licenses|publicdomain)/([^/]+)/(\d\.\d)", url_lower)
    if not match:
        return "cc-unknown", None

    license_code = match.group(1)  # e.g. 'by-nc-nd' or 'zero'
    license_code = re.sub(r"^[^a-z]+|[^a-z]+$", "", license_code)
    version = match.group(2)  # e.g. '4.0' or '1.0'

    if license_code in CC_ABBR_TO_LICENSE:
        return license_code, version
    else:
        return "cc-unknown", None


class ParserException(Exception):
    """An Exception to be raised when all parsers fail"""

    def __init__(self, message_or_exception):
        if isinstance(message_or_exception, Exception):
            e = message_or_exception
            message_or_exception = "%s: %s" % (e.__class__.__name__, str(e))
        super().__init__(message_or_exception)


def find_cc_licenses_in_html(html: str) -> list[tuple[abbr_type, str | None, location_type]]:
    """
    Returns a list of tuples (cc_license_string, location_found),
    covering as many metadata locations as possible.
    """
    from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

    results = []

    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        try:
            soup = BeautifulSoup(html, "html5lib")
        except Exception:
            try:
                soup = BeautifulSoup(html, "lxml")
            except Exception:
                raise ParserException("Could not parse the document with html.parser, html5lib, nor lxml.")

    def parse_content_license(content: str, license_place: str):
        if content:
            license_abbr, license_version = parse_cc_license_url(content)
            if license_abbr:
                results.append((license_abbr, license_version, license_place))

    # Check <meta name="license"> or <meta property="og:license"> for its "content" attribute
    for meta_tag in soup.find_all("meta"):
        meta_name = meta_tag.get("name", "") or meta_tag.get("property", "")
        if meta_name.lower() in ["license", "og:license"]:
            parse_content_license(meta_tag.get("content"), "meta_tag")

    # Check <link href="..."> or <a href="..."> for its "href" attribute
    for tag in soup.find_all(("link", "a")):
        parse_content_license(tag.get("href"), f"{tag.name}_tag")

    # Check JSON-LD (Schema.org) for "license": "...", usually in <script type="application/ld+json">
    # Example JSON-LD:
    # <script type="application/ld+json">
    # {
    #     "@context": "http://schema.org",
    #     "@type": "CreativeWork",
    #     "license": {
    #         "@type": "CreativeWork",
    #         "url": "https://creativecommons.org/licenses/by-nc-nd/4.0/"
    #     }
    # }
    for script_tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            data = json.loads(script_tag.string or "")
            # data could be a list or dict
            if isinstance(data, dict):
                data_list = [data]
            elif isinstance(data, list):
                data_list = data
            else:
                data_list = []

            for item in data_list:
                if isinstance(item, dict) and "license" in item:
                    license_val = item["license"]
                    # license_val might be a string or dict (if typed)
                    if isinstance(license_val, dict):
                        # Some schema.org usage might embed the URL in "@id" or "url"
                        license_url = license_val.get("@id") or license_val.get("url")
                        parse_content_license(license_url, "json-ld")
                    elif isinstance(license_val, str):
                        parse_content_license(license_val, "json-ld")
        except json.JSONDecodeError:
            continue

    # If multiple licenses found, order of preference: meta_tag, json-ld, link_tag, a_tag
    location_order = {"meta_tag": 0, "json-ld": 1, "link_tag": 2, "a_tag": 3}
    results.sort(key=lambda x: location_order.get(x[2], 4))
    return results
