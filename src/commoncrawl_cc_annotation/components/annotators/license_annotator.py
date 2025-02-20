import json
import re
import warnings
from typing import Literal
from urllib.parse import unquote

from datatrove.data import Document

from .base import BaseAnnotator


class LicenseAnnotator(BaseAnnotator):
    name = "©️ License Annotator"

    _requires_dependencies = [("bs4", "beautifulsoup4")]

    def __init__(self, html_in_metadata: bool = False, remove_html: bool = False):
        super().__init__()
        self.html_in_metadata = html_in_metadata
        self.remove_html = remove_html

    def annotate(self, doc: Document) -> Document:
        license_abbr = None
        license_version = None
        license_location = None
        license_in_head = None
        license_in_footer = None

        potential_licenses = None
        license_parse_error = None
        license_disagreement = None

        if self.html_in_metadata:
            html = doc.metadata["html"]
        else:
            html = doc.text

        try:
            # List of tuples (license_abbr, license_version, location_found)
            potential_licenses = find_cc_licenses_in_html(html)
        except Exception:
            license_parse_error = True
        else:
            license_parse_error = False
            if potential_licenses:
                # Licenses are sorted by the best match
                # Order of preference based on where the license was found: meta_tag, json-ld, link_tag, a_tag
                extracted_license = potential_licenses[0]
                license_abbr, license_version, license_location, license_in_head, license_in_footer = extracted_license
                # If not all licenses have the same abbreviation, there is a disagreement
                license_disagreement = len({lic[0] for lic in potential_licenses}) > 1
                # Restructure licenses to be a dictionary because Arrow does not allow lists of heterogeneous types
                potential_licenses = {k: [r[i] for r in potential_licenses] for i, k in enumerate(license_tuple_keys)}

        doc.metadata["license_abbr"] = license_abbr
        doc.metadata["license_version"] = license_version
        doc.metadata["license_location"] = license_location
        doc.metadata["license_in_head"] = license_in_head
        doc.metadata["license_in_footer"] = license_in_footer

        doc.metadata["potential_licenses"] = potential_licenses
        doc.metadata["license_parse_error"] = license_parse_error
        doc.metadata["license_disagreement"] = license_disagreement

        if self.remove_html and self.html_in_metadata:
            del doc.metadata["html"]

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


location_type = Literal["meta_tag", "a_tag", "link_tag", "json-ld"]
abbr_type = Literal[
    "cc-unknown", "by", "by-sa", "by-nd", "by-nc", "by-nc-sa", "by-nc-nd", "zero", "certification", "mark"
]

location_preference_order = ["meta_tag", "json-ld", "link_tag", "a_tag"]
head_preference_order = [True, False]
footer_preference_order = [True, False]
license_tuple_keys = ("abbr", "version", "location", "in_head", "in_footer")


def parse_cc_license_url(license_url: str) -> tuple[abbr_type | None, str | None]:
    """Given a URL that might be from creativecommons.org, try to parse out the license type and version.

    Args:
        license_url: the URL to parse

    Returns:
        tuple[str, str]: the license abbreviation and version, e.g. ('by-nc-nd', '4.0')
    """
    url_lower = unquote(license_url).lower()

    if "creativecommons.org" not in url_lower:
        return None, None

    # Typical CC license URLs look like:
    #   https://creativecommons.org/licenses/by-nc-nd/4.0/
    # or
    #   https://creativecommons.org/publicdomain/zero/1.0/
    match = re.search(r"creativecommons\.org/(?:licenses|publicdomain)/([^/]+)/(\d\.\d)", url_lower)

    # "creativecommons.org" in the url but not a known license pattern
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


def find_cc_licenses_in_html(html: str) -> list[tuple[abbr_type, str | None, location_type, bool, bool]]:
    """Given an HTML document (as str), try to find all Creative Commons licenses, if any.

    Args:
        html: the HTML document as a string

    Returns:
        list[tuple[str, str, str, bool, bool]]: a list of tuples with the license abbreviation, version, location,
        whether it was found in the head, and whether it was found in the footer
    """
    from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning, Tag, XMLParsedAsHTMLWarning

    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
    # Some files are malformed and only contain text like
    # "Table './dlinksmf/smf_sessions' is marked as crashed and should be repaired"
    # Those should not be parsed and simply be raised as an error
    warnings.filterwarnings("error", category=MarkupResemblesLocatorWarning)

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

    def parse_content_license(potential_cc_url: str, license_place: str, tag: Tag):
        if potential_cc_url:
            license_abbr, license_version = parse_cc_license_url(potential_cc_url)
            if license_abbr:
                in_head, in_footer = has_head_or_footer_ancestor(tag)
                # These options are mutually exclusive
                results.append((license_abbr, license_version, license_place, in_head, in_footer))

    # Check <meta name="license"> or <meta property="og:license"> for its "content" attribute
    for meta_tag in soup.find_all("meta"):
        meta_name = meta_tag.get("name", "") or meta_tag.get("property", "")
        if meta_name.lower() in ["license", "og:license"]:
            if content := meta_tag.get("content"):
                parse_content_license(content, "meta_tag", meta_tag)

    # Check <link href="..."> or <a href="..."> for its "href" attribute
    for tag in soup.find_all(("link", "a")):
        if href := tag.get("href"):
            parse_content_license(href, f"{tag.name}_tag", tag)

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
        except json.JSONDecodeError:
            continue
        else:
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
                        if license_url := license_val.get("@id") or license_val.get("url"):
                            parse_content_license(license_url, "json-ld", script_tag)
                    elif isinstance(license_val, str):
                        parse_content_license(license_val, "json-ld", script_tag)

    return sort_licenses(results)


def sort_licenses(
    results: list[tuple[abbr_type, str | None, location_type, bool, bool]],
) -> list[tuple[abbr_type, str | None, location_type, bool, bool]]:
    """Sort the license results (list of tuples) by the following order of preference for each item in the tuple:
    1. location_preference_order: meta_tag, json-ld, link_tag, a_tag
    2. head_preference_order: True, False
    3. footer_preference_order: True, False

    Args:
        results: the list of license results to sort. Each result is a tuple with the license abbreviation, version,
        location, whether it was found in the head, and whether it was found in the footer

    Returns:
        list[tuple[str, str, str, bool, bool]]: the sorted list of license results
    """
    return sorted(
        results,
        key=lambda lic: (
            location_preference_order.index(lic[2]),
            head_preference_order.index(lic[3]),
            footer_preference_order.index(lic[4]),
        ),
    )


def has_head_or_footer_ancestor(tag) -> tuple[bool, bool]:
    """Check if the tag has a head ancestor or a footer ancestor. The
    options are mutually exclusive for normal HTML. The `head` cannot be in a
    `footer` element and a `footer` element cannot be in a `head` element.

    Args:
        tag: the bs4 Tag to check

    Returns:
        tuple[bool]: a tuple with two booleans, the first is True if the tag has a head ancestor,
        the second is True if the tag has a footer ancestor
    """
    if tag is None:
        return False, False

    if tag.name.lower() == "head":
        return True, False
    elif (
        tag.name.lower() == "footer"
        or any("footer" in cls.lower() for cls in tag.get("class", []))
        or "footer" in tag.get("id", "").lower()
    ):
        return False, True

    return has_head_or_footer_ancestor(tag.parent)
