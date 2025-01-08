"""
This file borrows from the work done by Robin Van Craenenbroek in ML6's Fondant project. The original source code can be found here:
https://github.com/ml6team/fondant-usecase-filter-creative-commons/blob/f4fc645bfc8eb9bf244565b9bce40cadd16d3597/image_extraction/components/extract_images_from_warc/src/main.py
"""

import re
from typing import Literal
from urllib.parse import unquote, urlparse

from bs4 import BeautifulSoup, Tag, XMLParsedAsHTMLWarning
import warnings

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

CC_LICENSES = ("by-nc-sa", "by-nc-nd", "by-nc", "by-nd", "by-sa", "by")


def get_license_from_html(html: str) -> tuple[Literal["public domain", "by-nc-sa", "by-nc-nd", "by-nc", "by-nd", "by-sa", "by", None], None | str]:
    """Returns the license from the parsed html code.
    Args:
        html: The parsed html code.
    Returns:
        The license.
    """
    licenses = []
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        pass
    else:
        for a_tag in soup.find_all("a"):
            if a_tag.has_attr("href") and "creativecommons.org" in a_tag["href"]:
                href = unquote(a_tag["href"])
                license_type, license_path = get_license_type_from_creative_commons_url(href)
                if license_type is not None:
                    license_location = get_license_location(a_tag)
                    licenses.append((license_type, license_path, license_location))

    if len(licenses) == 0:
        return None, None
    elif len(licenses) == 1:
        return licenses[0][:-1]
    else:
        # If multiple licenses found, prefer the license in the footer as "most sensible"
        for license in licenses:
            if license[2] == "footer":
                return license[:-1]

        # If no license in the footer, return the first license
        return licenses[0][:-1]


def get_license_location(element: Tag) -> str:
    """Returns the license location from the parsed html code.
    Args:
        element: The parsed html code.
    Returns:
        The license location.
    """
    parent = element.parent

    if parent is None:  # could not find an apprioriate tag
        return "other"

    if parent.name == "footer" or tag_has_id_or_class_with_text(parent, "footer"):
        return "footer"
    elif (
        parent.name == "aside"
        or tag_has_id_or_class_with_text(parent, "aside")
        or tag_has_id_or_class_with_text(parent, "sidebar")
    ):
        return "aside"
    else:
        return get_license_location(parent)


def tag_has_id_or_class_with_text(tag: Tag, string: str) -> bool:
    """Returns True if the tag has an id or class that contains the given string.
    Args:
        tag: The tag to check.
        string: The string to check for.
    Returns:
        True if the tag has an id or class that contains the given string.
    """
    if tag.has_attr("id") and string in tag["id"]:
        return True
    if tag.has_attr("class"):
        for class_ in tag["class"]:
            if string in class_:
                return True
    return False


def get_license_type_from_creative_commons_url(
    license_url: str,
) -> tuple[Literal["public domain", "by-nc-sa", "by-nc-nd", "by-nc", "by-nd", "by-sa", "by", None], None | str]:
    """Returns the license type from the creative commons url.
    Args:
        license_url: The creative commons url.
    Returns:
        The license type.
    """
    # License path may also include the language, e.g. /licenses/by/4.0/deed.en, or the specific
    # version, e.g. /licenses/by/4.0, so we keep it
    try:
        license_path = urlparse(license_url).path.strip("/")
        license_split = license_path.split("/")
    except ValueError:
        return None, None

    if "publicdomain" in license_split:
        return "public domain", license_path
    else:
        for short_license in license_split:
            if "by" in short_license.lower():
                short_license = refine_cc_license(short_license)
                if short_license is not None:
                    return short_license, license_path

    return None, None


def refine_cc_license(license: str) -> Literal["by-nc-sa", "by-nc-nd", "by-nc", "by-nd", "by-sa", "by", None]:
    """
    Retrieve the Creative Commons license from a string. Exact filtering is done to avoid cases where URLs coincidentally
    contain `by`, e.g. `astrology-by-sfgate-uncover-your-cosmic-destiny`.

    Args:
        license: The license string with potential noise.
    Returns:
        The Creative Commons license, or None if it could not be found.
    """
    license = license.lower()
    # Strip non-alphabetic characters from the beginning and end of the string
    license = re.sub(r"^[^a-z]+|[^a-z]+$", "", license)

    # Splitting because sometimes a URL is malformed and contains the license in the URL with a space between
    # E.g. "licenses by-nc-nd"
    for chunk in license.split():
        for valid_license in CC_LICENSES:
            if valid_license == chunk:
                return valid_license

    return None
