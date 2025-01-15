import pytest
from commoncrawl_cc_annotation.components.annotators.license_annotator import parse_cc_license_url, find_cc_licenses_in_html, sort_licenses


@pytest.mark.parametrize(
    "license_url, expected_abbr, expected_version",
    [
        ("https://creativecommons.org/licenses/by-nc-nd/4.0/", "by-nc-nd", "4.0"),
        ("https://creativecommons.org/publicdomain/zero/1.0/", "zero", "1.0"),
        ("https://creativecommons.org/licenses/by/3.0/", "by", "3.0"),
        ("https://creativecommons.org/licenses/by-sa/2.5/", "by-sa", "2.5"),
        ("https://creativecommons.org/licenses/by-nd/1.0/", "by-nd", "1.0"),
        ("https://creativecommons.org/licenses/by-nc/4.0/", "by-nc", "4.0"),
        ("https://creativecommons.org/licenses/by-nc-sa/3.0/", "by-nc-sa", "3.0"),
        ("https://creativecommons.org/licenses/by-nc-nd/2.0/", "by-nc-nd", "2.0"),
        ("https://creativecommons.org/licenses/certification/1.0/", "certification", "1.0"),
        ("https://creativecommons.org/licenses/mark/1.0/", "mark", "1.0"),
        ("https://creativecommons.org/licenses/unknown/1.0/", "cc-unknown", None),
        ("https://example.com/licenses/by-nc-nd/4.0/", None, None),
        ("https://creativecommons.org/licenses/by-nc-nd/", "cc-unknown", None),
        ("https://creativecommons.org/licenses/by-nc-nd/4.0/some-extra-path", "by-nc-nd", "4.0"),
    ]
)
def test_parse_cc_license_url(license_url, expected_abbr, expected_version):
    abbr, version = parse_cc_license_url(license_url)
    assert abbr == expected_abbr
    assert version == expected_version

@pytest.mark.parametrize(
    "html,expected",
    [
        (
            "<html><head></head><body>No license here</body></html>",
            []
        ),
        (
            """<html><head><meta name="license" content="https://creativecommons.org/licenses/by-nc-nd/4.0/"></head></html>""",
            [("by-nc-nd", "4.0", "meta_tag", True, False)]
        ),
        (
            """<html><head><link href="https://creativecommons.org/licenses/by/3.0/"></head></html>""",
            [("by", "3.0", "link_tag", True, False)]
        ),
        (
            """<html><footer><a href="https://creativecommons.org/licenses/by-sa/2.0/"></a></footer></html>""",
            [("by-sa", "2.0", "a_tag", False, True)]
        ),
        (
            """<html><script type="application/ld+json">
            {"@context":"http://schema.org","license":"https://creativecommons.org/licenses/by-nd/4.0/"}
            </script></html>""",
            [("by-nd", "4.0", "json-ld", False, False)]
        ),
        (
            """<html>
            <head><meta name="license" content="https://creativecommons.org/licenses/zero/1.0/"></head>
            <body><a href="https://creativecommons.org/licenses/by/4.0/"></a></body>
            </html>""",
            [("zero", "1.0", "meta_tag", True, False), ("by", "4.0", "a_tag", False, False)]
        ),
        (
            """<html><body><a href="https://example.com/licenses/by-nc-nd/4.0/"></a></body></html>""",
            []
        ),
        (
            """<html><body><a href="https://creativecommons.org/licenses/unknown/2.0/"></a></body></html>""",
            [("cc-unknown", None, "a_tag", False, False)]
        ),
    ],
)
def test_find_cc_licenses_in_html(html, expected):
    results = find_cc_licenses_in_html(html)
    assert results == expected

@pytest.mark.parametrize(
    "unsorted_results, expected_sorted",
    [
        # Single item
        (
            [("by", "3.0", "meta_tag", True, False)],
            [("by", "3.0", "meta_tag", True, False)],
        ),
        # Different locations
        (
            [
                ("by", "4.0", "link_tag", True, False),
                ("by-nc-nd", "4.0", "meta_tag", False, False),
                ("by-sa", "3.0", "a_tag", False, False),
                ("zero", "1.0", "json-ld", False, False),
            ],
            [
                ("by-nc-nd", "4.0", "meta_tag", False, False),
                ("zero", "1.0", "json-ld", False, False),
                ("by", "4.0", "link_tag", True, False),
                ("by-sa", "3.0", "a_tag", False, False),
            ],
        ),
        # Same location, different head/footer
        (
            [
                ("by-nc-sa", "3.0", "link_tag", False, True),
                ("by-nc", "4.0", "link_tag", True, False),
                ("by-nd", "4.0", "link_tag", False, False),
            ],
            [
                ("by-nc", "4.0", "link_tag", True, False),
                ("by-nc-sa", "3.0", "link_tag", False, True),
                ("by-nd", "4.0", "link_tag", False, False),
            ],
        ),
    ],
)
def test_sort_licenses(unsorted_results, expected_sorted):
    assert sort_licenses(unsorted_results) == expected_sorted
