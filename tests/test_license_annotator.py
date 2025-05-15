import pytest
from c5.components.annotators.license_annotator import License, parse_cc_license_url, find_cc_licenses_in_html, sort_licenses


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
        # No license
        (
            "<html><head></head><body>No license here</body></html>",
            []
        ),
        # License in meta tag
        (
            """<html><head><meta name="license" content="https://creativecommons.org/licenses/by-nc-nd/4.0/"></head></html>""",
            [License(abbr="by-nc-nd", version="4.0", location="meta_tag", in_head=True, in_footer=False)]
        ),
        # License in meta tag (cased)
        (
            """<html><head><META NAME="LICENSE" CONTENT="https://creativecommons.org/licenses/by-nc/3.0/"></head></html>""",
            [License(abbr="by-nc", version="3.0", location="meta_tag", in_head=True, in_footer=False)]
        ),
        # License in link tag
        (
            """<html><head><link href="https://creativecommons.org/licenses/by/3.0/"></head></html>""",
            [License(abbr="by", version="3.0", location="link_tag", in_head=True, in_footer=False)]
        ),
        # License in link tag (cased)
        (
            """<html><head><LINK HREF="https://creativecommons.org/licenses/by/3.0/"></head></html>""",
            [License(abbr="by", version="3.0", location="link_tag", in_head=True, in_footer=False)]
        ),
        # License in a tag
        (
            """<html><body><footer><a href="https://creativecommons.org/licenses/by-sa/2.0/"></a></footer></body></html>""",
            [License(abbr="by-sa", version="2.0", location="a_tag", in_head=False, in_footer=True)]
        ),
        # License in a tag (cased)
        (
            """<html><body><FOOTER><A HREF="https://creativecommons.org/licenses/by-sa/2.0/"></A></FOOTER></body></html>""",
            [License(abbr="by-sa", version="2.0", location="a_tag", in_head=False, in_footer=True)]
        ),
        # License in JSON-LD
        (
            """<html><body><script type="application/ld+json">
            {"@context":"http://schema.org","license":"https://creativecommons.org/licenses/by-nd/4.0/"}
            </script></body/html>""",
            [License(abbr="by-nd", version="4.0", location="json-ld", in_head=False, in_footer=False)]
        ),
        # License in multiple locations
        (
            """<html>
            <head><meta name="license" content="https://creativecommons.org/licenses/zero/1.0/"></head>
            <body><a href="https://creativecommons.org/licenses/by/4.0/"></a></body>
            </html>""",
            [License(abbr="zero", version="1.0", location="meta_tag", in_head=True, in_footer=False),
             License(abbr="by", version="4.0", location="a_tag", in_head=False, in_footer=False)]
        ),
        # Invalid license URL
        (
            """<html><body><a href="https://example.com/licenses/by-nc-nd/4.0/"></a></body></html>""",
            []
        ),
        # License with unknown version
        (
            """<html><body><a href="https://creativecommons.org/licenses/unknown/2.0/"></a></body></html>""",
            [License(abbr="cc-unknown", version=None, location="a_tag", in_head=False, in_footer=False)]
        ),
        # License in footer element
        (
            """<html><body><footer><a href="https://creativecommons.org/licenses/by-nc-nd/4.0/"></a></footer></body></html>""",
            [License(abbr="by-nc-nd", version="4.0", location="a_tag", in_head=False, in_footer=True)]
        ),        
        # License in footer element (cased)
        (
            """<html><body><FOOTER><a href="https://creativecommons.org/licenses/by-nc-nd/4.0/"></a></FOOTER></body></html>""",
            [License(abbr="by-nc-nd", version="4.0", location="a_tag", in_head=False, in_footer=True)]
        ),
        # License in json-ld as a dict
        (
            """<html><body><script type="application/ld+json">
            {"@context":"http://schema.org","license":{"url":"https://creativecommons.org/licenses/by-nd/4.0/"}}
            </script></body></html>""",
            [License(abbr="by-nd", version="4.0", location="json-ld", in_head=False, in_footer=False)]
        ),
        # License in json-ld as a list
        (
            """<html><body><script type="application/ld+json">
            {"@context":"http://schema.org","license":[{"url":"https://creativecommons.org/licenses/by-nd/4.0/"}]}
            </script></body></html>""",
            [License(abbr="by-nd", version="4.0", location="json-ld", in_head=False, in_footer=False)]
        ),
        # License in json-ld as a list with multiple licenses
        (
            """<html><body><script type="application/ld+json">
            {"@context":"http://schema.org","license":[{"url":"https://creativecommons.org/licenses/by-nd/4.0/"},{"url":"https://creativecommons.org/licenses/by/3.0/"}]}
            </script></body></html>""",
            [License(abbr="by-nd", version="4.0", location="json-ld", in_head=False, in_footer=False),
             License(abbr="by", version="3.0", location="json-ld", in_head=False, in_footer=False)]
        ),

        # License in json-ld as a dict (cased)
        (
            """<html><body><SCRIPT TYPE="APPLICATION/LD+JSON">
            {"@context":"http://schema.org","LICENSE":{"url":"https://creativecommons.org/licenses/by-nd/4.0/"}}
            </SCRIPT></body></html>""",
            [License(abbr="by-nd", version="4.0", location="json-ld", in_head=False, in_footer=False)]
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
            [License(abbr="by", version="3.0", location="meta_tag", in_head=True, in_footer=False)],
            [License(abbr="by", version="3.0", location="meta_tag", in_head=True, in_footer=False)]
        ),
        # Different locations
        (
            [
                License(abbr="by", version="4.0", location="link_tag", in_head=True, in_footer=False),
                License(abbr="by-nc-nd", version="4.0", location="meta_tag", in_head=False,in_footer= False),
                License(abbr="by-sa", version="3.0", location="a_tag", in_head=False, in_footer=False),
                License(abbr="zero", version="1.0", location="json-ld", in_head=False, in_footer=False),
            ],
            [
                License(abbr="by-nc-nd", version="4.0", location="meta_tag", in_head=False, in_footer=False),
                License(abbr="zero", version="1.0", location="json-ld", in_head=False, in_footer=False),
                License(abbr="by", version="4.0", location="link_tag", in_head=True, in_footer=False),
                License(abbr="by-sa", version="3.0", location="a_tag", in_head=False, in_footer=False),
            ],
        ),
        # Same location, different head/footer
        (
            [
                License(abbr="by-nc-sa", version="3.0", location="link_tag", in_head=False, in_footer=True),
                License(abbr="by-nc", version="4.0", location="link_tag", in_head=True, in_footer=False),
                License(abbr="by-nd", version="4.0", location="link_tag", in_head=False, in_footer=False),
                
            ],
            [
                License(abbr="by-nc", version="4.0", location="link_tag", in_head=True, in_footer=False),
                License(abbr="by-nc-sa", version="3.0", location="link_tag", in_head=False, in_footer=True),
                License(abbr="by-nd", version="4.0", location="link_tag", in_head=False, in_footer=False),
            ],
        ),
    ],
)
def test_sort_licenses(unsorted_results, expected_sorted):
    assert sort_licenses(unsorted_results) == expected_sorted
