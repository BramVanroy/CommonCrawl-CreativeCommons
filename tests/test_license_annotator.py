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
        ("http://creativecommons.org/licenses/by/3.0/es/", "by", "3.0"),
        ("https://creativecommons.org/publicdomain/mark/1.0/", "mark", "1.0"),
        # URL encoded spaces and characters - parse_cc_license_url expects unescaped '&'
        ("https://creativecommons.org/licenses/by-sa/4.0/?ref=chooser-v1&id=my%20work", "by-sa", "4.0"),
        # No scheme
        ("creativecommons.org/licenses/by/4.0/", "by", "4.0"),
    ]
)
def test_parse_cc_license_url(license_url, expected_abbr, expected_version):
    abbr, version = parse_cc_license_url(license_url)
    assert abbr == expected_abbr
    assert version == expected_version

@pytest.mark.parametrize(
    "html_input, expected_licenses",
    [
        # --- Basic Cases ---
        ("", []), # Empty HTML
        ("<!-- just a comment -->", []), # HTML with only comments
        ("<html><head></head><body>No license here</body></html>", []),
        # License in <a> tag with an HTML comment in context (which is ignored)
        (
            """<html><body><!-- License comment --><a href="https://creativecommons.org/licenses/by/4.0/">License text</a></body></html>""",
            [License(abbr="by", version="4.0", location="a_tag", in_head=False, in_footer=False,
                  tag='<a href="https://creativecommons.org/licenses/by/4.0/">License text</a>',
                  left_context="<html><body>",
                  right_context="</body></html>")]
        ),
        # License in meta tag
        (
            """<html><head><meta name="license" content="https://creativecommons.org/licenses/by-nc-nd/4.0/"></head></html>""",
            [License(abbr="by-nc-nd", version="4.0", location="meta_tag", in_head=True, in_footer=False,
                  tag='<meta content="https://creativecommons.org/licenses/by-nc-nd/4.0/" name="license"/>',
                  left_context="<html><head>",
                  right_context="</head></html>")]
        ),
        # License in link tag (rel="license" is common)
        (
            """<html><head><link rel="license" href="https://creativecommons.org/licenses/by/3.0/"/></head></html>""",
            [License(abbr="by", version="3.0", location="link_tag", in_head=True, in_footer=False,
                  tag='<link href="https://creativecommons.org/licenses/by/3.0/" rel="license"/>',
                  left_context="<html><head>",
                  right_context="</head></html>")]
        ),
        # License in <a> tag in a div that acts as a footer by class name
        (
            """<html><body><div class="footer-class"><a href="https://creativecommons.org/licenses/by-sa/2.0/">License text</a></div></body></html>""",
            [License(abbr="by-sa", version="2.0", location="a_tag", in_head=False, in_footer=True,
                  tag='<a href="https://creativecommons.org/licenses/by-sa/2.0/">License text</a>',
                  left_context='<html><body><div class="footer-class">',
                  right_context="</div></body></html>")]
        ),
        # License in JSON-LD (simple string value)
        (
            """<html><body><script type="application/ld+json">
            {"@context":"http://schema.org","license":"https://creativecommons.org/licenses/by-nd/4.0/"}
            </script></body></html>""",
            [License(abbr="by-nd", version="4.0", location="json-ld", in_head=False, in_footer=False,
                  tag='<script type="application/ld+json">{"@context":"http://schema.org","license":"https://creativecommons.org/licenses/by-nd/4.0/"}</script>',
                  left_context="<html><body>",
                  right_context="</body></html>")]
        ),
        # --- Multiple Licenses & Sorting by find_cc_licenses_in_html ---
        (
            """<html>
            <head><meta name="license" content="https://creativecommons.org/licenses/zero/1.0/"></head>  
            <body><a href="https://creativecommons.org/licenses/by/4.0/">BY License</a></body>          
            </html>""",
            [ # find_cc_licenses_in_html sorts them: meta_tag first, then a_tag
                License(abbr="zero", version="1.0", location="meta_tag", in_head=True, in_footer=False,
                     tag='<meta content="https://creativecommons.org/licenses/zero/1.0/" name="license"/>',
                     left_context="<html><head>", # soup structure might change order of head/body
                     right_context='</head><body><a href="https://creativecommons.org/licenses/by/4.0/">BY License</a></body></html>'),
                License(abbr="by", version="4.0", location="a_tag", in_head=False, in_footer=False,
                     tag='<a href="https://creativecommons.org/licenses/by/4.0/">BY License</a>',
                     left_context='head><meta content="https://creativecommons.org/licenses/zero/1.0/" name="license"/></head><body>',
                     right_context="</body></html>")
            ]
        ),
        # --- Edge Cases & Variations ---
        # License with "unknown" type in URL (becomes cc-unknown, version None)
        (
            """<html><body><a href="https://creativecommons.org/licenses/unknown/2.0/">Unknown License</a></body></html>""",
            [License(abbr="cc-unknown", version=None, location="a_tag", in_head=False, in_footer=False,
                  tag='<a href="https://creativecommons.org/licenses/unknown/2.0/">Unknown License</a>',
                  left_context="<html><body>",
                  right_context="</body></html>")]
        ),
        # License in JSON-LD (typed, license is a dict with "url")
        # Note: white-space after opening script tag... Cleaning up the HTML is not easy
        # so we'll keep the white-space edge case (after `>`) as is.
        (
            """<html><body><script type="application/ld+json">
            {"@context":"http://schema.org","license":{"@type":"CreativeWork", "url":"https://creativecommons.org/licenses/by-nd/4.0/"}}
            </script></body></html>""",
            [License(abbr="by-nd", version="4.0", location="json-ld", in_head=False, in_footer=False,
                  tag='<script type="application/ld+json">{"@context":"http://schema.org","license":{"@type":"CreativeWork", "url":"https://creativecommons.org/licenses/by-nd/4.0/"}}</script>',
                  left_context="<html><body>",
                  right_context="</body></html>")]
        ),
        # Malformed HTML (missing closing > for meta) that bs4 can often handle
        (
            """<html><head><meta name="license" content="https://creativecommons.org/licenses/by/4.0/"</head><body><p>text</body></html>""",
            [License(abbr="by", version="4.0", location="meta_tag", in_head=True, in_footer=False,
                  tag='<meta content="https://creativecommons.org/licenses/by/4.0/" name="license"/>', # bs4 fixes this
                  left_context="<html><head>",
                  right_context="</head><body><p>text</p></body></html>")]
        ),
        # License at the very beginning of the document (minimal HTML)
        (
            """<a href="https://creativecommons.org/licenses/by/4.0/">CC-BY</a>""",
            [License(abbr="by", version="4.0", location="a_tag", in_head=False, in_footer=False,
                  tag='<a href="https://creativecommons.org/licenses/by/4.0/">CC-BY</a>',
                  left_context="<html><body>",
                  right_context="</body></html>")]
        ),
        # Invalid JSON in script tag
        (
            """<html><body><script type="application/ld+json">{license: "https://creativecommons.org/licenses/by/4.0/"}</script></body></html>""",
            []
        ),
        # URL with HTML entities in attribute: tag string representation keeps &amp;, but parsed URL for abbr/version uses &
        (
            """<html><head><meta name="license" content="https://creativecommons.org/licenses/by/4.0/?ref=chooser-v1&amp;id=my%20work"></head></html>""",
            [License(abbr="by", version="4.0", location="meta_tag", in_head=True, in_footer=False,
                  tag='<meta content="https://creativecommons.org/licenses/by/4.0/?ref=chooser-v1&amp;id=my%20work" name="license"/>', # str(tag) keeps &amp;
                  left_context="<html><head>",
                  right_context="</head></html>")]
        ),
        # --- Real-World Like Examples with Context ---
        # Context: Copyright notice nearby. Note that the attributes are sorted after parsing (eg style is after src)
        (
            """<html><body><p>Copyright © 2024 My Site. All rights reserved.</p>
               <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">
               <img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" />
               </a><p>Content licensed under CC.</p></body></html>""",
            [License(abbr="by", version="4.0", location="a_tag", in_head=False, in_footer=False,
                  tag='<a href="http://creativecommons.org/licenses/by/4.0/" rel="license"><img alt="Creative Commons License" src="https://i.creativecommons.org/l/by/4.0/88x31.png" style="border-width:0"/></a>',
                  left_context="<html><body><p>Copyright © 2024 My Site. All rights reserved.</p>",
                  right_context="<p>Content licensed under CC.</p></body></html>")]
        ),
        # Context: License for specific icon set mentioned in text
        (
            """<div><p>The icons used on this page are from <a href="flaticon.com">Flaticon</a> and are licensed under <a href="https://creativecommons.org/licenses/by/3.0/">CC BY 3.0</a>. Thanks Flaticon!</p></div>""",
            [License(abbr="by", version="3.0", location="a_tag", in_head=False, in_footer=False,
                  tag='<a href="https://creativecommons.org/licenses/by/3.0/">CC BY 3.0</a>',
                  left_context='<html><body><div><p>The icons used on this page are from <a href="flaticon.com">Flaticon</a> and are licensed under ',
                  right_context=". Thanks Flaticon!</p></div></body></html>")]
        ),
        # Context: Photographer's work in a figure caption
        (
            """<figure>
            <img src="photo.jpg" alt="A beautiful landscape">
            <figcaption>Photograph by Jane Artist (<a href="https://creativecommons.org/licenses/by-nc/2.0/">CC BY-NC 2.0</a>). Taken in 2023.</figcaption>
            </figure>""",
            [License(abbr="by-nc", version="2.0", location="a_tag", in_head=False, in_footer=False,
                  tag='<a href="https://creativecommons.org/licenses/by-nc/2.0/">CC BY-NC 2.0</a>',
                  left_context='<html><body><figure><img alt="A beautiful landscape" src="photo.jpg"/><figcaption>Photograph by Jane Artist (',
                  right_context="). Taken in 2023.</figcaption></figure></body></html>")]
        ),
        # License in a data attribute (not standard, but people do weird things - should NOT be found by current selectors)
        (
            """<div data-license-url="https://creativecommons.org/licenses/by-sa/4.0/">Content</div>""",
            []
        ),
        # License URL as plain text (should NOT be found by current selectors)
        (
            """<p>This work is licensed under https://creativecommons.org/licenses/by/4.0/</p>""",
            []
        ),
        # License in a footer identified by id
        (
            """<html><body><div id="site-footer"><a href="https://creativecommons.org/licenses/by/4.0/">License</a></div></body></html>""",
            [License(abbr="by", version="4.0", location="a_tag", in_head=False, in_footer=True,
                  tag='<a href="https://creativecommons.org/licenses/by/4.0/">License</a>',
                  left_context='<html><body><div id="site-footer">',
                  right_context="</div></body></html>")]
        ),
    ],
)
def test_find_cc_licenses_in_html(html_input, expected_licenses):
    results = find_cc_licenses_in_html(html_input)

    assert len(results) == len(expected_licenses), (
        f"Expected {len(expected_licenses)} licenses, got {len(results)}. "
        f"Results: {results}, Expected: {expected_licenses} "
        f"for HTML: {html_input[:100]}..."
    )

    for i, (res, exp) in enumerate(zip(results, expected_licenses)):
        assert res.abbr == exp.abbr, f"Test {i} (HTML: {html_input[:50]}...): abbr mismatch. Got {res.abbr}, Exp {exp.abbr}"
        assert res.version == exp.version, f"Test {i} (HTML: {html_input[:50]}...): version mismatch. Got {res.version}, Exp {exp.version}"
        assert res.location == exp.location, f"Test {i} (HTML: {html_input[:50]}...): location mismatch. Got {res.location}, Exp {exp.location}"
        assert res.in_head == exp.in_head, f"Test {i} (HTML: {html_input[:50]}...): in_head mismatch. Got {res.in_head}, Exp {exp.in_head}"
        assert res.in_footer == exp.in_footer, f"Test {i} (HTML: {html_input[:50]}...): in_footer mismatch. Got {res.in_footer}, Exp {exp.in_footer}"
        assert res.tag == exp.tag, f"Test {i} (HTML: {html_input[:50]}...): tag mismatch. Got '{res.tag}', Exp '{exp.tag}'"
        assert res.left_context == exp.left_context, f"Test {i} (HTML: {html_input[:50]}...): left_context mismatch. Got '{res.left_context}', Exp '{exp.left_context}'"
        assert res.right_context == exp.right_context, f"Test {i} (HTML: {html_input[:50]}...): right_context mismatch. Got '{res.right_context}', Exp '{exp.right_context}'"

@pytest.mark.parametrize(
    "unsorted_licenses, expected_sorted_licenses",
    [
        # Single item
        (
            [License(abbr="by", version="3.0", location="meta_tag", in_head=True, in_footer=False, tag="<meta.../>", left_context="L", right_context="R")],
            [License(abbr="by", version="3.0", location="meta_tag", in_head=True, in_footer=False, tag="<meta.../>", left_context="L", right_context="R")]
        ),
        # Different locations (meta > json-ld > link > a)
        (
            [
                License(abbr="link", version="4.0", location="link_tag", in_head=True, in_footer=False, tag="<l/>", left_context="", right_context=""),
                License(abbr="meta", version="4.0", location="meta_tag", in_head=False,in_footer= False, tag="<m/>", left_context="", right_context=""),
                License(abbr="a", version="3.0", location="a_tag", in_head=False, in_footer=False, tag="<a/>", left_context="", right_context=""),
                License(abbr="json", version="1.0", location="json-ld", in_head=False, in_footer=False, tag="<j/>", left_context="", right_context=""),
            ],
            [
                License(abbr="meta", version="4.0", location="meta_tag", in_head=False, in_footer=False, tag="<m/>", left_context="", right_context=""),
                License(abbr="json", version="1.0", location="json-ld", in_head=False, in_footer=False, tag="<j/>", left_context="", right_context=""),
                License(abbr="link", version="4.0", location="link_tag", in_head=True, in_footer=False, tag="<l/>", left_context="", right_context=""),
                License(abbr="a", version="3.0", location="a_tag", in_head=False, in_footer=False, tag="<a/>", left_context="", right_context=""),
            ],
        ),
        # Same location, different head/footer (head=T > footer=T > others)
        (
            [
                License(abbr="link_F_T", version="3.0", location="link_tag", in_head=False, in_footer=True, tag="<ft/>", left_context="", right_context=""),
                License(abbr="link_T_F", version="4.0", location="link_tag", in_head=True, in_footer=False, tag="<h/>", left_context="", right_context=""),
                License(abbr="link_F_F", version="4.0", location="link_tag", in_head=False, in_footer=False, tag="<n/>", left_context="", right_context=""),
            ],
            [
                License(abbr="link_T_F", version="4.0", location="link_tag", in_head=True, in_footer=False, tag="<h/>", left_context="", right_context=""),
                License(abbr="link_F_T", version="3.0", location="link_tag", in_head=False, in_footer=True, tag="<ft/>", left_context="", right_context=""),
                License(abbr="link_F_F", version="4.0", location="link_tag", in_head=False, in_footer=False, tag="<n/>", left_context="", right_context=""),
            ],
        ),
        # Empty list
        ([], [])
    ],
)
def test_sort_licenses(unsorted_licenses, expected_sorted_licenses):
    assert sort_licenses(unsorted_licenses) == expected_sorted_licenses
