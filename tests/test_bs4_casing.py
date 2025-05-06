import pytest
from bs4 import BeautifulSoup

"""
Just some sanity checks to make sure that we do not have to run lower-casing on:
- tag names, e.g. <head>, <body>, <p>, <footer>
- attribute **names**, e.g. id, class, href, rel, src, alt, http-equiv

But we **do** have to run lower-casing on:
- attribute **values**, e.g. class names or content fields
"""


def test_tag_name_casing():
    """
    Tests that BeautifulSoup normalizes tag names to lowercase.
    """
    html_doc = "<HEAD><TITLE>Test Page</TITLE></HEAD><BODY><P>Hello</P><FOOTER>End</FOOTER></BODY>"
    soup = BeautifulSoup(html_doc, 'lxml')

    assert soup.head is not None, "Should find <HEAD> as soup.head"
    assert soup.head.name == "head", f"Expected 'head', got '{soup.head.name}'"

    assert soup.body is not None, "Should find <BODY> as soup.body"
    assert soup.body.name == "body", f"Expected 'body', got '{soup.body.name}'"

    assert soup.p is not None, "Should find <P> as soup.p"
    assert soup.p.name == "p", f"Expected 'p', got '{soup.p.name}'"

    assert soup.footer is not None, "Should find <FOOTER> as soup.footer"
    assert soup.footer.name == "footer", f"Expected 'footer', got '{soup.footer.name}'"

    # Search in lower-case should work fine
    mixed_case_head_tag = soup.find("head")
    assert mixed_case_head_tag is not None, "Should find tag regardless of query case"
    assert mixed_case_head_tag.name == "head", "Tag name should still be normalized"


def test_attribute_name_casing():
    """
    Tests that BeautifulSoup normalizes attribute names to lowercase,
    allowing access via lowercase keys.
    """
    html_doc = """
    <div ID="mainDiv" CLASS="container ExampleClass">
        <a HREF="/test-link" ReL="nofollow">Link</a>
        <img SRC="image.JPG" ALT="An Image">
        <meta HTTP-EQUIV="Content-Type" CONTENT="text/html;charset=UTF-8">
    </div>
    """
    soup = BeautifulSoup(html_doc, 'html.parser')

    div_tag = soup.find('div')
    assert div_tag is not None
    assert div_tag.get('id') == "mainDiv", "Should get 'ID' as 'id'"
    assert div_tag.get('class') == ['container', 'ExampleClass'], "Should get 'CLASS' as 'class' (using get)"
    assert div_tag['class'] == ['container', 'ExampleClass'], "Should get 'CLASS' as 'class' (using dict access)"
    assert 'CLASS' not in div_tag.attrs, "Original 'CLASS' should not be in attrs keys"
    assert 'class' in div_tag.attrs, "'class' should be in attrs keys"

    a_tag = soup.find('a')
    assert a_tag is not None
    assert a_tag.get('href') == "/test-link", "Should get 'HREF' as 'href'"
    assert a_tag['href'] == "/test-link", "Should get 'HREF' as 'href'"
    # rel returns a list of values
    assert a_tag.get('rel') == ["nofollow"], "Should get 'ReL' as 'rel'"

    img_tag = soup.find('img')
    assert img_tag is not None
    assert img_tag.get('src') == "image.JPG", "Should get 'SRC' as 'src'"
    assert img_tag.get('alt') == "An Image", "Should get 'ALT' as 'alt'"

    meta_tag = soup.find('meta')
    assert meta_tag is not None
    assert meta_tag.get('http-equiv') == "Content-Type", "Should get 'HTTP-EQUIV' as 'http-equiv'"
    assert meta_tag.get('content') == "text/html;charset=UTF-8", "Should get 'CONTENT' as 'content' (get access)"
    assert meta_tag['content'] == "text/html;charset=UTF-8", "Should get 'CONTENT' as 'content' (dict access)"


def test_attribute_value_casing():
    """
    Tests that BeautifulSoup preserves the original casing of attribute values.
    """
    html_doc = """
    <div class="MyClass AnotherCLASS MiXeDcAsE"></div>
    <input type="text" NAME="UserName" VALUE="TestUser123">
    <a data-customAttribute="PreserveThisValue">Link</a>
    """
    soup = BeautifulSoup(html_doc, 'html.parser')

    div_tag = soup.find('div')
    assert div_tag is not None
    # The 'class' attribute can have multiple values, returned as a list
    expected_classes = ["MyClass", "AnotherCLASS", "MiXeDcAsE"]
    actual_classes = div_tag.get('class')
    assert actual_classes == expected_classes, \
        f"Expected class values {expected_classes}, got {actual_classes}"

    input_tag = soup.find('input')
    assert input_tag is not None
    assert input_tag.get('name') == "UserName", "Attribute name 'NAME' should be accessible as 'name'"
    assert input_tag.get('value') == "TestUser123", \
        f"Expected value 'TestUser123', got '{input_tag.get('value')}'"

    a_tag = soup.find('a')
    assert a_tag is not None
    # BeautifulSoup normalizes attribute names with hyphens as well
    assert a_tag.get('data-customattribute') == "PreserveThisValue", \
        f"Expected value 'PreserveThisValue', got '{a_tag.get('data-customattribute')}'"

    # Ensure direct access to .attrs also preserves value casing
    assert div_tag.attrs['class'] == expected_classes
    assert input_tag.attrs['value'] == "TestUser123"
    assert a_tag.attrs['data-customattribute'] == "PreserveThisValue"
