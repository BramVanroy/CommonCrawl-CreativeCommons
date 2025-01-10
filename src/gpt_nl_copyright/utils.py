import dataclasses

from datatrove.data import Document


def prepare_for_writing(self, document: Document, output_text: bool = True, output_html: bool = False) -> dict:
    """
    Potentially remove text and/or html from the document before writing to disk. The 'self' is needed
    because this function is passed to the JsonlWriter as an adapter where it will be turned into a method.
    Args:
        document: document to format
        output_text: whether to include the text in the output
        output_html: whether to include the html in the output

    Returns: a dictionary to write to disk

    """
    data = {key: val for key, val in dataclasses.asdict(document).items() if val}
    # Expand metadata into its own columns
    data |= data.pop("metadata")

    if not output_text:
        data.pop("text", None)
    if not output_html:
        data.pop("html", None)

    return data
