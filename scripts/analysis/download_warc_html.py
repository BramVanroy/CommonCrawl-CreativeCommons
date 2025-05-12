import requests
from datatrove.pipeline.readers.warc import process_record
from tqdm import tqdm
from warcio.archiveiterator import ArchiveIterator


def download_warc_file(file_url, local_file_path):
    """
    Downloads the WARC file from Common Crawl's HTTP URL and saves it locally.
    """
    response = requests.get(file_url, stream=True)
    if response.status_code == 200:
        total_size = int(response.headers.get("content-length", 0))  # Get the total file size

        with (
            open(local_file_path, "wb") as fhout,
            tqdm(
                desc=f"Downloading {local_file_path}",
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar,
        ):
            for chunk in response.iter_content(chunk_size=8192):
                fhout.write(chunk)
                pbar.update(len(chunk))
        print(f"Downloaded WARC file to {local_file_path}")
    else:
        raise Exception(f"Failed to download WARC file: HTTP {response.status_code}")


def extract_html_from_warc(warc_file_path, target_uuid):
    """
    Extracts the HTML content of the record with the specified UUID from the WARC file.
    """
    with open(warc_file_path, "rb") as warc_file:
        for record in ArchiveIterator(warc_file):
            if record.rec_type == "response":  # Process only HTTP response records
                # Check if the UUID matches
                if record.rec_headers.get_header("WARC-Record-ID") == target_uuid:
                    # Extract and return the text (HTML content)
                    return process_record(record)["text"]
    return None


def main(
    target_uuid: str = "<urn:uuid:d5a13e0d-1197-4f6e-bbc7-fedee55980ae>",
    warc_path: str = "crawl-data/CC-MAIN-2013-20/segments/1368702730377/warc/CC-MAIN-20130516111210-00038-ip-10-60-113-184.ec2.internal.warc.gz",
    tmp_local_warc_file="downloaded_warc_file.warc.gz",
):
    file_url = f"https://data.commoncrawl.org/{warc_path}"

    try:
        # Step 1: Download the WARC file
        download_warc_file(file_url, tmp_local_warc_file)

        # Step 2: Extract the HTML content for the specific UUID
        print(f"Extracting HTML content for UUID {target_uuid}...")
        html_content = extract_html_from_warc(tmp_local_warc_file, target_uuid)

        if html_content:
            # Step 3: Save or display the extracted HTML
            output_file = "extracted.html"
            with open(output_file, "w", encoding="utf-8") as fhout:
                fhout.write(html_content)
            print(f"HTML content extracted and saved to {output_file}")
        else:
            print(f"The target UUID {target_uuid} was not found in the WARC file.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
