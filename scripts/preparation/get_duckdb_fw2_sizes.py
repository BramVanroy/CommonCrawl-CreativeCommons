import json

import requests


repo_id = "BramVanroy/fineweb-2-duckdbs"
api_url = f"https://huggingface.co/api/datasets/{repo_id}?blobs=true"
request_timeout_seconds = 30


total_size_bytes = 0
duckdb_files_info = []

print(f"Fetching file list and sizes from API: {api_url}")

try:
    response = requests.get(api_url, timeout=request_timeout_seconds)
    response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)

    repo_data = response.json()

    if "siblings" in repo_data and isinstance(repo_data["siblings"], list):
        for file_info in repo_data["siblings"]:
            if "rfilename" in file_info and "size" in file_info:
                if file_info["rfilename"].endswith(".duckdb") and "_removed" not in file_info["rfilename"]:
                    file_name = file_info["rfilename"]
                    file_size = file_info["size"]  # This should be in bytes as per the API

                    if isinstance(file_size, (int, float)) and file_size >= 0:
                        duckdb_files_info.append({"name": file_name, "size": file_size})
                        total_size_bytes += file_size
                    else:
                        print(f"Warning: Invalid or missing size for {file_name}: {file_size}. Skipping.")
            else:
                print(f"Warning: Skipping sibling due to missing 'rfilename' or 'size': {file_info}")
    else:
        print("Error: 'siblings' key not found in API response or is not a list.")
        print("API Response snippet:", response.text[:500])  # Print a snippet for debugging

except requests.exceptions.Timeout:
    print(f"Error: The request to {api_url} timed out after {request_timeout_seconds} seconds.")
except requests.exceptions.HTTPError as http_err:
    print(f"Error: HTTP error occurred: {http_err} - {response.status_code}")
    print("API Response:", response.text)
except requests.exceptions.RequestException as req_err:
    print(f"Error: A network request error occurred: {req_err}")
except json.JSONDecodeError:
    print("Error: Failed to decode JSON response from the API.")
    print("API Response snippet:", response.text[:500])
except Exception as e:
    print(f"An unexpected error occurred: {e}")


print("\n--- Summary ---")
if duckdb_files_info:
    print(f"Found {len(duckdb_files_info)} not-removed 'duckdb' files:")
    for f_info in duckdb_files_info:
        print(f"- {f_info['name']}: {f_info['size']} bytes")

    print(f"\nTotal size of these not-removed duckdb files: {total_size_bytes} bytes")
    if total_size_bytes > 0:
        gb_size = total_size_bytes / (1024**3)
        print(f"Which is approximately: {gb_size:.2f} GB")
else:
    print("No not-removed duckdb files found or an error occurred preventing processing.")
