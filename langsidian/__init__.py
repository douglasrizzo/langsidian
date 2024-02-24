from pathlib import Path

import requests
from tqdm import tqdm


def download_with_progress(url: str | Path, filepath: Path) -> None:
  """Download a file from the given URL to the specified filepath while displaying the download progress.

  From https://stackoverflow.com/a/37573701/1245214.

  Args:
      url (str | Path): The URL of the file to download.
      filepath (Path): The filepath to save the downloaded file.
  """
  # Sizes in bytes.
  response = requests.get(url, stream=True)
  total_size = int(response.headers.get("content-length", 0))
  block_size = 1024

  with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar, filepath.open("wb") as file:
    for data in response.iter_content(block_size):
      progress_bar.update(len(data))
      file.write(data)
