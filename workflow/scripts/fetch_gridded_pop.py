"""
Fetch gridded world population data for China from World Pop Project

Data Reference
Bondarenko M. et al. Constrained estimates of 2015-2030 total number of people per grid square 
at a resolution of 30 arc (approximately 1km at the equator) R2025A version v1. Global Demographic Data Project
 - Funded by The Bill and Melinda Gates Foundation (INV-045237). 
 WorldPop - School of Geography and Environmental Science, University of Southampton. DOI:10.5258/SOTON/WP00840
"""

import logging
import os

import requests
from _helpers import configure_logging, mock_snakemake
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def find_worldpop_url(url):
    """Explore the contents of a WorldPop directory to find available files."""

    logger.info(f"Searching WorldPop directory: {url}")
    try:
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
        links = soup.find_all('a', href=True)

        files = []
        for link in links:
            href = link['href']
            if href.endswith('.tif') or href.endswith('.zip'):
                files.append(href)

        return files
    except Exception as e:
        print(f"Error exploring directory: {e}")
        return []


def download_world_pop(base_url: str, filename: str, output_path: str = "worldpop_2024_data.tif") -> str:
    """Download the population per pixel raster data from WorldPop.
    
    Args:
        base_url (str): Base URL for the WorldPop data directory
        filename (str): Name of the file to download.
        output_path (os.PathLike, Optional): Directory to save the downloaded file. Defaults to "worldpop_2024_data.tif".
        
    Returns:
        Path to the downloaded file, or None if download failed
        
    Raises:
        requests.RequestException: If the download request fails
        IOError: If file writing fails
    """
    url = base_url + filename

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Check if file already exists
    if os.path.exists(output_path):
        print(f"File already exists: {output_path}")
        return output_path

    logger.info(f"Downloading {filename}...")
    logger.info(f"URL: {url} ")

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Get file size for progress tracking
        total_size = int(response.headers.get('content-length', 0))
        logger.info(f"File size: {total_size / (1024*1024):.1f} MB")

        with open(output_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        logger.info(f"\rProgress: {progress:.1f}%")

        logger.info(f"Downloaded: {output_path}")

    except requests.RequestException as e:
        logger.info(f"Download failed (network error): {e}")
        return None
    except OSError as e:
        logger.info(f"Download failed (file error): {e}")
        return None
    except Exception as e:
        logger.info(f"Download failed (unexpected error): {e}")
        return None

    return output_path


if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if "snakemake" not in globals():
        snakemake = mock_snakemake("fetch_gridded_population")
    configure_logging(snakemake, logger=logger)

    base_url = snakemake.params.base_url

    available_files = find_worldpop_url(base_url)

    if not available_files:
        raise ValueError(f"No world population files found at {base_url}")

    logger.info(f"Found {len(available_files)} file(s): \n\t{'\n\t'.join(available_files)}. First will be downloaded.")

    # Download the data & save it
    china_2024_file = download_world_pop(base_url, available_files[0], snakemake.output.pop_raster)

    if china_2024_file is None:
        logger.info("Failed to download the data file. Please check the URL and try again.")
    else:
        logger.info(f"Successfully downloaded: {china_2024_file}")
