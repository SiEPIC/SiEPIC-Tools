import os as _os
import sys as _sys
import yaml as _yaml
import shutil as _shutil
from io import BytesIO as _BytesIO
from pathlib import Path as _Path
from zipfile import ZipFile as _ZipFile
from urllib.request import urlopen as _urlopen


def remove_library(library_name):
    """
    Removes an OPICS library.

    Args:
        library_name (str): The library name.
    """

    curr_dir = _Path(__file__).parent.resolve()

    # read _yaml file for available libraries in the catalogue
    with open(curr_dir / "catalogue.yaml", "r") as stream:
        lib_catalogue = _yaml.safe_load(stream)

    # if the library is installed
    if lib_catalogue[f"{library_name}"]["installed"] is True:

        # get local installation _Path
        library_dirpath = lib_catalogue[f"{library_name}"]["library_path"]

        # if the directory exists, remove it
        if _os.path.exists(library_dirpath):
            _shutil.rmtree(library_dirpath, ignore_errors=True)

        # update the _yaml data-entries
        lib_catalogue[f"{library_name}"]["installed"] = False
        lib_catalogue[f"{library_name}"]["library_path"] = ""

        # write the updated data-entries to the _yaml file
        with open(curr_dir / "catalogue.yaml", "w") as file:
            _yaml.dump(lib_catalogue, file)

        return True

    return False


def download_library(library_name="", library_url="", library_path=""):
    """
    Downloads OPICS libraries from GitHub.

    Args:
        library_name (str): The library name.
        library_url (str): The library_url link to download the library ZIP file.
        library_path (str): The folder to extract the installed library to.
    """

    curr_dir = _Path(__file__).parent.resolve()

    if library_url == "":
        return False

    # specify the directory to download and extract the library to
    if library_path == "":
        library_path = input(
            "Specify the _Path to download and extract the library to \n"
        )

        # the user can specify the current folder with a period '.'
        if library_path == ".":
            library_path = str(curr_dir)

    # if no input is provided
    if len(library_path) == 0:
        return False

    # create the folder if it does not exist
    if not _os.path.exists(library_path):
        _os.makedirs(library_path)

    # read _yaml file for available libraries in the catalogue
    with open(curr_dir / "catalogue.yaml", "r") as stream:
        lib_catalogue = _yaml.safe_load(stream)

    if lib_catalogue[f"{library_name}"]["installed"] is True:
        return True

    # download and extract the library to the folder, returns the dirpath with the extract foldername appended to the _Path
    library_dirpath = download_and_extract(library_url, library_path)

    if library_dirpath is False:
        print("library download failed.")
        return False

    lib_catalogue[library_name]["installed"] = True
    lib_catalogue[library_name]["library_path"] = library_dirpath

    # write the new data to the _yaml file
    with open(curr_dir / "catalogue.yaml", "w") as file:
        _yaml.dump(lib_catalogue, file)

    return True


def download_and_extract(library_url, library_path):
    """
    Downloads and extracts OPICS librarier from GitHub.

    Args:
        library_url (str): The library_url link to download the library ZIP file.
        library_path (str): The folder to extract the installed library to.

    """
    if library_url[:8] != "https://":
        return False

    with _urlopen(library_url) as Response:

        Length = Response.getheader("content-length")
        BlockSize = 1000000  # default blocksize value

        if Length:
            Length = int(Length)
            BlockSize = max(4096, Length // 20)

        BufferAll = _BytesIO()
        Size = 0
        print("Download start")
        while True:
            BufferNow = Response.read(BlockSize)
            if not BufferNow:
                break
            BufferAll.write(BufferNow)
            Size += len(BufferNow)

            if Length:
                Percent = int((Size / Length) * 100)
                print(f"download: {Percent}% {library_url}")
                _sys.stdout.flush()

        zip_file = _ZipFile(BufferAll)
        dl_folder_name = zip_file.namelist()[0]
        zip_file.extractall(library_path)
    print("Download finished.")

    return f"{library_path}/{dl_folder_name}"
