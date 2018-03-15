# coding: utf-8
"""

"""

import ctypes
import os
import sys


def required_runtime_version():
    # TODO: load runtime version from file
    return '0.28.1'


def _version_string_to_tuple(version_string):
    if not isinstance(version_string, str):
        version_string = version_string.decode('utf-8')
    if version_string.startswith('v'):
        version_string = version_string[1:]
    if '-' in version_string:
        version_string = version_string.split('-', 1)[0]
    version_string = version_string.replace('.post', '.')
    return tuple(int(v) for v in version_string.split('.', 3))


def load_runtime(search_dirs=[], silent=False):
    if sys.platform == "win32":
        library_extension = ".dll"
    else:
        library_extension = ".so"

    search_directories = list(search_dirs)
    search_directories.extend([
        os.environ.get('GRLIB'),
        os.path.realpath(os.path.dirname(__file__)),
        os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'build', 'lib', 'gr')),
    ])
    if sys.platform != "win32":
        search_directories.extend(
            [
                os.path.join(os.path.expanduser('~'), 'gr', 'lib'),
                '/usr/local/gr/lib',
                '/usr/gr/lib',
            ]
        )

    search_path = os.environ.get('PATH', '')
    for directory in search_directories:
        if directory is None:
            continue
        if not os.path.isdir(directory):
            continue
        directory = os.path.abspath(directory)
        library_filename = os.path.join(directory, 'libGR' + library_extension)
        if os.path.isfile(library_filename):
            if sys.platform == "win32":
                os.environ["PATH"] = search_path + ";" + directory
            try:
                library = ctypes.CDLL(library_filename)
            except OSError:
                # library exists but could not be loaded (e.g. due to missing dependencies)
                if silent:
                    break
                else:
                    raise
            library.gr_version.argtypes = []
            library.gr_version.restype = ctypes.c_char_p
            library_version_string = library.gr_version()
            library_version = _version_string_to_tuple(library_version_string)
            required_version = _version_string_to_tuple(required_runtime_version())
            version_compatible = library_version[0] == required_version[0] and library_version >= required_version
            if version_compatible:
                return library
    if not silent:
        sys.stderr.write("""GR runtime not found.
Please visit https://gr-framework.org and install at least the following version of the GR runtime:
{}

Also, please ensure that you have all required dependencies:
Debian/Ubuntu: apt install libxt6 libxrender1
CentOS 7: yum install libXt libXrender libXext
Fedora 26: dnf install -y libXt libXrender libXext
openSUSE 42.3: zypper install -y libXt6 libXrender1 libXext6
""".format(required_runtime_version()))
    return None


def register_gksterm():
    if sys.platform == 'darwin':
        # register GKSTerm.app on macOS
        app = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'GKSTerm.app'))
        os.system('/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -f {}'.format(app))
