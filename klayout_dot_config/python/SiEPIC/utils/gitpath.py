"""All credit goes to https://github.com/MaxNoe/python-gitpath


The MIT License (MIT)

Copyright (c) 2015 Maximilian Nöthe

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

from subprocess import check_output, CalledProcessError
from functools import lru_cache
import os.path


@lru_cache(maxsize=1)
def root():
    ''' returns the absolute path of the repository root '''
    try:
        base = check_output(['git', 'rev-parse', '--show-toplevel'])
    except CalledProcessError:
        raise IOError(f"'{os.getcwd()}' is not a git repository")
    return base.decode('utf-8').strip()


def abspath(relpath):
    ''' returns the absolute path for a path given relative to the root of
    the git repository
    '''
    return os.path.join(root(), relpath)
