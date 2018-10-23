import os
import sys

siepic_package_path = os.path.join(os.path.dirname(__file__), '..', 'klayout_dot_config', 'python')
sys.path.append(siepic_package_path)

import SiEPIC

def test_first():
    pass