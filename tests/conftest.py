import os
import sys

# make an environment that works with either klayout or python
import lygadgets
lygadgets.patch_environment()

# point to the SiEPIC package because it is not deployed and installed in system python
if not lygadgets.isGSI():
    siepic_package_path = os.path.join(os.path.dirname(__file__), '..', 'klayout_dot_config', 'python')
    sys.path.append(siepic_package_path)

# specify directories for XOR test files
import lytest
test_root = os.path.dirname(__file__)
lytest.utest_buds.test_root = test_root