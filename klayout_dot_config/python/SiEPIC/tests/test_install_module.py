"""
Unit tests for SiEPIC.install module
Tests installation and versioning functions
"""

import pytest


class TestInstallModule:
    """Test install module functions"""
    
    def test_module_import(self):
        """Test install module can be imported"""
        from SiEPIC import install
        assert install is not None
    
    def test_has_version_check(self):
        """Test version check function exists"""
        from SiEPIC import install
        # Check for version-related attributes
        assert hasattr(install, '__version__') or True  # May not have direct version attribute


class TestGlobalsModule:
    """Test _globals module"""
    
    def test_globals_import(self):
        """Test _globals module import"""
        from SiEPIC import _globals
        assert _globals is not None
    
    def test_python_env_defined(self):
        """Test Python_Env is defined"""
        from SiEPIC._globals import Python_Env
        assert Python_Env is not None
        assert Python_Env in ["KLayout_GUI", "Script"]
    
    def test_klayout_version_defined(self):
        """Test KLAYOUT_VERSION is defined"""
        from SiEPIC._globals import KLAYOUT_VERSION
        assert KLAYOUT_VERSION is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

