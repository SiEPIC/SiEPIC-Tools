"""
Unit tests for commonly used functions in SiEPIC.scripts module
"""

import pytest
import pya
from SiEPIC._globals import Python_Env


class TestScriptsModule:
    """Test SiEPIC.scripts module functions"""
    
    def test_module_import(self):
        """Test scripts module can be imported"""
        from SiEPIC import scripts
        assert scripts is not None
    
    def test_zoom_out_function_exists(self):
        """Test zoom_out function exists"""
        from SiEPIC import scripts
        assert hasattr(scripts, 'zoom_out')
    
    def test_export_layout_function_exists(self):
        """Test export_layout function exists"""
        from SiEPIC import scripts
        assert hasattr(scripts, 'export_layout')
    
    def test_connect_pins_function_exists(self):
        """Test connect_pins_with_waveguide function exists"""
        from SiEPIC import scripts
        assert hasattr(scripts, 'connect_pins_with_waveguide')
    
    def test_connect_cell_function_exists(self):
        """Test connect_cell function exists"""
        from SiEPIC import scripts
        assert hasattr(scripts, 'connect_cell')


class TestLayoutUtilityFunctions:
    """Test layout utility functions"""
    
    def test_instantiate_all_library_cells_exists(self):
        """Test instantiate_all_library_cells function exists"""
        from SiEPIC.scripts import instantiate_all_library_cells
        assert instantiate_all_library_cells is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


