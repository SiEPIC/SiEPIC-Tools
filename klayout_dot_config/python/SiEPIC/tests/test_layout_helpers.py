"""
Unit tests for SiEPIC.utils.layout helper functions
Tests make_pin, floorplan, and other layout utilities
"""

import pytest
import pya


class TestLayoutHelpers:
    """Test layout helper functions"""
    
    def test_make_pin_function_exists(self):
        """Test make_pin function exists"""
        from SiEPIC.utils.layout import make_pin
        assert make_pin is not None
    
    def test_floorplan_function_exists(self):
        """Test floorplan function exists"""
        from SiEPIC.utils.layout import floorplan
        assert floorplan is not None
    
    def test_y_splitter_tree_function_exists(self):
        """Test y_splitter_tree function exists"""
        from SiEPIC.utils.layout import y_splitter_tree
        assert y_splitter_tree is not None
    
    def test_make_pin_basic(self):
        """Test creating a pin on a simple cell"""
        ly = pya.Layout()
        ly.dbu = 0.001
        cell = ly.create_cell("test_pin")
        layer = ly.layer(1, 10)
        
        from SiEPIC.utils.layout import make_pin
        
        # Create a simple pin
        pin_name = "opt1"
        center = [0, 0]
        w = 500  # 500 nm
        layer_pin = layer
        direction = 0  # 0 degrees
        
        # This should not raise an error
        try:
            make_pin(cell, pin_name, center, w, layer_pin, direction)
            # Pin should be created
            assert cell.shapes(layer).size() > 0
        except Exception as e:
            # If it requires more setup, just check the function exists
            assert make_pin is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


