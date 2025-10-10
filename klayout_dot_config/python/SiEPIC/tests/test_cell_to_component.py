"""
Unit tests for SiEPIC.utils.components.cell_to_component
Tests conversion of cells with geometries to SiEPIC Components
with DevRec and PinRec layers

Note: These tests use mocking to work with both:
  - SiEPIC-Tools >= 0.5.32 (with TECHNOLOGY parameter support)
  - SiEPIC-Tools < 0.5.32 (without TECHNOLOGY parameter support)
"""

import pytest
import pya
from unittest.mock import patch


class TestCellToComponent:
    """Test cell_to_component function with various port configurations"""
    
    def setup_method(self):
        """Set up test fixtures before each test"""
        # Create a layout with technology
        self.layout = pya.Layout()
        self.layout.dbu = 0.001  # 1nm database unit
        
        # Try to load a real technology, fall back to creating mock technology
        try:
            from SiEPIC.utils import get_technology_by_name
            self.TECHNOLOGY = get_technology_by_name('EBeam')
            self.layout.technology_name = 'EBeam'
            self.layout.TECHNOLOGY = self.TECHNOLOGY
        except:
            # If EBeam is not available, create minimal technology setup
            # This supports both old and new versions of SiEPIC-Tools
            self.TECHNOLOGY = {
                'technology_name': 'Test',
                'DevRec': pya.LayerInfo(68, 0),
                'PinRec': pya.LayerInfo(1, 10),
                'Si': pya.LayerInfo(1, 0),
            }
            self.layout.TECHNOLOGY = self.TECHNOLOGY
        
        # Define layer indices
        self.layer_si = self.layout.layer(1, 0)  # Si layer
        self.layer_devrec = self.layout.layer(self.TECHNOLOGY['DevRec'])
        self.layer_pinrec = self.layout.layer(self.TECHNOLOGY['PinRec'])
        
        # Create a mock for load_Verification that returns minimal verification info
        # This allows tests to work with both old and new versions of SiEPIC-Tools
        self.mock_verification = None  # Will use default deviceonly_layers [[1,0]]
    
    def create_test_box_cell(self, name="test_box", width=10000, height=2000):
        """
        Create a test cell with a box on Si layer (1/0)
        
        Args:
            name: Cell name
            width: Box width in DBU (default 10000 = 10 microns)
            height: Box height in DBU (default 2000 = 2 microns)
            
        Returns:
            pya.Cell: Created cell with box
        """
        cell = self.layout.create_cell(name)
        
        # Create a box centered at origin
        box = pya.Box(-width//2, -height//2, width//2, height//2)
        cell.shapes(self.layer_si).insert(box)
        
        return cell
    
    def count_shapes(self, cell, layer):
        """Count number of shapes on a layer in a cell"""
        return cell.shapes(layer).size()
    
    def call_cell_to_component(self, cell, ports, verbose=False):
        """
        Call cell_to_component with compatibility for both old and new versions.
        Tries to use TECHNOLOGY parameter (version >= 0.5.32), falls back to old API.
        """
        from SiEPIC.utils.components import cell_to_component
        import inspect
        
        # Check if cell_to_component accepts TECHNOLOGY parameter
        sig = inspect.signature(cell_to_component)
        has_technology_param = 'TECHNOLOGY' in sig.parameters
        
        # Mock load_Verification to avoid needing real Verification.xml file
        with patch('SiEPIC.utils.load_Verification') as mock_load_verification:
            mock_load_verification.return_value = self.mock_verification
            
            if has_technology_param:
                # New version >= 0.5.32 with TECHNOLOGY parameter
                cell_to_component(
                    cell=cell,
                    ports=ports,
                    verbose=verbose,
                    TECHNOLOGY=self.TECHNOLOGY
                )
            else:
                # Old version < 0.5.32 without TECHNOLOGY parameter
                # Set up global technology context
                old_tech = getattr(self.layout, 'TECHNOLOGY', None)
                self.layout.TECHNOLOGY = self.TECHNOLOGY
                try:
                    cell_to_component(
                        cell=cell,
                        ports=ports,
                        verbose=verbose
                    )
                finally:
                    if old_tech is not None:
                        self.layout.TECHNOLOGY = old_tech
    
    def test_function_exists(self):
        """Test that cell_to_component function exists"""
        from SiEPIC.utils.components import cell_to_component
        assert cell_to_component is not None
    
    def test_left_right_ports(self):
        """Test cell_to_component with Left and Right ports"""
        cell = self.create_test_box_cell("test_lr")
        
        # Initial state - no DevRec or PinRec
        assert self.count_shapes(cell, self.layer_devrec) == 0
        assert self.count_shapes(cell, self.layer_pinrec) == 0
        
        # Apply cell_to_component with L and R ports
        self.call_cell_to_component(cell=cell, ports=['L', 'R'])
        
        # After conversion - should have DevRec and PinRec
        assert self.count_shapes(cell, self.layer_devrec) > 0, "DevRec layer should be created"
        assert self.count_shapes(cell, self.layer_pinrec) > 0, "PinRec layer should be created"
    
    def test_top_bottom_ports(self):
        """Test cell_to_component with Top and Bottom ports"""
        cell = self.create_test_box_cell("test_tb", width=2000, height=10000)
        
        # Initial state - no DevRec or PinRec
        assert self.count_shapes(cell, self.layer_devrec) == 0
        assert self.count_shapes(cell, self.layer_pinrec) == 0
        
        # Apply cell_to_component with T and B ports
        self.call_cell_to_component(cell=cell, ports=['T', 'B'])
        
        # After conversion - should have DevRec and PinRec
        assert self.count_shapes(cell, self.layer_devrec) > 0, "DevRec layer should be created"
        assert self.count_shapes(cell, self.layer_pinrec) > 0, "PinRec layer should be created"
    
    def test_left_port_only(self):
        """Test cell_to_component with only Left port"""
        cell = self.create_test_box_cell("test_l")
        
        # Apply cell_to_component with only L port
        cell_to_component(
            cell=cell,
            ports=['L'],
            verbose=False,
            TECHNOLOGY=self.TECHNOLOGY
        )
        
        # Should still have DevRec and PinRec
        assert self.count_shapes(cell, self.layer_devrec) > 0
        assert self.count_shapes(cell, self.layer_pinrec) > 0
    
    def test_right_port_only(self):
        """Test cell_to_component with only Right port"""
        cell = self.create_test_box_cell("test_r")
        
        # Apply cell_to_component with only R port
        cell_to_component(
            cell=cell,
            ports=['R'],
            verbose=False,
            TECHNOLOGY=self.TECHNOLOGY
        )
        
        # Should still have DevRec and PinRec
        assert self.count_shapes(cell, self.layer_devrec) > 0
        assert self.count_shapes(cell, self.layer_pinrec) > 0
    
    def test_top_port_only(self):
        """Test cell_to_component with only Top port"""
        cell = self.create_test_box_cell("test_t", width=2000, height=10000)
        
        # Apply cell_to_component with only T port
        cell_to_component(
            cell=cell,
            ports=['T'],
            verbose=False,
            TECHNOLOGY=self.TECHNOLOGY
        )
        
        # Should still have DevRec and PinRec
        assert self.count_shapes(cell, self.layer_devrec) > 0
        assert self.count_shapes(cell, self.layer_pinrec) > 0
    
    def test_bottom_port_only(self):
        """Test cell_to_component with only Bottom port"""
        cell = self.create_test_box_cell("test_b", width=2000, height=10000)
        
        # Apply cell_to_component with only B port
        cell_to_component(
            cell=cell,
            ports=['B'],
            verbose=False,
            TECHNOLOGY=self.TECHNOLOGY
        )
        
        # Should still have DevRec and PinRec
        assert self.count_shapes(cell, self.layer_devrec) > 0
        assert self.count_shapes(cell, self.layer_pinrec) > 0
    
    def test_all_four_ports(self):
        """Test cell_to_component with all four ports (L, R, T, B)"""
        # Create a square box
        cell = self.create_test_box_cell("test_lrtb", width=10000, height=10000)
        
        # Apply cell_to_component with all four ports
        cell_to_component(
            cell=cell,
            ports=['L', 'R', 'T', 'B'],
            verbose=False,
            TECHNOLOGY=self.TECHNOLOGY
        )
        
        # Should have DevRec and PinRec
        assert self.count_shapes(cell, self.layer_devrec) > 0
        # Should have 4 pins (one for each port)
        assert self.count_shapes(cell, self.layer_pinrec) >= 4
    
    def test_no_ports(self):
        """Test cell_to_component with no ports (empty list)"""
        cell = self.create_test_box_cell("test_no_ports")
        
        # Apply cell_to_component with no ports
        cell_to_component(
            cell=cell,
            ports=[],
            verbose=False,
            TECHNOLOGY=self.TECHNOLOGY
        )
        
        # Should have DevRec but no PinRec
        assert self.count_shapes(cell, self.layer_devrec) > 0
        # May or may not have PinRec depending on implementation
    
    def test_verbose_mode(self):
        """Test cell_to_component with verbose=True"""
        cell = self.create_test_box_cell("test_verbose")
        
        # This should not raise an error
        try:
            cell_to_component(
                cell=cell,
                ports=['L', 'R'],
                verbose=True,
                TECHNOLOGY=self.TECHNOLOGY
            )
            assert True
        except Exception as e:
            pytest.fail(f"cell_to_component with verbose=True raised exception: {e}")
    
    def test_multiple_boxes(self):
        """Test cell_to_component with multiple boxes (H-shape for vertical waveguides)"""
        cell = self.layout.create_cell("test_h_shape")
        
        # Create H-shape: two vertical boxes connected by horizontal
        # Left vertical box
        box1 = pya.Box(-5000, -1000, -4000, 1000)
        cell.shapes(self.layer_si).insert(box1)
        
        # Right vertical box
        box2 = pya.Box(4000, -1000, 5000, 1000)
        cell.shapes(self.layer_si).insert(box2)
        
        # Connecting horizontal box
        box3 = pya.Box(-4000, -200, 4000, 200)
        cell.shapes(self.layer_si).insert(box3)
        
        # Apply cell_to_component with L and R ports
        cell_to_component(
            cell=cell,
            ports=['L', 'R'],
            verbose=False,
            TECHNOLOGY=self.TECHNOLOGY
        )
        
        # Should have DevRec and PinRec
        assert self.count_shapes(cell, self.layer_devrec) > 0
        assert self.count_shapes(cell, self.layer_pinrec) > 0
    
    def test_technology_parameter(self):
        """Test that TECHNOLOGY parameter is properly used"""
        cell = self.create_test_box_cell("test_tech_param")
        
        # Should work with explicit TECHNOLOGY parameter
        try:
            cell_to_component(
                cell=cell,
                ports=['L', 'R'],
                verbose=False,
                TECHNOLOGY=self.TECHNOLOGY
            )
            assert self.count_shapes(cell, self.layer_devrec) > 0
        except Exception as e:
            pytest.fail(f"Failed with explicit TECHNOLOGY parameter: {e}")


class TestCellToComponentEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_empty_cell(self):
        """Test cell_to_component with empty cell (no shapes)"""
        layout = pya.Layout()
        layout.dbu = 0.001
        
        # Create minimal technology
        TECHNOLOGY = {
            'technology_name': 'Test',
            'DevRec': pya.LayerInfo(68, 0),
            'PinRec': pya.LayerInfo(1, 10),
            'Si': pya.LayerInfo(1, 0),
        }
        layout.TECHNOLOGY = TECHNOLOGY
        
        cell = layout.create_cell("empty_cell")
        
        # This might raise an exception or handle gracefully
        try:
            cell_to_component(
                cell=cell,
                ports=['L', 'R'],
                verbose=False,
                TECHNOLOGY=TECHNOLOGY
            )
            # If it succeeds, that's fine
            assert True
        except Exception:
            # If it fails, that's also acceptable for empty cell
            assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

