"""
Test for SiEPIC.utils.geometry.box_bezier_corners

by Lukas Chrostowski 2024

"""

def test_box_bezier_corners():
    '''
    Draw a box, measure the area, and check
    '''

    import pya
    import SiEPIC
    from SiEPIC.utils.geometry import box_bezier_corners

    from packaging import version
    if version.parse(SiEPIC.__version__) < version.parse("0.5.4"):
        raise Exception("Errors", "This example requires SiEPIC-Tools version 0.5.4 or greater.")

    '''
    Create a box with rounded corners
    '''    
    width, height = 100, 100
    dt_bezier_corner = 0.1
    dbu = 0.001
    polygon = box_bezier_corners(width, height, dt_bezier_corner, accuracy = 0.001).to_itype(dbu).transformed(pya.Trans(-width/2,0))
    
    # return polygon.area(), polygon.num_points()
    assert polygon.area() == 9937944371
    assert polygon.num_points() == 844
    
if __name__ == "__main__":
    test_box_bezier_corners()
