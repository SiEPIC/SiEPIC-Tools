'''
Test for SiEPIC.extend.radius_check

by Lukas Chrostowski, 2025
'''

def test_path_radius_check():
    import pya
    import SiEPIC
    
    # test path:
    pts = [[23.200,251.000], 
           [46.760,251.000],
           [46.760,256.240],
           [70.320,256.240]
    ]
    path = pya.Path(pts, 0.5e3)

    # this path does not work unless you include an SBend    
    assert path.radius_check(5, sbends=False) == False
    assert path.radius_check(5, sbends=True) == True

    # the first segment is too short
    pts = [[577.200,100.680], [621.100,100.680], [621.100, 269.000], [738.000, 269.000] ]
    path = pya.Path(pts, 0.5e3)
    assert path.radius_check(50, sbends=True) == False
    
if __name__ == "__main__":
    test_path_radius_check()
    