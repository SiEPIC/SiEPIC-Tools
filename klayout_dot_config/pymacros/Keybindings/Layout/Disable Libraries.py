# $description: Disable Libraries
# $show-in-menu
# $menu-path: siepic_menu.layout.end
import pya

def disable_libraries():
    print('Disabling KLayout libraries')
    for l in pya.Library().library_ids():
        print(' - %s' % pya.Library().library_by_id(l).name())
        pya.Library().library_by_id(l).delete()

disable_libraries()
