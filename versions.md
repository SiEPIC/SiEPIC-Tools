# SiEPIC-Tools version history

v0.3.11
* adding generic silicon photonics (GSiP) technology to this package; removing from SiEPIC-EBeam-PDK
* updates to the ring modulator transceiver example
* electrical pins can now be either box or simple_polygon
* version number in the menu title

v0.3.10
* error checking and improvements to the Lumerical tool integration

v0.3.9
* moved missing Python libraries (requests, urllib3, certifi, chardet, idna) from Windows_Python_packages_for_KLayout to klayout folder, since OSX is also missing these libraries by default (though they can be easily installed with "pip").
* support for multiple technologies for INTERCONNECT CML library registration and simulations, for KLayout 0.24

