[![pypi](https://img.shields.io/pypi/v/SiEPIC)](https://pypi.org/project/SiEPIC/)
[![codecov](https://codecov.io/gh/SiEPIC/SiEPIC-Tools/graph/badge.svg?token=MD7OI5BGZ3)](https://codecov.io/gh/SiEPIC/SiEPIC-Tools)
[![issues](https://img.shields.io/github/issues/SiEPIC/SiEPIC-Tools)](https://github.com/SiEPIC/SiEPIC-Tools/issues)
![forks](https://img.shields.io/github/forks/SiEPIC/SiEPIC-Tools)
![Stars](https://img.shields.io/github/stars/SiEPIC/SiEPIC-Tools)


# SiEPIC-Tools

- <a href="http://www.siepic.ubc.ca">SiEPIC</a>-Tools - for silicon photonics layout, design, verification and circuit simulation
- Developed by <a href="https://ca.linkedin.com/in/chrostowski">Lukas Chrostowski</a>, with contributions by: <a href="https://ca.linkedin.com/in/zeqin-lu-13a52394">Zeqin Lu</a>, <a href="https://uk.linkedin.com/in/jaspreet-jhoja-00a56b64">Jaspreet Jhoja</a>, <a href="https://www.linkedin.com/in/xu-photonics/">Xu Wang</a>, <a href="https://ca.linkedin.com/in/jonas-flückiger-92a4831">Jonas Flueckiger</a>, and <a href="https://github.com/SiEPIC/SiEPIC-Tools/graphs/contributors">others</a>.
- This is a package implemented using Python in <a href="http://www.klayout.de">KLayout</a>.
- Instruction on design, layout, fabrication, test, data analysis for silicon photonics provided in the edX course: <a href="http://edx.org/course/silicon-photonics-design-fabrication-ubcx-phot1x">Silicon Photonics Design, Fabrication and Data Analysis</a> and textbook <a href="http://www.cambridge.org/ca/academic/subjects/engineering/electronic-optoelectronic-devices-and-nanotechnology/silicon-photonics-design-devices-systems">Silicon Photonics Design: From Devices to Systems</a> by Lukas Chrostowski and Michael Hochberg.
- Fabrication runs via Electron Beam Lithography are available, including <a href="https://ebeam.mff.uw.edu/ebeamweb/news/projects/projects/silicon_photonics_1.html">the University of Washington</a>, <a href="https://www.appliednt.com/nanosoi-fabrication-service">Applied Nanotools Inc.</a>, and <a href="https://www.siepic.ca/fabrication">SiEPICfab</a>.
- Process Design Kits that use KLayout SiEPIC-Tools are available for multiple foundries including <a href="https://www.cmc.ca/amf-silicon-photonics-general-purpose/">AMF</a>, <a href="https://www.aimphotonics.com/">AIM Photonics</a>.
- Citing this work:  
  - Lukas Chrostowski, Zeqin Lu, Jonas Flueckiger, Xu Wang, Jackson Klein, Amy Liu, Jaspreet Jhoja, James Pond,
"<a href=https://doi.org/10.1117/12.2230376>Design and simulation of silicon photonic schematics and layouts</a>," Proc. SPIE 9891, Silicon Photonics and Photonic Integrated Circuits V, 989114 (May 13, 2016); doi:10.1117/12.2230376.
  - Lukas Chrostowski, Hossam Shoman, Mustafa Hammood, Han Yun,  Jaspreet Jhoja, Enxiao Luan,  Stephen Lin, Ajay Mistry, Donald Witt, Nicolas A. F. Jaeger, Sudip Shekhar,  Hasitha Jayatilleka, Philippe Jean, Simon B.-de Villers, Jonathan Cauchon, Wei Shi,  Cameron Horvath, Jocelyn N. Westwood-Bachman, Kevin Setzer, Mirwais Aktary, N. Shane Patrick, Richard Bojko, Amin Khavasi, Xu Wang, Thomas Ferreira de Lima,  Alexander N. Tait, Paul R. Prucnal, David E. Hagan, Doris Stevanovic, Andy P. Knights, "<a href="https://doi.org/10.1109/JSTQE.2019.2917501">Silicon Photonic Circuit Design Using Rapid Prototyping Foundry Process Design Kits</a>" IEEE Journal of Selected Topics in Quantum Electronics, Volume: 25, Issue: 5, Sept.-Oct. 2019. (<a href="https://www.dropbox.com/s/i1z4ackr3q7fz1l/2019_JSTQE_foundry.pdf?dl=1">PDF</a>)

## Download and Installation instructions:
 - in KLayout (version 0.27 or greater, preferably 0.29), use Tools | Package Manager, and find SiEPIC-Tools there (more details in the [Wiki Instructions](https://github.com/SiEPIC/SiEPIC-Tools/wiki/Installation))
 - install PDK, e.g., <a href="https://github.com/siepic/SiEPIC_EBeam_PDK/wiki/Installation-instructions">SiEPIC_EBeam_PDK download and installation instructions</a> on the wiki page.  

 
## Objectives:
 - Use an open-source layout tool (KLayout) to implement a sophisticated layout design environment for silicon photonics
 - Support for both GUI and Python script-based layout, or combinations of both.
 - KLayout-INTERCONNECT integration offers a layout-first design methodology. Inspired by Layout Versus Schematic tools, this includes netlist extraction routines to generate a schematic from the layout. This allows the user to directly simulate from the layout, without needing to create the schematic first. This approach is appealing to photonics designers who are accustomed to designing physical layouts, rather than schematics. A library of components (layout and compact models) is included in the SiEPIC-EBeam-PDK Process Design Kit, specifically for silicon photonics fabrication via Electron Beam Lithography.
 - Whereas a typical schematic-driven design flow includes a schematic, circuit simulation, layout, and verification (see Chapter 10 of the <a href="http://www.cambridge.org/ca/academic/subjects/engineering/electronic-optoelectronic-devices-and-nanotechnology/silicon-photonics-design-devices-systems">textbook</a>), the approach taken here is <b>Layout-driven</b>, followed by verification, then a schematic (via a netlist extraction) and simulations.


**Video of a layout and simulation of a ring resonator circuit**:

<p align="center">
  <a href="https://www.youtube.com/watch?v=1E47VP6Fod0">
  <img src="http://img.youtube.com/vi/1E47VP6Fod0/0.jpg" alt="Layout and simulation of a ring resonator circuit"/>
  </a>
</p>

**Monte Carlo simulations of a ring resonator circuit, showing fabrication variations**:

<p align="center">
  <a href="https://www.youtube.com/watch?v=gUiBsVRlzPE">
  <img src="http://img.youtube.com/vi/gUiBsVRlzPE/0.jpg" alt="Monte Carlo simulations of a ring resonator circuit"/>
  </a>
</p>

**Layout of a Mach-Zehnder Interferometer**:

<p align="center">
  <a href="http://www.youtube.com/watch?v=FRmkGjVUIH4">
  <img src="http://img.youtube.com/vi/FRmkGjVUIH4/0.jpg" alt="Layout of a Mach-Zehnder Interferometer"/>
  </a>
</p>

**Simulations for the MZI**:

<p align="center">
  <a href="http://www.youtube.com/watch?v=1bVO4bpiO58">
  <img src="http://img.youtube.com/vi/1bVO4bpiO58/0.jpg" alt="Lumerical INTERCONNECT simulations"/>
  </a>
</p>

## Package includes:

- Generic Silicon Photonics (GSiP) Process Design Kit (PDK): this package, including fabrication documentation, scripts, etc.
- PCells: ring modulator
- GDS Library: grating coupler, detector, edge coupler.

- Verification: 
  - Scanning the layout. Finding waveguides, devices, pins.  
  - Verification: Identifying if there are missing connections, mismatched waveguides, too few points in a bend, etc. 
  - Example layouts using the library for verification (EBeam_LukasChrostowski_E_LVS.gds, SiEPIC_EBeam_PDK_Verification_Check.gds).
  - Verification for automated measurements
- Circuit simulations:
  - Netlist generation
  - Creating a Spice netlist suitable for for circuit simulations. This includes extracting the waveguide length (wg_length) for all waveguides.
  - Menu item "Lumerical INTERCONNECT" will automatically: generate the netlist, launch Lumerical INTERCONNECT to perform the circuit simulations, and pop-up a plot of the transmission spectrum.
  - Monte Carlo simulations, including waveguides, ring resonators built using directional couplers, y-branches, grating couplers.
- Waveguide functionality: 
  - Hot Key "W": selected paths are first snapped to the nearest pins, then converted to waveguides.
  - Hot Key "Shift-W": selected waveguides are converted back to paths.
  - Hot Key "Ctrl-Shift-W": measure the length of the selected waveguides.
  - Hot Key "Ctrl-Shift-R": resize the waveguides, for a given target length.
- Layout object snapping
- Hot Key "Shift-O": Snaps the selected object to the one where the mouse is hovering over.
- Helper functions from Python-scripted layouts
  - snapping components
  - adding waveguides between components
  - layout primitives (arcs, bezier curves, waveguides, tapers, rings, etc)




## Contributing to this project:

Thank you to the contributors!

[![Contributors!](https://contrib.rocks/image?repo=SiEPIC/SiEPIC-Tools)](https://github.com/SiEPIC/SiEPIC-Tools/graphs/contributors)

You can download the latest development version (master) of the PDK: <a href="https://github.com/siepic/SiEPIC-Tools/archive/master.zip">Zip file download of the PDK</a>

It is posted on GitHub for 1) revision control, 2) so that others can contribute to it, find bugs, 3) easy download of the latest version.

To contribute to the PDK:
 - On the GitHub web page, Fork a copy of the project into your own account.
 - Clone to your Desktop
 - Make edits/contributions.  You can use the KLayout IDE to write Python (or Ruby) scripts; <a href = http://www.klayout.de/doc/about/macro_editor.html>KLayout Python IDE for writing/debugging PCells/scripts/macros</a>.
 - "Commit to master" (your own master)
 - Create a <a href="https://help.github.com/articles/using-pull-requests/">Pull Request</a> -- this will notify me of your contribution, which I can merge into the main project

You can use <a href="https://desktop.github.com/">GitHub desktop</a> to synchronize  files. [Then create symbolic links to your .klayout folder to point to the local copy of this repository](https://www.youtube.com/watch?v=Y5a9kZVgZns). This is useful to automatically update the local KLayout installation (e.g., multiple computers), as changes are made in GitHub by others.

## Screenshots:

![Screenshot1](https://s3.amazonaws.com/edx-course-phot1x-chrostowski/PastedGraphic-9.png)
![Screenshot2](https://s3.amazonaws.com/edx-course-phot1x-chrostowski/PastedGraphic-10.png)
![Screenshot3](https://s3.amazonaws.com/edx-course-phot1x-chrostowski/KLayout_INTERCONNECT.png)

