This folder contains the source files used to create the Compact Model Library

Library name must be identical everywhere (matching technology name in KLayout)

To make updates to the library:

1) make changes to the library

2) update the version folder with the current date (v2018_01_15)

3) package the CML:
zip -r ../klayout_dot_config/tech/GSiP/GSiP_2018_01_20.cml GSiP
(with the current date)

4) For KLayout integration:
	Copy the .cml file into:
	../klayout_dot_config/tech/GSiP

