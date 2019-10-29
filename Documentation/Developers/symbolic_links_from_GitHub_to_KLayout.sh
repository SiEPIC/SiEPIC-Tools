#!/bin/bash

# OSX GitHub repository installation of SiEPIC files for KLayout and Lumerical INTERCONNECT

# assumes that 
# - SiEPIC-* repositories are in ~/Documents/GitHub
# - KLAYOUT_HOME is ~/.klayout

# to run:
# source symbolic_links_from_GitHub_to_KLayout.sh

export SRC=$HOME/Documents/GitHub
export DEST=$HOME/.klayout
export INTC=$HOME/.config/Lumerical

#mkdir $DEST/pymacros/SiEPIC-Tools
ln -s $SRC/SiEPIC-Tools/klayout_dot_config/pymacros/ $DEST/pymacros/SiEPIC-Tools
ln -s $SRC/SiEPIC-Tools/klayout_dot_config/python/* $DEST/python/
ln -s $SRC/SiEPIC-Tools/Python_packages_for_KLayout/python/* $DEST/python/
mkdir $DEST/tech
ln -s $SRC/SiEPIC-Tools/klayout_dot_config/tech/* $DEST/tech/

ln -s $SRC/SiEPIC-Tools/Lumerical_CML_GSiP/GSiP $INTC/Custom

grep -q -F '[Design%20kits]' $INTC/INTERCONNECT.ini || echo '[Design%20kits]' >> $INTC/INTERCONNECT.ini

grep -q -F '/GSiP' $INTC/INTERCONNECT.ini || sed -i .bak '/Design/a\
GSiP='$SRC'/SiEPIC-Tools/Lumerical_CML_GSiP/GSiP
' $INTC/INTERCONNECT.ini

