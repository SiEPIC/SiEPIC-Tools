KLAYOUT_DIR=$$HOME/.klayout
INSTALL_DIR=$$HOME/SiEPIC-Tools
LUMERICAL_DIR=$$HOME/.config/Lumerical

all: copyall klayout lumerical

copyall:
	mkdir -p $(INSTALL_DIR)
	cp -r * $(INSTALL_DIR) 

klayout:
	ln -s $(INSTALL_DIR)/klayout_dot_config/pymacros/* $(KLAYOUT_DIR)/pymacros/
	ln -s $(INSTALL_DIR)/klayout_dot_config/python/* $(KLAYOUT_DIR)/python/
	ln -s $(INSTALL_DIR)/Python_packages_for_KLayout/python/* $(KLAYOUT_DIR)/python/

	mkdir -p "$(KLAYOUT_DIR)/tech"
	ln -s $(INSTALL_DIR)/klayout_dot_config/tech/* $(KLAYOUT_DIR)/tech/

lumerical:
	mkdir -p "$(LUMERICAL_DIR)/Custom"
	ln -s $(INSTALL_DIR)/Lumerical_CML_GSiP/GSiP/* $(LUMERICAL_DIR)/Custom/

	# Create a backup file of the configuration and delete the old one

	if [ -f "$(LUMERICAL_DIR)/INTERCONNECT.ini.bak" ]; then rm "$(LUMERICAL_DIR)/INTERCONNECT.ini.bak"; fi

	cp $(LUMERICAL_DIR)/INTERCONNECT.ini $(LUMERICAL_DIR)/INTERCONNECT.ini.bak

	crudini --set $(LUMERICAL_DIR)/INTERCONNECT.ini "Design%20kits" GSIP $(INSTALL_DIR)/Lumerical_CML_GSIP/GSIP

uninstall:
	rm -r $(INSTALL_DIR)

	find $(KLAYOUT_DIR)/pymacros/ -xtype l -delete
	find $(KLAYOUT_DIR)/python/ -xtype l -delete
	find $(KLAYOUT_DIR)/tech/ -xtype l -delete
	find $(LUMERICAL_DIR)/Custom/ -xtype l -delete

	crudini --del $(LUMERICAL_DIR)/INTERCONNECT.ini "Design%20kits" GSIP
