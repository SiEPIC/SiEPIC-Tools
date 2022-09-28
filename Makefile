KLAYOUT_DIR=$$HOME/.klayout
INSTALL_DIR=$$HOME/SiEPIC-Tools
LUMERICAL_DIR=$$HOME/.config/Lumerical

install:
	mkdir -p $(INSTALL_DIR)
	cp -rv * $(INSTALL_DIR) 

	ln -s $(INSTALL_DIR)/klayout_dot_config/pymacros/* $(KLAYOUT_DIR)/pymacros/
	ln -s $(INSTALL_DIR)/klayout_dot_config/python/* $(KLAYOUT_DIR)/python/
	ln -s $(INSTALL_DIR)/Python_packages_for_KLayout/python/* $(KLAYOUT_DIR)/python/

	mkdir -p "$(KLAYOUT_DIR)/tech"
	ln -s $(INSTALL_DIR)/klayout_dot_config/tech/* $(KLAYOUT_DIR)/tech/

	mkdir -p "$(LUMERICAL_DIR)/Custom"
	ln -s $(INSTALL_DIR)/Lumerical_CML_GSiP/GSiP/* $(LUMERICAL_DIR)/Custom/

	# Create a backup file of the configuration and delete the old one

	if [ -f "$(LUMERICAL_DIR)/INTERCONNECT.ini.bak" ]; then rm "$(LUMERICAL_DIR)/INTERCONNECT.ini.bak"; fi

	cp -v $(LUMERICAL_DIR)/INTERCONNECT.ini $(LUMERICAL_DIR)/INTERCONNECT.ini.bak

	grep -q -F '[Design%20kits]' "$(LUMERICAL_DIR)/INTERCONNECT.ini" || echo '[Design%20kits]' >> "$(LUMERICAL_DIR)/INTERCONNECT.ini"
	grep -q -F '/GSip' $(LUMERICAL_DIR)/INTERCONNECT.ini || cat $(LUMERICAL_DIR)/INTERCONNECT.ini | sed "/Design/a\ \nGSiP=$HOME/SiEPIC-Tools/Lumerical_CML_GSiP/GSiP\n" > $(LUMERICAL_DIR)/INTERCONNECT.ini

uninstall:
	rm -rvf $(INSTALL_DIR)

	find $(KLAYOUT_DIR)/pymacros/ -xtype l -delete
	find $(KLAYOUT_DIR)/python/ -xtype l -delete
	find $(KLAYOUT_DIR)/tech/ -xtype l -delete
	find $(LUMERICAL_DIR)/Custom/ -xtype l -delete

	cp $(LUMERICAL_DIR)/INTERCONNECT.ini $(LUMERICAL_DIR)/INTERCONNECT.ini.bak.bak
	mv $(LUMERICAL_DIR)/INTERCONNECT.ini.bak $(LUMERICAL_DIR)/INTERCONNECT.ini
	mv $(LUMERICAL_DIR)/INTERCONNECT.ini.bak.bak $(LUMERICAL_DIR)/INTERCONNECT.ini.bak
