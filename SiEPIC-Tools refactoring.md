The idea is that we need the entire database layout functionality to be available to the main python package, including the tech definitions. All the GUI functionality must be implemented in the SiEPIC_klayout module and the `*.lym` files. This way the main package can be sent to pypi.

Let's keep clean python code in the `*_klayout` folders, and have the keybindings and loading codes in `*.lym`. This way we can use the editor we like, and even write tests!

Rules of thumb:

- `pya.Application` or `pya.Q*` stuff belongs to klayout.
- here is the list of *database* classes in `pya`: `'Box', 'Cell', 'CellInstArray', 'CellMapping', 'CplxTrans', 'DBox', 'DCellInstArray', 'DCplxTrans', 'DEdge', 'DEdgePair', 'DPath', 'DPoint', 'DPolygon', 'DSimplePolygon', 'DText', 'DTrans', 'DVector', 'Edge', 'EdgePair', 'EdgePairs', 'EdgeProcessor', 'Edges', 'ICplxTrans', 'InstElement', 'Instance', 'LayerInfo', 'LayerMap', 'LayerMapping', 'Layout', 'LayoutDiff', 'LayoutMetaInfo', 'LayoutQuery', 'LayoutQueryIterator', 'Library', 'LoadLayoutOptions', 'Manager', 'Matrix2d', 'Matrix3d', 'PCellDeclaration', 'PCellDeclaration_Native', 'PCellParameterDeclaration', 'ParentInstArray', 'Path', 'Point', 'Polygon', 'RecursiveShapeIterator', 'Region', 'SaveLayoutOptions', 'Shape', 'ShapeProcessor', 'Shapes', 'SimplePolygon', 'Technology', 'TechnologyComponent', 'Text', 'TextGenerator', 'TileOutputReceiver', 'TileOutputReceiverBase', 'TilingProcessor', 'Trans', 'VCplxTrans', 'Vector'`
- If layout functions need information about the tech, then pass it as an argument. Let's not have global variables inside functions! Let's have the technology selection logic in the `_klayout` folder in SiEPIC-tools.
- Please use lazy loading of lumerical tools. It is only used in specific environments (not everyone has, or will need, lumerical)

## SiEPIC Package structure
```
klayout_dot_config
├── pymacros
│   └── *.lym (SiEPIC package initialization)
├── python
│   ├── SiEPIC (module to be loaded from inside klayout)
│   └── siepic_tools (main python package)
│       └── tech (symlink to klayout's tech)
├── tech (files regarding tech)
│   └── tech_name
└── icons
```

## SiEPIC EBeam PDK Package structure
```
klayout_dot_config
├── pymacros
│   └── *.lym (siepic_ebeam_pdk_klayout package initialization)
├── python
│   ├── siepic_ebeam_pdk_klayout (module to be loaded from inside klayout)
│   └── siepic_ebeam_pdk (main python package)
│       └── tech (symlink to klayout's tech)
├── tech (files regarding tech)
│   └── EBeam
│       └── *.xml (files describing the tech)
└── icons
```
