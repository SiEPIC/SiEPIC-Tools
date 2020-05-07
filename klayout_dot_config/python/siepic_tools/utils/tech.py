from __future__ import print_function
from os.path import expanduser, realpath, dirname
from pya import \
    LayerInfo, \
    Technology

# Module attributes

technologies = {}

# Technology management


def get_technology_by_name(name):
    try:
        return technologies[name]
    except KeyError:
        print("Technology %s not found." % name)
        raise


def register_siepic_technology(lyt_filename):
    # Loading db Technology from file
    pya_tech = _load_pya_tech(lyt_filename)

    # Registering technology in db Technology
    registered_pya_tech = _register_pyatech(pya_tech)

    tech_name = pya_tech.name
    # Registering technology in siepic_tools.utils.tech.technologies
    technologies[tech_name] = _load_siepic_technology(registered_pya_tech)

    return technologies[tech_name]


def _load_siepic_technology(pya_tech):
    technology = {}
    tech_name = pya_tech.name
    technology['technology_name'] = tech_name
    technology['dbu'] = pya_tech.dbu

    import os
    lyp_file = pya_tech.eff_layer_properties_file()
    technology['base_path'] = pya_tech.base_path()

    # Load CML file location
    head, tail = os.path.split(lyp_file)
    technology['base_path'] = head
    cml_files = [x for x in os.listdir(technology['base_path']) if x.lower().endswith(".cml")]
    if cml_files:
        technology['INTC_CML'] = cml_files[-1]
        technology['INTC_CML_path'] = os.path.join(technology['base_path'], cml_files[-1])
        technology['INTC_CML_version'] = cml_files[-1].replace(tech_name + '_', '')
    else:
        technology['INTC_CML'] = ''
        technology['INTC_CML_path'] = ''
        technology['INTC_CML_version'] = ''

    # Layers:
    layer_map = parse_layer_map(lyp_file)
    technology.update(layer_map)

    return technology


def _load_pya_tech(lyt_filename):
    # Parsing lyt filename

    # workaround while https://github.com/klayoutmatthias/klayout/pull/215 is not solved
    absolute_filepath = realpath(expanduser(lyt_filename))
    with open(absolute_filepath, 'r') as file:
        lyt_xml = file.read()
    pya_tech = Technology.technology_from_xml(lyt_xml)
    pya_tech.default_base_path = dirname(absolute_filepath)
    # end of workaround

    return pya_tech


def _register_pyatech(pya_tech):
    if Technology.has_technology(pya_tech.name):
        print("Warning: overwriting %s technology" % pya_tech.name)
        new_tech = Technology.technology_by_name(pya_tech.name)
    else:
        new_tech = Technology.create_technology(pya_tech.name)

    # Registering new technology to klayout's database
    new_tech.assign(pya_tech)
    return new_tech

# Layer Properties


def parse_layer_map(lyp_filename):
    with open(lyp_filename, 'r') as file:
        layer_dict = xml_to_dict(file.read())['layer-properties']['properties']

    layer_map = {}

    for k in layer_dict:
        layerInfo = k['source'].split('@')[0]
        if 'group-members' in k:
            # encoutered a layer group, look inside:
            j = k['group-members']
            if 'name' in j:
                layerInfo_j = j['source'].split('@')[0]
                layer_map[j['name']] = LayerInfo(
                    int(layerInfo_j.split('/')[0]), int(layerInfo_j.split('/')[1]))
            else:
                for j in k['group-members']:
                    layerInfo_j = j['source'].split('@')[0]
                    layer_map[j['name']] = LayerInfo(
                        int(layerInfo_j.split('/')[0]), int(layerInfo_j.split('/')[1]))
            if k['source'] != '*/*@*':
                layer_map[k['name']] = LayerInfo(
                    int(layerInfo.split('/')[0]), int(layerInfo.split('/')[1]))
        else:
            layer_map[k['name']] = LayerInfo(
                int(layerInfo.split('/')[0]), int(layerInfo.split('/')[1]))

    return layer_map


# XML functions


def etree_to_dict(t):
    '''XML to Dict parser
    from: https://stackoverflow.com/questions/2148119/how-to-convert-an-xml-string-to-a-dictionary-in-python/10077069
    '''
    from collections import defaultdict
    d = {t.tag: {} if t.attrib else None}
    children = list(t)
    if children:
        dd = defaultdict(list)
        for dc in map(etree_to_dict, children):
            for k, v in dc.items():
                dd[k].append(v)
        d = {t.tag: {k: v[0] if len(v) == 1 else v for k, v in dd.items()}}
    if t.attrib:
        d[t.tag].update(('@' + k, v) for k, v in t.attrib.items())
    if t.text:
        text = t.text.strip()
        if children or t.attrib:
            if text:
                d[t.tag]['#text'] = text
        else:
            d[t.tag] = text
    return d


def xml_to_dict(t):
    from xml.etree import ElementTree as ET
    try:
        e = ET.XML(t)
    except ET.ParseError:
        raise
    except Exception:
        raise UserWarning("Error in the XML file.")
    return etree_to_dict(e)
