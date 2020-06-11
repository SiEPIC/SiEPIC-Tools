'''
Download files from GitHub using GitHub API and raw download
Requires additional Python modules to be installed: requests
on OSX: > sudo easy_install requests
    or  > pip install requests
'''

print(' loading SiEPIC.github')

# Loading requests during KLayout start-up prevents a ton of exception
# messages if it was loaded on first usage.

try:
    import requests
except ImportError:
    pass

import sys
if 'requests' not in sys.modules:
    try:
        import pip
    except ImportError:
        pass
    if 'pip' in sys.modules:
        import pya
        install = pya.MessageBox.warning(
            "Install package?", "Install package 'requests' using pip?",  pya.MessageBox.Yes + pya.MessageBox.No)
        if install == pya.MessageBox.Yes:
            # try installing using pip
            from SiEPIC.install import get_pip_main
            main = get_pip_main()
            main(['install', 'requests'])


if 'json' not in sys.modules:
    try:
        import pip
    except ImportError:
        pass
    if 'pip' in sys.modules:
        import pya
        install = pya.MessageBox.warning(
            "Install package?", "Install package 'json' using pip?",  pya.MessageBox.Yes + pya.MessageBox.No)
        if install == pya.MessageBox.Yes:
            # try installing using pip
            from SiEPIC.install import get_pip_main
            main = get_pip_main()
            main(['install', 'json'])



# Search the GitHub repository for files containing the string
# "filesearch", with optional extension


def github_check_SiEPICTools_version():

    import sys
    import pya

    try:
        import requests
    except ImportError:
        warning = pya.QMessageBox()
        warning.setStandardButtons(pya.QMessageBox.Ok)
        warning.setText(
            "Missing Python module: 'requests'.  Please install, restart KLayout, and try again.")
        pya.QMessageBox_StandardButton(warning.exec_())
        return []

    import json
    import os
    try:
        r = requests.get("https://api.github.com/repos/lukasc-ubc/SiEPIC-Tools/releases/latest")
    except:
        return ''
    if 'name' not in json.loads(r.text):
        if 'message' in json.loads(r.text):
            message = json.loads(r.text)['message']
        else:
            message = json.loads(r.text)
        pya.MessageBox.warning("GitHub error", "GitHub error: %s" % (message), pya.MessageBox.Ok)
        return ''

    version = json.loads(r.text)['name']
    print(version)
    
    from SiEPIC.__init__ import __version__
    if __version__ not in version:
        pya.MessageBox.warning("SiEPIC-Tools: new version available", "SiEPIC-Tools: new version available: %s.\nUpgrade using Tools > Manage Packages > Update Packages" % (version), pya.MessageBox.Ok)
      


# Search the GitHub repository for files containing the string
# "filesearch", with optional extension


def github_get_filenames(user, repo, filesearch, extension='', auth=None, verbose=None):

    import sys
    import pya

    try:
        import requests
    except ImportError:
        warning = pya.QMessageBox()
        warning.setStandardButtons(pya.QMessageBox.Ok)
        warning.setText(
            "Missing Python module: 'requests'.  Please install, restart KLayout, and try again.")
        pya.QMessageBox_StandardButton(warning.exec_())
        return []

    import json
    import os
    filenames = []
    folders = []
    filesearch = filesearch.replace('%20', ' ')
    r = requests.get("https://api.github.com/search/code?q='%s'+in:path+repo:%s/%s" %
                     (filesearch, user, repo), auth=auth)
    if 'items' not in json.loads(r.text):
        if 'message' in json.loads(r.text):
            message = json.loads(r.text)['message']
        else:
            message = json.loads(r.text)
        pya.MessageBox.warning("GitHub error", "GitHub error: %s" % (message), pya.MessageBox.Ok)
        return ''

    for r in json.loads(r.text)['items']:
        dl = ('https://github.com/' + user + '/' + repo + '/raw/master/' +
              str(r['url']).split('/contents/')[1]).split('?')[0]
        filename = dl.split('/')[-1]
        path = dl.split('/raw/master/')[-1]
        if extension in filename[-len(extension):]:
            filenames.append([filename, path])
        if verbose:
            print('     %s: %s' % (filename, path))
    return filenames

# Get all files from the respository with filename = filename_search
# write to a single folder: save_folder
# or recreate the folder tree, if include_path = True


def github_get_files(user, repo, filename_search, save_folder=None, auth=None, include_path=None, verbose=None):
    import requests
    import json
    import os
    savefilepath = []
    filename_search = filename_search.replace('%20', ' ')
    req = requests.get("https://api.github.com/search/code?q='%s'+in:path+repo:%s/%s" %
                       (filename_search, user, repo), auth=auth)
    for r in json.loads(req.text)['items']:
        dl = ('https://github.com/' + user + '/' + repo + '/raw/master/' +
              str(r['url']).split('/contents/')[1]).split('?')[0]
        filename = dl.split('/')[-1]
        path = str(r['url']).split('/contents/')[1].split('?')[0][0:-
                                                                  len(filename)].replace('%20', '_')
        if verbose:
            print([filename, path, dl])
        req = requests.get(dl, auth=auth)
        if include_path:
            base_path = os.path.join(save_folder, path)
            if not os.path.exists(base_path):
                os.makedirs(base_path)
            savefilepath.append(os.path.join(base_path, filename))
        else:
            savefilepath.append(os.path.join(
                save_folder, path[:-1].replace('/', '-')) + '-' + filename)
        open(savefilepath[-1], 'wb').write(req.content)
    return savefilepath

# Get specific file from the respository with filename = filename_search
# which is located in the path: filepath_search
# write to a folder: save_folder


def github_get_file(user, repo, filename_search, filepath_search, save_folder=None, auth=None, include_path=None, verbose=None):
    import requests
    import json
    import os
    savefilepath = None
    req = requests.get("https://api.github.com/search/code?q='%s'+in:path+repo:%s/%s" %
                       (filename_search, user, repo), auth=auth)
    for r in json.loads(req.text)['items']:
        dl = ('https://github.com/' + user + '/' + repo + '/raw/master/' +
              str(r['url']).split('/contents/')[1]).split('?')[0]
        filename = dl.split('/')[-1]
        path = str(r['url']).split('/contents/')[1].split('?')[0][0:-
                                                                  len(filename)].replace('%20', '_')
        if verbose:
            print([filename, path, dl])
        if filename == filename_search:
            req = requests.get(dl, auth=auth)
            if include_path:
                base_path = os.path.join(save_folder, path)
                if not os.path.exists(base_path):
                    os.makedirs(base_path)
                savefilepath = os.path.join(base_path, filename)
            else:
                savefilepath = os.path.join(save_folder, filename)
            open(savefilepath, 'wb').write(req.content)
    return savefilepath


'''

github_get_files(user='lukasc-ubc', repo='SiEPIC_EBeam_PDK', filesearch='SiEPIC_EBeam_UW_PDK.pdf', save_folder='/tmp/t')

if 0:
  dl='https://github.com/lukasc-ubc/edX-Phot1x/raw/master/2017T3/ANT_chip/TE%2030%20deg%20C/04_Dec_2017_17_27_53/mchaudhery13_WG35_1095.pdf'
  r = requests.get (dl,  auth=('lukasc-ubc',''))
  import os
  open('%s' % (os.path.join('/tmp/t/','mchaudhery13_WG35_1095.pdf')),'w').write(r.content)

  github_get_files(user='lukasc-ubc', repo='edX-Phot1x', filesearch='.pdf', save_folder='/tmp/t')

# Enter your Python code here ..

# from github import Github

g = Github(login_or_token="lukasc-ubc", password="")
for repo in g.get_user().get_repos():
    print repo.name


# pycurl: http://pycurl.io/docs/latest/

# curl -i https://api.github.com/users/lukasc-ubc/repos
# curl -i https://api.github.com/repos/lukasc-ubc/SiEPIC_EBeam_PDK
# curl -i https://api.github.com/search/code?q=addClass+in:file+language:js+repo:jquery/jquery
# curl -i https://api.github.com/search/code?q=lumerical+in:path+repo:lukasc-ubc/SiEPIC-Tools

# curl -i https://api.github.com/search/code?q="ebeam_taper_475_500_te1550_sparams.txt"+in:path+repo:lukasc-ubc/SiEPIC-Tools


# import pycurl

# sudo easy_install requests

# search for filename in repo

print('********************************************************************************')
if 0:
  repo = 'lukasc-ubc/SiEPIC-Tools'
  filename = 'ebeam_taper_475_500_te1550_sparams.txt'
if 1:
  repo = 'lukasc-ubc/SiEPIC_EBeam_PDK'
  filename = 'SiEPIC_EBeam_UW_PDK.pdf'
  filename = '.pdf'
#  filename = 'ol-39-19-5519.pdf'

if 0:  # working exploratory code
  r = requests.get ("https://api.github.com/search/code?q='%s'+in:path+repo:%s" % (filename, repo))
  r.text
  ri = json.loads(r.text)
  print('********************************************************************************')
  print(type(ri))
  print(json.dumps(ri, sort_keys=True, indent=4))
  ri = ri['items'][0]
  print('********************************************************************************')
  print(type(ri))
  print(json.dumps(ri, sort_keys=True, indent=4))
  ri['url']
  print('********************************************************************************')
  r = requests.get (ri['url'])
  ri = json.loads(r.text)
  print(type(ri))
  print(json.dumps(ri, sort_keys=True, indent=4))
  ri['download_url']
  print('********************************************************************************')
#  r = requests.get (ri['download_url'])
#  open('/tmp/%s' % filename,'w').write(r.content)

if 0: # shorter version
  r = requests.get ("https://api.github.com/search/code?q='%s'+in:path+repo:%s" % (filename, repo))
  r = requests.get (json.loads(r.text)['items'][0]['url'])
  r = requests.get (json.loads(r.text)['download_url'])
  open('/tmp/%s' % filename,'w').write(r.content)

if 0: # many files
  r = requests.get ("https://api.github.com/search/code?q='%s'+in:path+repo:%s" % (filename, repo))
  for r in json.loads(r.text)['items']:
    r = requests.get(r['url'])
    print(r)
    print(json.loads(r.text)['download_url'])
#    r = requests.get (json.loads(r.text)['download_url'])
#    open('/tmp/t/%s' % filename,'w').write(r.content)

if 0:
  r = requests.get('https://github.com/lukasc-ubc/SiEPIC_EBeam_PDK/raw/master/Documentation/SiEPIC_EBeam_UW_PDK.pdf')
  open('/tmp/t/SiEPIC_EBeam_UW_PDK.pdf','w').write(r.content)


if 0:  #
  r = requests.get ("https://api.github.com/search/code?q='%s'+in:path+repo:%s" % (filename, repo))
  r.text
  ri = json.loads(r.text)
  print('********************************************************************************')
  print(type(ri))
  print(json.dumps(ri, sort_keys=True, indent=4))
  ri = ri['items'][0]
  print('********************************************************************************')
  print(type(ri))
  print(json.dumps(ri, sort_keys=True, indent=4))
  dl = ri['url'].replace('https://api.github.com/repositories/46163444/contents','https://github.com/lukasc-ubc/SiEPIC_EBeam_PDK/raw/master')
  r = requests.get (dl)
  open('/tmp/t/%s' % filename,'w').write(r.content)



  repo = 'lukasc-ubc/SiEPIC_EBeam_PDK'
  filename = 'SiEPIC_EBeam_UW_PDK.pdf'
  filename = '.pdf'
'''
