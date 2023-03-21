'''
from SiEPIC.install import install
install('nextcloud-api-wrapper','nextcloud')

nextcloud.api_wrappers.WebDAV.upload_file('lukas', '/tmp/test.gds', 'https://qdot-nexus.phas.ubc.ca:25683/s/b74MAb2Bdz6SLAm', timestamp=None)
-- didn't work, unknown error in the API
'''


'''
from SiEPIC.install import install
install('pyocclient','owncloud')

-- install error
  × python setup.py egg_info did not run successfully.
  │ exit code: 1
  ╰─> [1 lines of output]
      ERROR: Unknown option: --egg-base

public_link = 'https://qdot-nexus.phas.ubc.ca:25683/s/b74MAb2Bdz6SLAm'

oc = owncloud.Client.from_public_link(public_link)
oc.drop_file('/tmp/test.gds')
'''

'''
from SiEPIC.install import install
install('pycurl')
-- install error
  × python setup.py egg_info did not run successfully.
  │ exit code: 1
  ╰─> [1 lines of output]
      ERROR: Unknown option: --egg-base
'''

'''
from SiEPIC.install import install
install('pyncclient')
-- install error
  × python setup.py egg_info did not run successfully.
  │ exit code: 1
  ╰─> [1 lines of output]
      ERROR: Unknown option: --egg-base
'''


