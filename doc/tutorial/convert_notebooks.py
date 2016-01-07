import os
import subprocess
import shutil

notebooks_dir = '../../../scikit-beam-examples'
copy_to_dir = '/tmp/scikit-beam-examples'

git_checkout_master_cmd = ['git', 'checkout', 'master', '--force']

if not os.path.exists(copy_to_dir):
    shutil.copytree(notebooks_dir, copy_to_dir)
# force checkout the master branch, then build the docs
subprocess.check_output(git_checkout_master_cmd, cwd=notebooks_dir)

c.NbConvertApp.notebooks = [
    os.path.join(path, f) for (path, folders, files) in os.walk(notebooks_dir) for f in files
    if f.endswith('.ipynb')
]

