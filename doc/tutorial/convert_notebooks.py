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

<<<<<<< HEAD
c.NbConvertApp.notebooks = [
    os.path.join(path, f) for (path, folders, files) in os.walk(notebooks_dir) for f in files
    if f.endswith('.ipynb')
]

=======
notebooks = [
    os.path.join(path, f) for (path, folders, files) in os.walk(notebooks_dir)
    for f in files if f.endswith('.ipynb')
]
c.NbConvertApp.notebooks = notebooks

# make the index page
from jinja2 import Environment, FileSystemLoader
jinja_env = Environment(loader=FileSystemLoader(os.path.abspath('../')))
template = jinja_env.get_template('index.rst.tmpl')
example_names = ['tutorial/%s' % os.path.splitext(f)[0]
                 for (path, folders, files) in os.walk(notebooks_dir)
                 for f in files if f.endswith('.ipynb')]
template = template.render(examples=example_names)
with open('../index.rst', 'w') as f:
    f.write(template)
>>>>>>> e1fc9e0... Try to programmatically generate all notebooks
