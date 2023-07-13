import os
import tempfile
import glob
import shutil
import jinja2
from collections import defaultdict
jinja_env = jinja2.Environment(loader=jinja2.FileSystemLoader('../'))
# make the index page
# I originally wanted to use jinja2 to do this, but the CSS has double curly
# brackets, which jinja does not like, so here we go!
examples_dir = os.path.join(tempfile.gettempdir(), 'scikit-beam-examples')
output_dir = os.path.join(tempfile.gettempdir(), 'notebooks')
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

os.mkdir(output_dir)

# copy all notebooks from the git repo to /tmp/notebooks
nbs = glob.glob('/tmp/scikit-beam-examples/demos/**/*.ipynb', recursive=True)

# copy all notebooks to a common directory to make nbconvert easier
for nb in nbs:
    shutil.copy(nb, os.path.join(output_dir, os.path.split(nb)[1]))
notebooks = os.listdir(output_dir)

# write the jinja templated file
nbdict = defaultdict(list)
for nb in nbs:
    folder_tree, nb = os.path.split(nb)
    nb = os.path.splitext(nb)[0]
    folder_tree, category = os.path.split(folder_tree)
    nbdict[category].append(nb)

# write the jinja templated file
template = jinja_env.get_template('example.rst.tmpl')
rendered = template.render(nbs=nbdict)
with open('../example.rst', 'w') as f:
    f.write(rendered)
