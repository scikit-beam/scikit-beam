import os
import tempfile
import glob
import shutil
# make the index page
# I originally wanted to use jinja2 to do this, but the CSS has double curly
# brackets, which jinja does not like, so here we go!
examples_dir = os.path.join(tempfile.gettempdir(), 'scikit-beam-examples')
output_dir = os.path.join(tempfile.gettempdir(), 'notebooks')
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

os.mkdir(output_dir)

# copy all notebooks from the git repo to /tmp/notebooks
nbs = glob.glob('/tmp/scikit-beam-examples/**/*.ipynb', recursive=True)

for nb in nbs:
    shutil.copy(nb, os.path.join(output_dir, os.path.split(nb)[1]))
notebooks = os.listdir(output_dir)
# read in the template
files_to_convert = {'../index.rst.tmpl': '../index.rst',
                    '../tutorial.rst.tmpl': '../tutorial.rst'}
for template_file, output_file in files_to_convert.items():
    with open(template_file, 'r') as f:
        new_file = []
        for single_line in f.readlines():
            single_line = single_line[:-1]
            # find the line where I want to write the list of examples
            if 'REPLACE_WITH_LIST_OF_EXAMPLES' in single_line:
                tutorial_lines = [
                    '   tutorial/%s' % os.path.splitext(notebook_path)[0]
                    for notebook_path in notebooks]
                new_file.extend(tutorial_lines)
            else:
                new_file.append(single_line)

    with open(output_file, 'w') as f:
        output = '\n'.join(new_file)
        f.write(output)
