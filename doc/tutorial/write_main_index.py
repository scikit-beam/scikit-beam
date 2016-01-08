import os
import tempfile
# make the index page
# I originally wanted to use jinja2 to do this, but the CSS has double curly
# brackets, which jinja does not like, so here we go!
output_dir = os.path.join(tempfile.gettempdir(), 'notebooks')
notebooks = sorted([f for f in os.listdir(output_dir) if f.endswith('.ipynb')])

# read in the template
with open('../index.rst.tmpl', 'r') as f:
    new_file = []
    for line in f.readlines():
        if line.endswith('\n'):
            line = line[:-1]
        # find the line where I want to write the list of examples
        if 'REPLACE_WITH_LIST_OF_EXAMPLES' in line:
            line = ['   tutorial/%s' % os.path.splitext(notebook_path)[0]
                    for notebook_path in notebooks]
            new_file.extend(line)
        else:
            new_file.append(line)

with open('../index.rst', 'w') as f:
    output = '\n'.join(new_file)
    f.write(output)
