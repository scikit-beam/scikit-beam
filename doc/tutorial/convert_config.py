import os
import tempfile
# set up the config to convert the notebooks to html
output_dir = os.path.join(tempfile.gettempdir(), 'notebooks')
notebooks = sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.ipynb')])
c.NbConvertApp.notebooks = notebooks