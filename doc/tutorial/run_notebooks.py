import os
import shutil
import tempfile

notebooks_dir = os.path.abspath(os.path.join(*['..', '..', '..',
                                               'scikit-beam-examples']))

if not os.path.exists(notebooks_dir):
    raise IOError("Notebooks directory not found at %s" % notebooks_dir)

# make sure that the output path is present and empty
output_dir = os.path.join(tempfile.gettempdir(), 'notebooks')
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.mkdir(output_dir)

# find the notebooks in the notebooks dir
notebooks = [
    os.path.join(path, f) for (path, folders, files) in os.walk(notebooks_dir)
    for f in files if f.endswith('.ipynb')
]
from nbconvert.preprocessors import ExecutePreprocessor
import nbformat
def process_notebook(notebook):
    print("Running %s" % notebook)
    # thanks https://github.com/jupyter/nbconvert/issues/125!
    ep = ExecutePreprocessor(timeout=3600, allow_errors=True)
    nb = nbformat.read(notebook, as_version=4)
    nb.metadata = {}
    processed, resources = ep.preprocess(nb, {'metadata': {'path': './'}})
    output_name = os.path.join(output_dir, os.path.split(notebook)[1])
    nbformat.write(processed, output_name)

for notebook in notebooks:
    process_notebook(notebook)
# execute the notebooks
# import multiprocessing
# num_processors_to_use = multiprocessing.cpu_count()-1
# pool = multiprocessing.Pool(num_processors_to_use)
# result_pool = [pool.apply_async(process_notebook, (notebook,))
#                for notebook in notebooks]
# results = [r.get() for r in result_pool]
# pool.terminate()
# pool.join()
