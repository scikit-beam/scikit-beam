import subprocess
import tempfile
import os
clone_dir = os.path.join(tempfile.gettempdir(), 'scikit-beam-examples')
try:
    ret = subprocess.check_output(
        ['git', 'clone', 'https://github.com/ericdill/scikit-beam-examples',
         clone_dir])
except subprocess.CalledProcessError:
    print("scikit-beam-examples already exists at %s" % (clone_dir))
    print("resetting to the master branch")
    subprocess.Popen(['git', 'remote', 'update'], cwd=clone_dir)
    subprocess.Popen(['git', 'reset', '--hard', 'origin/update-notebook-headers'],
                     cwd=clone_dir)