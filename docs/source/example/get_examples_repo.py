import os
import subprocess
import tempfile

clone_dir = os.path.join(tempfile.gettempdir(), 'scikit-beam-examples')
try:
    ret = subprocess.check_output(
        ['git', 'clone', 'https://github.com/scikit-beam/scikit-beam-examples',
         clone_dir])
except subprocess.CalledProcessError:
    print("scikit-beam-examples already exists at %s" % (clone_dir))
    print("resetting to the master branch")
    subprocess.Popen(['git', 'remote', 'update'], cwd=clone_dir)
    subprocess.Popen(['git', 'reset', '--hard', 'origin/master'],
                     cwd=clone_dir)
