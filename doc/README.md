To build the docs

Install:

```
conda install sphinx numpydoc
pip install sphinx_bootstrap_theme
```

NOTE: You need pandoc v1.13.2 in order for the notebooks to render correctly.
      In newer versions of pandoc, the headings are being [normalized in an odd
      way] (https://github.com/jupyter/nbconvert/issues/97)
      Binaries can be found [here] (https://github.com/jgm/pandoc/releases/tag/1.13.2)

Then you `make notebooks` and `make html` in the `/scikit-beam/doc` folder.  
This will convert the example notebooks into html and generate the html for
the API documentation.

Inspect the output at `/scikit-beam/doc/_build/html/index.html`. If everything
looks good, run `make gh-pages`
