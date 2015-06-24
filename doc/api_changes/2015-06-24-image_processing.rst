Image Processing: API changes
-----------------------
Changed/Added
Added image processing API namespace configuration files. The API
configuration files result in an image processing function and tool tree
which categorizes tools based on their function type (e.g. thresholding
tools are included and compiled in thresholding.py, which results in the
tools being listed under a thresholding dropdown tab in the VisTrails
function list GUI window).
The categories which are defined in this API change include:
  - arithmetic: A folder which includes two separate API config files including
    * basic_math: includes tools for simple image arithmetic
    * logic: includes tools for applying logical operations to input data
  - filtering: image filtering tools
  - histogram: tools for evaluating image histograms (e.g. for
    identification of potential thresholding points)
  - morphology: tools for evaluating image, or object,
    morphology (e.g. erosion).
  - thresholding: image and volume thresholding tools
  - transformation: image and volume transformation tools (e.g. cropping)
  - registration: image and volume registration tools
