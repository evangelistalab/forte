### Building Forte's manual locally

If you are temporarily offline but would like to check the manual,
you might want to compile the manual locally.
This can be done using `sphinx`.
Additionally, the `sphinx_rtd_theme` and `nbsphinx` modules are required.
Since you already found this README file, simply call `make html` in the current directory.
If successful, the HTML documents can be found under the directory `build/html/`.
Otherwise, it is likely some Python modules are missing,
and they should be installed via `conda` or `pip`.