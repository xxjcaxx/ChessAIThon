# c

C/C++ chess-related libraries, bindings and helper code.

Overview

- This folder contains native code and wrappers used by other parts of the project. Subfolders may include full C/C++ projects (for example `chess-library-master`) or Python bindings.

Building

Each native subproject typically contains its own build files (Meson, Makefile, CMake, or package.json for JS bindings). Inspect the subfolder you need to build.

Example (general guidance):

```bash
# change into the native library folder
cd c/chess-library-master
# follow the project's build instructions (Meson, Makefile, etc.)
```

Notes

- If you need Python bindings, check for files named like `*_wrapper.*` or `pybind11`/`swig` configuration.
- The C subprojects may produce shared libraries used by Python or the included examples.
