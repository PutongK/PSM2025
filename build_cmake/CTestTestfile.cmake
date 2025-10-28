# CMake generated Testfile for 
# Source directory: /home/pkk5131/workspace/src/PSM2025
# Build directory: /home/pkk5131/workspace/src/PSM2025/build_cmake
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(test1 "/home/pkk5131/workspace/src/PSM2025/bin/testshell")
set_tests_properties(test1 PROPERTIES  _BACKTRACE_TRIPLES "/home/pkk5131/workspace/src/PSM2025/CMakeLists.txt;175;add_test;/home/pkk5131/workspace/src/PSM2025/CMakeLists.txt;0;")
subdirs("_deps/triangle-build")
subdirs("_deps/libigl-build")
subdirs("_deps/googletest-build")
