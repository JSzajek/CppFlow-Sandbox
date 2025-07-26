@echo off

call git init
call git submodule add https://github.com/JSzajek/Json-Lib-fork.git "vendor/json_lib/"
call git submodule add https://github.com/JSzajek/CppFlow-Lib-fork.git "vendor/cppflow_lib/"
PAUSE