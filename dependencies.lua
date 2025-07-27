include "vendor/cppflow_lib/init_tensorflowlib.lua"
include "vendor/cppflow_lib/cppflowlink.lua"
include "vendor/json_lib/jsonlink.lua"
include "vendor/opencv_lib/opencv4link.lua"
include "TFModelCore/tfmodellink.lua"

IncludeDir = {}
LibraryDir = {}

Library = {}

-- Windows
Library["WinSock"] = "Ws2_32.lib"
Library["WinMM"] = "Winmm.lib"
Library["WinVersion"] = "Version.lib"