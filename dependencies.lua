include "vendor/cppflow_lib/init_tensorflowlib.lua"
include "vendor/cppflow_lib/cppflowlink.lua"
include "vendor/json_lib/jsonlink.lua"

IncludeDir = {}
LibraryDir = {}

Library = {}

-- Windows
Library["WinSock"] = "Ws2_32.lib"
Library["WinMM"] = "Winmm.lib"
Library["WinVersion"] = "Version.lib"