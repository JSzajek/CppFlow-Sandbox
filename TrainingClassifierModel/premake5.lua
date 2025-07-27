include "dependencies.lua"

project "TrainingClassifierModel"
	kind "ConsoleApp"

	language "C++"
	cppdialect "C++20"

	staticruntime "off"

	targetdir ("%{wks.location}/Binaries/" .. outputdir .. "/%{prj.name}")
	objdir ("%{wks.location}/Intermediates/" .. outputdir .. "/%{prj.name}")

	files
	{
		"src/**.h",
		"src/**.cpp",
	}

	includedirs
	{
		"src",
	}

	LinkCppFlow()
	LinkJson()
	LinkTFModel()
	LinkOpenCV4()
	
	filter "system:windows"
		systemversion "latest"
	filter "configurations:Debug"
		symbols "On"
	filter "configurations:Release"
		optimize "On"
	filter "configurations:Dist"
		optimize "Full"