function LinkTFModel()
	-- Get the location of the current script file
    local scriptLocation = debug.getinfo(1, "S").source:sub(2)

    -- Determine the relative directory based on the current script location
    local relativeDir = path.getdirectory(scriptLocation)

	filter "configurations:Debug"
		includedirs
		{
			relativeDir .. "/src/"
		}
		links
		{
			"TFModelCore"
		}
	filter "configurations:Release"
		includedirs
		{
			relativeDir .. "/src/"
		}
		links
		{
			"TFModelCore"
		}
	filter "configurations:Dist"
		includedirs
		{
			relativeDir .. "/src/"
		}
		links
		{
			"TFModelCore"
		}
end