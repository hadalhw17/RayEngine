-- premake5.lua


require "nvcc"


workspace "RayEngine"
	architecture "x64"
	configurations { "Debug", "Release", "Dist" }

	
outputdir = "%{cfg.buildcfg}-%{cfg.system}-%{cfg.architecture}"


project "RayEngine"
	location "RayEngine"
   kind "SharedLib"
   language "C++"
   targetdir "bin/%{outputdir}/%{prj.name}"
   objdir  "bin-int/%{outputdir}/%{prj.name}"
   toolset "nvcc"

   files
   {
		"%{prj.name}/src/**.h",
		"%{prj.name}/src/**.cpp",
		"%{prj.name}/src/**.cuh",
		"%{prj.name}/src/**.cu"
   }
   
   includedirs
   {
		"%{prj.name}/src",
		"%{prj.name}/vendor/MetaStuff/include",
		"%{prj.name}/Platform/OpenGL",
		"%{prj.name}/includes/imgui",
		"%{prj.name}/includes",
		"$(CudaToolkitDir)/include"
   }


   libdirs { "%{prj.name}/libs" }

   filter "system:windows"
   	   cppdialect "C++11"
	   staticruntime "On"
	   systemversion "10.0.18362.0"
		defines 
		{
			"RE_BUILD_DLL",
			"RE_PLATFORM_WINDOWS"
		}
		postbuildcommands
		{
			("{COPY} %{cfg.buildtarget.relpath} ../bin/" ..outputdir.. "/Sandbox")
		}

	filter "configurations:Release"
		defines {"RE_RELEASE"}
		optimize "On"

	filter "configurations:Dist"
		defines {"RE_RELEASE"}
		optimize "On"

	filter "configurations:Debug"
		defines {"RE_DEBUG"}
		symbols "On"
		optimize "On"



project "Sandbox"
	location "Sandbox"
	kind "ConsoleApp"
	language "C++"

   targetdir "bin/%{outputdir}/%{prj.name}"
   objdir  "bin-int/%{outputdir}/%{prj.name}"

   files
   {
		"%{prj.name}/src/**.h",
		"%{prj.name}/src/**.cpp"
   }
   
   includedirs
   {
		"%{prj.name}/src",
		"RayEngine/vendor/MetaStuff/include",
		"RayEngine/src",
		"$(CudaToolkitDir)/include"

   }

	links
	{
		"RayEngine"
	}

   filter "system:window"
   	   cppdialect "C++11"
	   staticruntime "On"
	   systemversion "10.0.18362.0"
		defines 
		{
			"RE_PLATFORM_WINDOWS"
		}


	filter "configurations:Release"
		defines {"RE_RELEASE"}
		optimize "On"

	filter "configurations:Dist"
		defines {"RE_RELEASE"}
		optimize "On"

	filter "configurations:Debug"
		defines {"RE_DEBUG"}
		symbols "On"
		optimize "On"

