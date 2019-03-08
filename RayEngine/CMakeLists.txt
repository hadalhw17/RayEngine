cmake_minimum_required(VERSION 3.12.2)

project(RayEngine)

set(CMAKE_CONFIGURATION_TYPES "Debug;Release"
    CACHE STRING "Configuration types" FORCE)

add_executable(RayEngine
  ../../RayTraceer/rply.h
  Atmosphere.cuh
  BoundingVolume.cpp
  BoundingVolume.h
  Camera.cpp
  Camera.h
  Character.cpp
  Character.h
  Cow.cpp
  Cow.h
  CUDARayTracing.cuh
  D:/CPPLibs/GLAD/src/glad.c
  Floor.cpp
  Floor.h
  Glass.cpp
  Glass.h
  GPUBoundingBox.h
  helper_math.h
  jpeg.cpp
  jpeg.h
  jpeglib.h
  KDThreeGPU.cpp
  KDThreeGPU.h
  KDTree.cpp
  KDTree.h
  KDTreeGPUUtills.cpp
  KDTreeGPUUtills.h
  Light.cpp
  Light.h
  MainWindow.cpp
  MainWindow.h
  MovableCamera.cpp
  MovableCamera.h
  Object.cpp
  Object.h
  ObjectComponent.cpp
  ObjectComponent.h
  ObjFileReader.cpp
  ObjFileReader.h
  Plane.cpp
  Plane.h
  Ray.cpp
  Ray.h
  RayEngine.cpp
  RayEngine.h
  RayTracer.cpp
  RayTracer.h
  rply.c
  rplyfile.h
  RStaticMesh.cpp
  RStaticMesh.h
  Scene.cpp
  Scene.h
  SceneObject.cpp
  SceneObject.h
  Source.cpp
  Source.h
  Sphere.cpp
  Sphere.h
  Triangle.cpp
  Triangle.h
)
# Preprocessor definitions
target_compile_definitions(RayEngine PRIVATE
  $<$<CONFIG:Debug>:_MBCS;>
  $<$<CONFIG:Release>:_MBCS;>
)

# SDL check
target_compile_options(RayEngine PRIVATE
  "$<$<CONFIG:Debug>:/sdl>"
  "$<$<CONFIG:Release>:/sdl>"
)

# Minimal rebuild
if (MSVC)
  target_compile_options(RayEngine PRIVATE
    "$<$<CONFIG:Debug>:/Gm->"
    "$<$<CONFIG:Release>:/Gm->"
  )
endif ()

# Precompiled header files
if (MSVC)
  target_compile_options(RayEngine PRIVATE
    ""
  )
endif ()

# Link libraries
set_property(TARGET RayEngine
  APPEND PROPERTY LINK_LIBRARIES
  "$<$<CONFIG:Debug>:opengl32.lib;glfw3.lib;cudart.lib;cudart_static.lib;freetype.lib>"
  "$<$<CONFIG:Release>:opengl32.lib;glfw3.lib;cudart.lib;cudart_static.lib>"
)
