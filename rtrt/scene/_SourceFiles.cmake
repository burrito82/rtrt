# $Id:$

set( RelativeDir "./rtrt/scene" )
set( RelativeSourceGroup "source\\scene" )
set( SubDirs accel)

set( DirFiles
    BarycentricCoords.h
    BoundingBox.h
    Color.h
    HitPoint.h
    Intersection.h
    LoaderObj.cpp
    LoaderObj.h
    Material.h
    PointLight.h
    Ray.cuh
    RayTriangleIntersection.cuh
    Scene.cpp
    Scene.h
    Scene.cu
    Scene.cuh
    SceneIntersectLinear.inl
    Triangle.h
    TriangleGeometry.h
    TriangleGeometryDesc.h
    TriangleObject.h

    _SourceFiles.cmake
)
set( DirFiles_SourceGroup "${RelativeSourceGroup}" )

set( LocalSourceGroupFiles  )
foreach( File ${DirFiles} )
	list( APPEND LocalSourceGroupFiles "${RelativeDir}/${File}" )
	list( APPEND ProjectSources "${RelativeDir}/${File}" )
endforeach()
source_group( ${DirFiles_SourceGroup} FILES ${LocalSourceGroupFiles} )

set( SubDirFiles "" )
foreach( Dir ${SubDirs} )
	list( APPEND SubDirFiles "${RelativeDir}/${Dir}/_SourceFiles.cmake" )
endforeach()

foreach( SubDirFile ${SubDirFiles} )
	include( ${SubDirFile} )
endforeach()

