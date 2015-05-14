# $Id:$

set( RelativeDir "./src/scene" )
set( RelativeSourceGroup "source\\scene" )
set( SubDirs accel)

set( DirFiles
    BoundingBox.h
    HitPoint.h
    Intersection.cu
    Intersection.cuh
    Intersection.h
    Ray.h
    TriangleObject.cpp
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

