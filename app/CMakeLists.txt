cmake_minimum_required(VERSION 3.4.1)

# 设置生成mace-jni路径
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/../app/libs/${ANDROID_ABI})

include_directories(${CMAKE_SOURCE_DIR}/)
include_directories(${CMAKE_SOURCE_DIR}/src/main/cpp/public)

set(mace_lib ${CMAKE_SOURCE_DIR}/src/main/cpp/lib/armeabi-v7a/libmace.a)

set(mobilenet_lib ${CMAKE_SOURCE_DIR}/src/main/cpp/model/armeabi-v7a/mobilenet.a)
# set(mobilenet_lib ${CMAKE_SOURCE_DIR}/src/main/cpp/model/armeabi-v7a/c10_1.4_256_student55.a)

add_library (mace_lib STATIC IMPORTED)
set_target_properties(mace_lib PROPERTIES IMPORTED_LOCATION ${mace_lib})

add_library (mobilenet_lib STATIC IMPORTED)
set_target_properties(mobilenet_lib PROPERTIES IMPORTED_LOCATION ${mobilenet_lib})

add_library( # Sets the name of the library.
             mace_jni
             SHARED
#             src/main/cpp/mace_jni.cpp
        src/main/cpp/fpi_video_mace_interface.cpp
             )
find_library(
              log-lib
              log )
target_link_libraries( # Specifies the target library.
                        mace_jni
                        mace_lib
                        mobilenet_lib
                       ${log-lib} )