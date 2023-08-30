cmake_minimum_required(VERSION 2.8)


project(camera_calib)
set(CMAKE_CXX_STANDARD 11)
find_package(OpenCV REQUIRED)
add_executable(calib test_calib_main.cpp
		cam_calibration.cpp
)
		

target_link_libraries(calib ${OpenCV_LIBS})

