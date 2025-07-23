# Install script for directory: /home/jackal/SAN/spiking-ddpg-mapless-navigation/ros/catkin_ws/src/simple_laserscan

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/jackal/SAN/spiking-ddpg-mapless-navigation/ros/catkin_ws/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/simple_laserscan/msg" TYPE FILE FILES "/home/jackal/SAN/spiking-ddpg-mapless-navigation/ros/catkin_ws/src/simple_laserscan/msg/SimpleScan.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/simple_laserscan/cmake" TYPE FILE FILES "/home/jackal/SAN/spiking-ddpg-mapless-navigation/ros/catkin_ws/build/simple_laserscan/catkin_generated/installspace/simple_laserscan-msg-paths.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/home/jackal/SAN/spiking-ddpg-mapless-navigation/ros/catkin_ws/devel/include/simple_laserscan")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/roseus/ros" TYPE DIRECTORY FILES "/home/jackal/SAN/spiking-ddpg-mapless-navigation/ros/catkin_ws/devel/share/roseus/ros/simple_laserscan")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/common-lisp/ros" TYPE DIRECTORY FILES "/home/jackal/SAN/spiking-ddpg-mapless-navigation/ros/catkin_ws/devel/share/common-lisp/ros/simple_laserscan")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/gennodejs/ros" TYPE DIRECTORY FILES "/home/jackal/SAN/spiking-ddpg-mapless-navigation/ros/catkin_ws/devel/share/gennodejs/ros/simple_laserscan")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  execute_process(COMMAND "/usr/bin/python2" -m compileall "/home/jackal/SAN/spiking-ddpg-mapless-navigation/ros/catkin_ws/devel/lib/python2.7/dist-packages/simple_laserscan")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/python2.7/dist-packages" TYPE DIRECTORY FILES "/home/jackal/SAN/spiking-ddpg-mapless-navigation/ros/catkin_ws/devel/lib/python2.7/dist-packages/simple_laserscan")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/home/jackal/SAN/spiking-ddpg-mapless-navigation/ros/catkin_ws/build/simple_laserscan/catkin_generated/installspace/simple_laserscan.pc")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/simple_laserscan/cmake" TYPE FILE FILES "/home/jackal/SAN/spiking-ddpg-mapless-navigation/ros/catkin_ws/build/simple_laserscan/catkin_generated/installspace/simple_laserscan-msg-extras.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/simple_laserscan/cmake" TYPE FILE FILES
    "/home/jackal/SAN/spiking-ddpg-mapless-navigation/ros/catkin_ws/build/simple_laserscan/catkin_generated/installspace/simple_laserscanConfig.cmake"
    "/home/jackal/SAN/spiking-ddpg-mapless-navigation/ros/catkin_ws/build/simple_laserscan/catkin_generated/installspace/simple_laserscanConfig-version.cmake"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/simple_laserscan" TYPE FILE FILES "/home/jackal/SAN/spiking-ddpg-mapless-navigation/ros/catkin_ws/src/simple_laserscan/package.xml")
endif()

