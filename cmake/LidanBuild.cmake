message(STATUS "include LidanBuild.cmake")


#从current到最底层所有的dir addsub
function(LidanAddSubDir path)
    file(GLOB_RECURSE children LIST_DIRECTORY true ${CMAKE_CURRENT_SOURCE_DIR}}/${path}/*)
    set(dirs "")
    list(APPEND children ${CMAKE_CURRENT_SOURCE_DIR}}/${path}/*)
    foreach(item ${children})
        if(IS_DIRECTORY ${item} and EXISTS "${item}/CMakeLists.txt")
            LIST(APPEND dirs ${item})
        endif()
    endforeach()
    foreach(dir ${dirs})
        add_subdirectory(${dir})
    endforeach()

endfunction()


#function(_Lidan_Add_Recurse_Source rst _sources)
#    set(tmp_rst "")
#    message(STATUS add ${${_sources}})
#
#    foreach(item ${${sources}})
#        if(IS_DIRECTORY ${item})
#            file(GLOB)
#        endif()
#    endforeach()
#根据source的所有内部item 如果item是dir 则下去将item下的所有文件加入 如果不是则加入
function(Lidan_ADD_SRC)
    message(STATUS "----------")
    cmake_parse_arguments("ARG" "" "PATH;SOURCEDIR" "" ${ARGN})
    message(STATUS "${ARG_PATH} ${ARG_MSVC} ${ARG_COMPONENTS} ")
    Lidan_Path_Back(_LIB_ROOT ${ARG_PATH} 1)

    set(paths "${_LIB_ROOT}/include;")
    set(pathtosrc "${_LIB_ROOT}/src")
    _Lidan_AddSource(srcdir paths)
    _lidan_AddSource(pathtosrc pathtosrc)
    set(srcresult "")
    list(APPEND srcresult ${srcdir})
    list(APPEND srcresult ${pathtosrc})
    set(${ARG_SOURCEDIR} ${srcresult} PARENT_SCOPE)
    include_directories(${_LIB_ROOT}/include/)

endfunction()


function(Lidan_ADD_LIB )
    message(STATUS "----------")
    cmake_parse_arguments("ARG" "" "PATH;MSVC;SOURCEDIR;INCLUDEDIRS" "COMPONENTS" ${ARGN})
    message(STATUS "${ARG_PATH} ${ARG_MSVC} ${ARG_COMPONENTS} ")
    Lidan_Path_Back(_LIB_ROOT ${ARG_PATH} 1)

    set(paths "${_LIB_ROOT}/include;")
    _Lidan_AddSource(srcdir paths)
    set(${ARG_SOURCEDIR} ${srcdir} PARENT_SCOPE)

    message(STATUS ${_LIB_ROOT})
    if(${ARG_MSVC} STREQUAL "TRUE")
        include_directories(${_LIB_ROOT}/include/)
#        set(${ARG_INCLUDEDIRS} "${_LIB_ROOT}/include/" PARENT_SCOPE)
        message(STATUS "include ${_LIB_ROOT}/include/}")
    endif()
    foreach(_cmpt ${ARG_COMPONENTS})
#        message(STATUS ${_cmpt})
        if(${ARG_MSVC} STREQUAL "TRUE")
            set(_dllPathR "${_LIB_ROOT}/lib-vc2019/${_cmpt}.dll")
            install(FILES ${_dllPathR} TYPE BIN CONFIGURATIONS Debug)
            install(FILES ${_dllPathR} TYPE BIN CONFIGURATIONS Release)
            MESSAGE(STATUS ${_LIB_ROOT}/lib-vc2019/${_cmpt})
#           install(FILES ${_dllPathR} DESTINATION ${CMAKE_BINARY_DIR} )
            link_directories("${_LIB_ROOT}/lib-vc2019/")
            file(COPY ${_dllPathR}
                    DESTINATION ${CMAKE_BINARY_DIR})
            message(STATUS "link ${_dllPathR}")
        endif()
    endforeach()

endfunction()



function(_Lidan_AddSource rst _sources)
    set(tmp_rst "")
    message(STATUS "Add ${${_sources}}")
    foreach(item ${${_sources}})
        if(IS_DIRECTORY ${item})
            file(GLOB_RECURSE itemSrcs
                    # cmake
                    ${item}/*.cmake

                    # msvc
                    ${item}/*.natvis

                    # INTERFACEer files
                    ${item}/*.h
                    ${item}/*.hpp
                    ${item}/*.cuh
                    ${item}/*.cu
                    ${item}/*.hxx
                    ${item}/*.inl

                    # source files
                    ${item}/*.c

                    ${item}/*.cc
                    ${item}/*.cpp
                    ${item}/*.cxx

                    # shader files
                    ${item}/*.vert # glsl vertex shader
                    ${item}/*.tesc # glsl tessellation control shader
                    ${item}/*.tese # glsl tessellation evaluation shader
                    ${item}/*.geom # glsl geometry shader
                    ${item}/*.frag # glsl fragment shader
                    ${item}/*.comp # glsl compute shader

                    #${item}/*.hlsl
                    #${item}/*.hlsli
                    #${item}/*.fx
                    #${item}/*.fxh

                    # Qt files
                    ${item}/*.qrc
                    ${item}/*.ui
                    )
            list(APPEND tmp_rst ${itemSrcs})
        else()
            if(NOT IS_ABSOLUTE "${item}")
                get_filename_component(item "${item}" ABSOLUTE)
            endif()
            list(APPEND tmp_rst ${item})
        endif()
    endforeach()
    set(${rst} ${tmp_rst} PARENT_SCOPE)
    message(STATUS ${tmp_rst})
endfunction()

# 加入src 加入此文件夹一下path下 二级内部所有文件
function(ADD_SUBSRC res path)
    set(next "" )
    set(paths "${CMAKE_CURRENT_SOURCE_DIR}/${path};")
    _Lidan_AddSource(next paths)
    set(${res} ${next} PARENT_SCOPE)
endfunction()

function(Lidan_Add_Target)
    message(STATUS "-------")

    set(arglist "")
    list(APPEND arglist SOURCE_PUBLIC INC LIB DEFINE C_OPTION L_OPTION)
    list(APPEND arglist SOURCE_INTERFACE INC_INTERFACE LIB_INTERFACE DEFINE_INTERFACE C_OPTION_INTERFACE L_OPTION_INTERFACE)
    list(APPEND arglist SOURCE INC_PRIVATE LIB_PRIVATE DEFINE_PRIVATE C_OPTION_PRIVATE L_OPTION_PRIVATE)

    cmake_parse_arguments("ARG" "TEST;QT;NOT_GROUP" "MODE;ADD_CURRENT_TO;RET_TARGET_NAME" "${arglist}" "${ARGN}")

    #default
    if("${ARG_ADD_CURRENT_TO}" STREQUAL "")
        set(ARG_ADD_CURRENT_TO PRIVATE)
        endif()
    if("${ARG_MODE}" STREQUAL "INTERFACE")
        list(APPEND ARG_SOURCE_INTERFACE   ${ARG_SOURCE_PUBLIC} ${ARG_SOURCE}          )
        list(APPEND ARG_INC_INTERFACE      ${ARG_INC}           ${ARG_INC_PRIVATE}     )
        list(APPEND ARG_LIB_INTERFACE      ${ARG_LIB}           ${ARG_LIB_PRIVATE}     )
        list(APPEND ARG_DEFINE_INTERFACE   ${ARG_DEFINE}        ${ARG_DEFINE_PRIVATE}  )
        list(APPEND ARG_C_OPTION_INTERFACE ${ARG_C_OPTION}      ${ARG_C_OPTION_PRIVATE})
        list(APPEND ARG_L_OPTION_INTERFACE ${ARG_L_OPTION}      ${ARG_L_OPTION_PRIVATE})
        set(ARG_SOURCE_PUBLIC    "")
        set(ARG_SOURCE           "")
        set(ARG_INC              "")
        set(ARG_INC_PRIVATE      "")
        set(ARG_LIB              "")
        set(ARG_LIB_PRIVATE      "")
        set(ARG_DEFINE           "")
        set(ARG_DEFINE_PRIVATE   "")
        set(ARG_C_OPTION         "")
        set(ARG_C_OPTION_PRIVATE "")
        set(ARG_L_OPTION         "")
        set(ARG_L_OPTION_PRIVATE "")




        if(NOT "${ARG_ADD_CURRENT_TO}" STREQUAL "NONE")
            set(ARG_ADD_CURRENT_TO "INTERFACE")
        endif()
    endif()


    # [option]
    # TEST
    # QT
    # NOT_GROUP
    # [value]
    # MODE: EXE / STATIC / SHARED / INTERFACE
    # ADD_CURRENT_TO: PUBLIC / INTERFACE / PRIVATE (default) / NONE
    # RET_TARGET_NAME
    # [list] : public, interface, private
    #    # SOURCE: dir(recursive), file, auto add currunt dir | target_sources
    #    # INC: dir                                           | target_include_directories
    # LIB: <lib-target>, *.lib                           | target_link_libraries
    # DEFINE: #define ...                                | target_compile_definitions
    # C_OPTION: compile options                          | target_compile_options
    # L_OPTION: link options                             | target_link_options




endfunction()