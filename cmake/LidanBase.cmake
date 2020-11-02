message(STATUS "include LidanBase.cmake")


#输出所有list
function(Lidan_List_Print)
    cmake_parse_arguments("ARG" "" "TITLE;PREFIX" "STRS" ${ARGN})
    list(LENGTH ARG_STRS strsize)
    if(NOT strsize)
        return()
    endif()

    if(NOT ${ARG_TITLE} STREQUAL "")
        message(STATUS ${ARG_TITLE})
    endif()

    foreach(str ${ARG_STRS})
        message(STATUS "${ARG_PREFIX}${str}")
    endforeach()
endfunction()

#设定上一级的dirname变量为cmake current source dir
function(Lidan_DIRNAME)
    string(Regex MATCH "([^/]*)$" TMP ${CMAKE_CURRENT_SOURCE_DIR})
    set(${dirName} ${TMP} PARENT_SCOPE)
endfunction()


function(Lidan_Path_Back rst path times)
    math(EXPR stop "${times}-1")
    set(curPath ${path})
    foreach(index RANGE ${stop})
        string(REGEX MATCH "(.*)/" _ ${curPath})
        set(curPath ${CMAKE_MATCH_1})
    endforeach()

    set(${rst} ${curPath} PARENT_SCOPE)
    message(STATUS ${curPath})
endfunction()
