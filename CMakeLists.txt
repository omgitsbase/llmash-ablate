cmake_minimum_required(VERSION 3.18)
project(llmash-ablate C CXX)

# llama-ablate needs llmash's llama.cpp fork, which can apply a control vector as a projection; upstream adds it
set(LLAMA_CPP_DIR "" CACHE PATH "a checkout of omgitsbase/llama.cpp, branch llmash; fetched when empty")

set(LLAMA_BUILD_COMMON   ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)

if (LLAMA_CPP_DIR)
    add_subdirectory(${LLAMA_CPP_DIR} llama.cpp EXCLUDE_FROM_ALL)
else()
    include(FetchContent)
    FetchContent_Declare(llama_cpp
        GIT_REPOSITORY https://github.com/omgitsbase/llama.cpp.git
        GIT_TAG        llmash
        GIT_SHALLOW    TRUE)
    FetchContent_MakeAvailable(llama_cpp)
    set(LLAMA_CPP_DIR ${llama_cpp_SOURCE_DIR})
endif()

add_executable(llama-ablate src/ablate.cpp)
target_include_directories(llama-ablate PRIVATE ${LLAMA_CPP_DIR}/src)
target_link_libraries(llama-ablate PRIVATE llama-common llama cpp-httplib)
target_compile_features(llama-ablate PRIVATE cxx_std_17)
install(TARGETS llama-ablate RUNTIME)
