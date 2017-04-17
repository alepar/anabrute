cmake_minimum_required(VERSION 3.0)
project (anabrute)

add_executable(anabrute anabrute.c anatypes.c seedphrase.c fact.c)
target_link_libraries(anabrute pthread)
set_property(TARGET anabrute PROPERTY C_STANDARD 99)

add_executable(anaperm anaperm.c anatypes.c seedphrase.c)
set_property(TARGET anaperm PROPERTY C_STANDARD 99)

add_executable(anaheap anaheap.c anatypes.c seedphrase.c)
set_property(TARGET anaheap PROPERTY C_STANDARD 99)

add_executable(anadecomp anadecomp.c fact.c)
set_property(TARGET anadecomp PROPERTY C_STANDARD 99)