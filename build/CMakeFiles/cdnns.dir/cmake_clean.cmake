file(REMOVE_RECURSE
  "libcdnns.pdb"
  "libcdnns.so"
)

# Per-language clean rules from dependency scanning.
foreach(lang C CUDA)
  include(CMakeFiles/cdnns.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
