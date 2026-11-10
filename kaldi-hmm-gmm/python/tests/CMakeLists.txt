function(khg_add_py_test source)
  get_filename_component(name ${source} NAME_WE)
  set(name "${name}_py")

  get_filename_component(khg_path ${CMAKE_CURRENT_LIST_DIR} DIRECTORY)

  # The extension module lives in ${khg_path}/kaldi_hmm_gmm/lib/ (built
  # there by the _kaldi_hmm_gmm target); helper modules live in tools/.
  # PYTHONPATH entries are separated by ":" on posix and ";" on Windows.
  # The ENVIRONMENT test property is a CMake list: a literal ";" in a value
  # splits it into bogus entries, and $<SEMICOLON> does not help because
  # ctest re-splits the property after evaluating the genex (so only the
  # first entry would survive). Run the test through "cmake -E env" instead:
  # it keeps the whole assignment as a single argument on all platforms.
  if(WIN32)
    set(_pathsep ";")
  else()
    set(_pathsep ":")
  endif()
  # kaldi_hmm_gmm is self-contained (plan-4.md): it does not import
  # kaldi_decoder, so no sibling-checkout python dir is needed here.
  set(_pythonpath
    "${khg_path}${_pathsep}${khg_path}/../../tools${_pathsep}$ENV{PYTHONPATH}"
  )

  add_test(NAME ${name}
    COMMAND
      "${CMAKE_COMMAND}" -E env "PYTHONPATH=${_pythonpath}"
      "${PYTHON_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/${source}"
  )
endfunction()

# please sort the files in alphabetic order
set(py_test_files
  test_ali_to_phones.py
  test_am_diag_gmm.py
  test_const_event_map.py
  test_context_dep.py
  test_diag_gmm.py
  test_draw_tree.py
  test_gauss_clusterable.py
  test_gmm_update_flags.py
  test_hmm_topology.py
  test_init_gmm_mono.py
  test_lifetime_safety.py
  test_mle_am_diag_gmm.py
  test_mle_diag_gmm.py
  test_scalar_clusterable.py
  test_selfcontained.py
  test_transition_model.py
)

foreach(source IN LISTS py_test_files)
  khg_add_py_test(${source})
endforeach()
