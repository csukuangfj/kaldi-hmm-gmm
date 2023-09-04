function(kgh_add_py_test source)
  get_filename_component(name ${source} NAME_WE)
  set(name "${name}_py")

  add_test(NAME ${name}
    COMMAND
      "${PYTHON_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/${source}"
  )

  get_filename_component(khg_path ${CMAKE_CURRENT_LIST_DIR} DIRECTORY)

  set_property(TEST ${name}
    PROPERTY ENVIRONMENT "PYTHONPATH=${khg_path}:$<TARGET_FILE_DIR:_kaldi_hmm_gmm>:$ENV{PYTHONPATH}"
  )
endfunction()

# please sort the files in alphabetic order
set(py_test_files
  test_am_diag_gmm.py
  test_const_event_map.py
  test_context_dep.py
  test_diag_gmm.py
  test_draw_tree.py
  test_eigen.py
  test_gauss_clusterable.py
  test_gmm_update_flags.py
  test_hmm_topology.py
  test_init_gmm_mono.py
  test_mle_am_diag_gmm.py
  test_mle_diag_gmm.py
  test_scalar_clusterable.py
  test_transition_model.py
)

foreach(source IN LISTS py_test_files)
  kgh_add_py_test(${source})
endforeach()
