# Package tests, run by ctest. GPU tests carry the label `gpu` (ctest -LE gpu runs the others).
find_package(GTest CONFIG REQUIRED)
include(GoogleTest)

add_executable(test_fasta test_fasta.cpp)
target_link_libraries(test_fasta PRIVATE p2vbme_io GTest::gtest_main)
gtest_discover_tests(test_fasta)

add_executable(test_maple test_maple.cpp)
target_link_libraries(test_maple PRIVATE p2vbme_io GTest::gtest_main)
gtest_discover_tests(test_maple)

add_executable(test_dedup test_dedup.cpp)
target_link_libraries(test_dedup PRIVATE p2vbme_io p2vbme_processing GTest::gtest_main)
gtest_discover_tests(test_dedup)

add_executable(test_processing test_processing.cpp)
target_link_libraries(test_processing PRIVATE p2vbme_io p2vbme_processing GTest::gtest_main)
gtest_discover_tests(test_processing)

add_executable(test_newick test_newick.cpp)
target_link_libraries(test_newick PRIVATE p2vbme_io GTest::gtest_main)
gtest_discover_tests(test_newick)

add_executable(test_simulations test_simulations.cpp)
target_compile_options(test_simulations PRIVATE -Wall -Wextra)
target_link_libraries(test_simulations PRIVATE p2vbme_simulations GTest::gtest_main)
gtest_discover_tests(test_simulations)

add_executable(test_distances test_distances.cpp)
target_compile_options(test_distances PRIVATE -Wall -Wextra)
target_link_libraries(test_distances PRIVATE p2vbme_distances p2vbme_simulations GTest::gtest_main)
gtest_discover_tests(test_distances)

# The CPU device (any build): brute force at small N, SPR optimality by the reference library, precision, workers
add_executable(test_cpu_device test_cpu_device.cpp ${BME_REFERENCE_SOURCE})
target_include_directories(test_cpu_device PRIVATE ${BME_REFERENCE_INCLUDE})
target_compile_options(test_cpu_device PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O2 -Wall -Wextra> $<$<COMPILE_LANGUAGE:C>:-O2>)
target_link_libraries(test_cpu_device PRIVATE p2vbme p2vbme_io p2vbme_processing p2vbme_simulations GTest::gtest_main
                                              m)
gtest_discover_tests(test_cpu_device DISCOVERY_TIMEOUT 60)

# O(N) branch lengths against the exact O(N^2) ones: CPU device; BranchLengthsGpu.* on the GPU
add_executable(test_branch_lengths test_branch_lengths.cpp)
target_compile_options(test_branch_lengths PRIVATE -O2 -Wall -Wextra)
target_link_libraries(test_branch_lengths PRIVATE p2vbme p2vbme_simulations GTest::gtest_main)
gtest_discover_tests(test_branch_lengths TEST_FILTER "-BranchLengthsGpu.*" DISCOVERY_TIMEOUT 60)
if(P2VBME_CUDA)
  gtest_discover_tests(test_branch_lengths TEST_FILTER "BranchLengthsGpu.*" PROPERTIES LABELS gpu DISCOVERY_TIMEOUT 60)
endif()

add_executable(test_json_writer test_json_writer.cpp)
target_compile_options(test_json_writer PRIVATE -Wall -Wextra)
target_link_libraries(test_json_writer PRIVATE p2vbme_json GTest::gtest_main)
gtest_discover_tests(test_json_writer)

add_executable(test_random test_random.cpp)
target_include_directories(test_random PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_compile_options(test_random PRIVATE -Wall -Wextra)
target_link_libraries(test_random PRIVATE GTest::gtest_main)
gtest_discover_tests(test_random)

# CPU core checks against the reference library (exit code)
add_executable(verify_all verify_all.c ${BME_REFERENCE_SOURCE})
target_include_directories(verify_all PRIVATE ${BME_REFERENCE_INCLUDE})
target_compile_options(verify_all PRIVATE -O2 -march=native)
target_link_libraries(verify_all PRIVATE m pthread)
add_test(NAME verify_all COMMAND verify_all)

if(P2VBME_CUDA)
  # Same tree for any number of devices (several devices on one GPU)
  add_executable(test_search_devices test_search_devices.cpp)
  target_compile_options(test_search_devices PRIVATE -Wall -Wextra)
  target_link_libraries(test_search_devices PRIVATE p2vbme p2vbme_simulations GTest::gtest_main)
  gtest_discover_tests(test_search_devices PROPERTIES LABELS gpu)

  # GPU distances under every model against the CPU formulas
  add_executable(test_device_distances test_device_distances.cpp)
  target_compile_options(test_device_distances PRIVATE -Wall -Wextra)
  target_link_libraries(test_device_distances PRIVATE p2vbme p2vbme_processing p2vbme_simulations GTest::gtest_main)
  gtest_discover_tests(test_device_distances PROPERTIES LABELS gpu DISCOVERY_TIMEOUT 60)

  # Error paths: device-step failures (a fake device), argument errors; Errors.Gpu*: out of memory, CUDA errors
  add_executable(test_errors test_errors.cpp)
  target_compile_options(test_errors PRIVATE -Wall -Wextra)
  target_link_libraries(test_errors PRIVATE p2vbme p2vbme_simulations GTest::gtest_main CUDA::cudart)
  gtest_discover_tests(test_errors TEST_FILTER "-Errors.Gpu*")
  gtest_discover_tests(test_errors TEST_FILTER "Errors.Gpu*" PROPERTIES LABELS gpu)
endif()

# The command line, in every build: help, version and argument errors; runs on the CPU; runs on
# the GPU (label gpu, CUDA builds); errors on stderr with a non-zero exit code; the reports parse as JSON.
add_executable(make_test_fasta make_test_fasta.cpp)
target_link_libraries(make_test_fasta PRIVATE p2vbme_io p2vbme_simulations)
set(CLI_DIR ${CMAKE_CURRENT_BINARY_DIR}/cli)
add_test(NAME cli_setup COMMAND ${CMAKE_COMMAND} -E make_directory ${CLI_DIR})
add_test(NAME cli_make_input COMMAND make_test_fasta ${CLI_DIR}/input 300 600 7)
set_tests_properties(cli_setup PROPERTIES FIXTURES_SETUP cli_dir)
set_tests_properties(cli_make_input PROPERTIES FIXTURES_SETUP cli_input FIXTURES_REQUIRED cli_dir)
function(p2vbme_cli_test name regex)
  add_test(NAME cli_${name} COMMAND p2v-bme ${ARGN})
  set_tests_properties(cli_${name} PROPERTIES FIXTURES_REQUIRED cli_input PASS_REGULAR_EXPRESSION "${regex}")
endfunction()
# no GPU needed
p2vbme_cli_test(help "-s ALIGNMENT -m MODEL.*All options: --help-all" --help)
p2vbme_cli_test(help_all "Search options:.*--max-rounds.*Start tree options:.*Branch lengths and output" --help-all)
p2vbme_cli_test(version "p2v-bme 0.1.0" --version)
p2vbme_cli_test(version_short "p2v-bme 0.1.0" -v)
p2vbme_cli_test(missing_args "-s ALIGNMENT and -m MODEL are required" -s ${CLI_DIR}/input.fa)
p2vbme_cli_test(error_message "p2v-bme: error: cannot open .*missing.fa" -s ${CLI_DIR}/missing.fa -m JC69)
add_test(NAME cli_error_exit_code COMMAND p2v-bme -s ${CLI_DIR}/missing.fa -m JC69)
set_tests_properties(cli_error_exit_code PROPERTIES WILL_FAIL TRUE)
p2vbme_cli_test(model_planned "model 'GTR\\+GC' is not available" -s ${CLI_DIR}/input.fa -m GTR+GC)
p2vbme_cli_test(model_plain_gtr "without gamma" -s ${CLI_DIR}/input.fa -m GTR)
p2vbme_cli_test(model_plus_g "continuous gamma is written" -s ${CLI_DIR}/input.fa -m GTR+G)
p2vbme_cli_test(model_unknown "unknown value for model: 'XYZ'" -s ${CLI_DIR}/input.fa -m XYZ)
p2vbme_cli_test(bad_devices "unknown value for devices: 'x'" -s ${CLI_DIR}/input.fa -m JC69 --devices x)
p2vbme_cli_test(bad_accelerator "unknown value for accelerator: 'tpu'" -s ${CLI_DIR}/input.fa -m JC69 --accelerator tpu)
p2vbme_cli_test(duplicate_devices "GPU 0 is listed twice" -s ${CLI_DIR}/input.fa -m JC69 --accelerator gpu
                --devices 0,0)
p2vbme_cli_test(bad_precision "unknown value for precision: 'quad'" -s ${CLI_DIR}/input.fa -m JC69 --precision
                quad)
p2vbme_cli_test(gpu_double_refused "the GPU computes in single precision" -s ${CLI_DIR}/input.fa -m JC69
                --accelerator gpu --precision double)
p2vbme_cli_test(cpu_device_list_refused "unknown value for devices on the CPU: '0,1'" -s
                ${CLI_DIR}/input.fa -m JC69 --accelerator cpu --devices 0,1)
p2vbme_cli_test(extra_argument "unexpected argument 'extra'" -s ${CLI_DIR}/input.fa -m JC69 extra)
p2vbme_cli_test(negative_time_limit "time limit must be >= 0" -s ${CLI_DIR}/input.fa -m JC69 --time-limit -5)
p2vbme_cli_test(unwritable_prefix "cannot write /nonexistent_dir/x.nwk" -s ${CLI_DIR}/input.fa -m JC69 --prefix
                /nonexistent_dir/x)
# -q: no progress (on the CPU, so no fallback warning either)
p2vbme_cli_test(quiet "^$" -s ${CLI_DIR}/input.fa -m JC69 --prefix ${CLI_DIR}/quiet --accelerator cpu -q)
# on the CPU: workers, double precision (auto selects the CPU for it)
p2vbme_cli_test(cpu "BME length" -s ${CLI_DIR}/input.fa -m JC69 --prefix ${CLI_DIR}/cpu --accelerator cpu
                --devices 2)
p2vbme_cli_test(cpu_double "double precision: running on the CPU.*BME length" -s ${CLI_DIR}/input.fa -m TN93
                --prefix ${CLI_DIR}/cpu_double --precision double --report-full)
set_tests_properties(cli_cpu_double PROPERTIES FIXTURES_SETUP cli_cpu_double_run)
find_package(Python3 COMPONENTS Interpreter)
if(Python3_FOUND)
  add_test(NAME cli_cpu_double_report
           COMMAND ${Python3_EXECUTABLE} -c "import json,sys; r=json.load(open(sys.argv[1])); d=r['details']; \
s=d['settings']; assert s['precision'] == 'double' and s['model'] == 'TN93' and s['accelerator'] == 'cpu', s; \
assert r['bme_length'] > 0 and r['warnings'] == ['double precision: running on the CPU'], r; \
assert d['devices'] == ['CPU: %d threads' % s['threads']], d['devices']; print('cpu report ok')"
           ${CLI_DIR}/cpu_double.json)
  set_tests_properties(cli_cpu_double_report PROPERTIES FIXTURES_REQUIRED cli_cpu_double_run)
endif()
# F81, F84, TN93: the base frequencies are those of the whole input, counted before dedup (as FastME counts them)
p2vbme_cli_test(cpu_f81_dups "BME length" -s ${CLI_DIR}/input_dups.fa -m F81 --prefix ${CLI_DIR}/f81_dups
                --accelerator cpu --report-full)
p2vbme_cli_test(cpu_f81_nodedup "BME length" -s ${CLI_DIR}/input_dups.fa -m F81 --prefix ${CLI_DIR}/f81_nodedup
                --accelerator cpu --no-dedup --report-full)
set_tests_properties(cli_cpu_f81_dups cli_cpu_f81_nodedup PROPERTIES FIXTURES_SETUP cli_f81_runs)
if(Python3_FOUND)
  add_test(NAME cli_f81_frequencies_of_the_input
           COMMAND ${Python3_EXECUTABLE} -c "import json, sys; \
f = [json.load(open(p))['details']['model']['base_frequencies'] for p in sys.argv[2:4]]; \
seq = ''.join(l.strip().upper() for l in open(sys.argv[1]) if not l.startswith('>')); \
n = [seq.count(b) for b in 'ACGT']; expect = [x / sum(n) for x in n]; \
assert f[0] == f[1], f; assert all(abs(a - b) < 1e-12 for a, b in zip(f[0], expect)), (f[0], expect); \
print('frequencies of the input')" ${CLI_DIR}/input_dups.fa ${CLI_DIR}/f81_dups.json ${CLI_DIR}/f81_nodedup.json)
  set_tests_properties(cli_f81_frequencies_of_the_input PROPERTIES FIXTURES_REQUIRED cli_f81_runs)
endif()

if(P2VBME_CUDA)
  # on the GPU: every core option and the advanced ones at a non-default value
  set(CLI_GPU_TESTS)
  # each asks for the GPU (so it cannot pass on the CPU fallback), except `default`, which checks auto
  macro(p2vbme_cli_gpu_test name regex)
    set(_args ${ARGN})
    if(NOT "${name}" STREQUAL "default" AND NOT "--accelerator" IN_LIST _args)
      list(APPEND _args --accelerator gpu)
    endif()
    p2vbme_cli_test(${name} "${regex}" ${_args})
    list(APPEND CLI_GPU_TESTS cli_${name})
  endmacro()
  p2vbme_cli_gpu_test(default "read 300 sequences x 600 sites.*BME length" -s ${CLI_DIR}/input.fa -m JC69 --prefix
                      ${CLI_DIR}/default)
  p2vbme_cli_gpu_test(progress "read 300 sequences x 600 sites.*sample-and-place: n0=30 .*BME length" -s
                      ${CLI_DIR}/input.fa -m jc --prefix ${CLI_DIR}/progress --accelerator gpu --devices 1
                      --seed 3)
  p2vbme_cli_gpu_test(device_list "BME length" -s ${CLI_DIR}/input.fa -m JC69 --prefix ${CLI_DIR}/list
                      --devices 0, -t random --max-rounds 3 --early-stopping 0 --time-limit 100)
  p2vbme_cli_gpu_test(advanced "BME length" -s ${CLI_DIR}/input.fa -m JC69 --prefix ${CLI_DIR}/advanced
                      --sample-size 40 --profile-cap 8 --exact-check-eta 0.01 --bl-profile-cap 8 --no-neg-as-zero)
  p2vbme_cli_gpu_test(no_branch_lengths "BME length 0.000000" -s ${CLI_DIR}/input.fa -m JC69 --prefix
                      ${CLI_DIR}/nobl --no-branch-lengths)
  p2vbme_cli_gpu_test(score_given_tree "0 rounds" -s ${CLI_DIR}/input.fa -m JC69 --prefix ${CLI_DIR}/score -t
                      ${CLI_DIR}/input_true.nwk --max-rounds 0 --report-full)
  # input: .maple gives the same tree as FASTA; identical sequences are put back
  p2vbme_cli_gpu_test(maple "BME length" -s ${CLI_DIR}/input.maple -m JC69 --prefix ${CLI_DIR}/maple)
  p2vbme_cli_gpu_test(dups "3 identical sequences set aside" -s ${CLI_DIR}/input_dups.fa -m JC69 --prefix
                      ${CLI_DIR}/dups --report-full)
  p2vbme_cli_gpu_test(no_dedup "BME length" -s ${CLI_DIR}/input_dups.fa -m JC69 --prefix ${CLI_DIR}/nodedup
                      --no-dedup --report-full)
  set_tests_properties(cli_maple cli_dups cli_no_dedup PROPERTIES FIXTURES_SETUP cli_input_runs)
  # every distance model of this version runs; its report names it (k2p = K80)
  foreach(model k2p F81 F84 TN93 LogDet)
    p2vbme_cli_gpu_test(model_${model} "BME length" -s ${CLI_DIR}/input.fa -m ${model} --prefix
                        ${CLI_DIR}/model_${model} --report-full)
    set_tests_properties(cli_model_${model} PROPERTIES FIXTURES_SETUP cli_model_${model}_run)
  endforeach()
  set_tests_properties(${CLI_GPU_TESTS} PROPERTIES LABELS gpu)
  set_tests_properties(cli_default PROPERTIES FIXTURES_SETUP cli_report)
  set_tests_properties(cli_score_given_tree PROPERTIES FIXTURES_SETUP cli_score)
  if(Python3_FOUND)
    add_test(NAME cli_report_is_json
             COMMAND ${Python3_EXECUTABLE} -c "import json,sys; r=json.load(open(sys.argv[1])); \
assert list(r) == ['version', 'command_line', 'date', 'bme_length', \
'rounds_run', 'stop_reason', 'time', 'warnings'], list(r); \
assert r['bme_length'] > 0 and r['rounds_run'] >= 1 and r['warnings'] == [], r; print('report ok')"
             ${CLI_DIR}/default.json)
    add_test(NAME cli_score_report
             COMMAND ${Python3_EXECUTABLE} -c "import json,sys; r=json.load(open(sys.argv[1])); \
assert r['rounds_run'] == 0 and r['stop_reason'] == 'none' and r['bme_length'] > 0, r; \
assert r['details']['round_gains'] == [], r['details']; print('score ok')" ${CLI_DIR}/score.json)
    set_tests_properties(cli_report_is_json PROPERTIES FIXTURES_REQUIRED cli_report LABELS gpu)
    add_test(NAME cli_maple_equals_fasta
             COMMAND ${Python3_EXECUTABLE} -c "import sys; a, b = (open(f).read() for f in sys.argv[1:]); \
assert a == b, 'trees differ'; print('same tree')" ${CLI_DIR}/default.nwk ${CLI_DIR}/maple.nwk)
    add_test(NAME cli_dups_every_name
             COMMAND ${Python3_EXECUTABLE} -c "import json, re, sys; t = open(sys.argv[1]).read(); \
names = sorted(re.findall(r'[(,]([A-Za-z0-9_]+):', t)); \
expect = sorted(['t%d' % i for i in range(300)] + ['t1_copy1', 't1_copy2', 't1_copy3', 't2_copy1', 't2_copy2']); \
assert names == expect, (len(names), len(expect)); \
assert json.load(open(sys.argv[2]))['details']['input']['identical_set_aside'] == 3; \
assert json.load(open(sys.argv[3]))['details']['input']['identical_set_aside'] == 0; print('all names')"
             ${CLI_DIR}/dups.nwk ${CLI_DIR}/dups.json ${CLI_DIR}/nodedup.json)
    set_tests_properties(cli_maple_equals_fasta cli_dups_every_name PROPERTIES
                         FIXTURES_REQUIRED "cli_report;cli_input_runs" LABELS gpu)
    set_tests_properties(cli_score_report PROPERTIES FIXTURES_REQUIRED cli_score LABELS gpu)
    foreach(pair "k2p=K80" "F81=F81" "F84=F84" "TN93=TN93" "LogDet=LogDet")
      string(REPLACE "=" ";" kv ${pair})
      list(GET kv 0 model)
      list(GET kv 1 name)
      add_test(NAME cli_model_${model}_report
               COMMAND ${Python3_EXECUTABLE} -c "import json,sys; r=json.load(open(sys.argv[1])); d=r['details']; \
assert list(r) == ['version', 'command_line', 'date', 'bme_length', \
'rounds_run', 'stop_reason', 'time', 'warnings'] + ['details'], list(r); \
assert d['settings']['model'] == sys.argv[2], d['settings']; \
assert r['bme_length'] > 0; f = d['model'].get('base_frequencies'); \
assert (f is not None and abs(sum(f) - 1) < 1e-9) == (sys.argv[2] in ('F81', 'F84', 'TN93')), d['model']; \
assert all(g >= 0 for g in d['round_gains']) and d['round_gains'][0] > 0, d['round_gains']; \
print('model ok')" ${CLI_DIR}/model_${model}.json ${name})
      set_tests_properties(cli_model_${model}_report PROPERTIES FIXTURES_REQUIRED cli_model_${model}_run LABELS gpu)
    endforeach()
  endif()
endif()
