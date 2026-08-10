# NAMBench conformance build.
#
# This is NOT the benchmark. The benchmark is the Xcode project generated from
# project.yml: one process, one in-RAM model, one in-RAM audio file, four
# frameworks built through one target template so that nothing but the engine
# can differ between two measured numbers. Nothing here reproduces that, and no
# timing produced by a binary built from this file is a result.
#
# What this file is for is the other half of the question. The engines are
# arithmetic, and arithmetic is portable in a way that `xcodebuild` is not. This
# build takes the same engine sources to Linux, Windows and Android, and to
# x86_64 as well as AArch64, under GCC, Clang and MSVC, and asks two things:
#
#   1. does it still compile, and
#   2. does it still compute the same audio.
#
# Both are checked by Tools/nam_conformance.cpp and Scripts/compare-conformance.py.
#
# Symbol collisions are avoided the same way the Xcode build avoids them, by a
# different means. There, four dynamic frameworks each hide their own nam::
# symbols behind -fvisibility=hidden and export one prefixed C API. Here, each
# variant becomes its own *executable*, so two definitions of nam::wavenet never
# meet in the first place. The consequence is that cross-variant comparison
# happens between processes, through the output files the driver writes, rather
# than in one address space.
#
# Usage:
#   cmake -S . -B build-conformance -DCMAKE_BUILD_TYPE=Release
#   cmake --build build-conformance --parallel 4
#   ctest --test-dir build-conformance --output-on-failure
#
# vendor/ must be populated first: ./Scripts/fetch-vendor.sh

cmake_minimum_required(VERSION 3.20)

project(NAMBenchConformance LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(NB_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(NB_VENDOR "${NB_ROOT}/vendor")

# ---------------------------------------------------------------------------
# vendor/ is fetched, not committed. Fail with the command to run rather than
# with three hundred lines of missing-header errors.
# ---------------------------------------------------------------------------
foreach(_required
	"${NB_VENDOR}/upstream/NAM/wavenet/a2_fast.cpp"
	"${NB_VENDOR}/fused/NAM/wavenet/fused.cpp"
	"${NB_VENDOR}/planar/NAM/wavenet/a2_planar.cpp"
	"${NB_VENDOR}/eigen/Eigen/Dense"
	"${NB_VENDOR}/upstream/Dependencies/nlohmann/json.hpp")
	if(NOT EXISTS "${_required}")
		message(FATAL_ERROR
			"missing ${_required}\n"
			"  vendor/ is fetched at pinned commits and is not committed. Run:\n"
			"      ./Scripts/fetch-vendor.sh")
	endif()
endforeach()

# ---------------------------------------------------------------------------
# Architecture.
#
# The fused engine and both kernel labs are hand-written NEON. fused.cpp guards
# its intrinsics and its detector returns false off AArch64, so the `fused`
# variant builds anywhere — it simply routes to the generic engine instead.
# That is not a degraded test: on x86_64 the upstream-vs-fused comparison
# becomes a2_fast against the generic reference implementation, which is a
# correctness check worth having and one the Xcode build cannot perform at all.
#
# The lab sources are not guarded that way and are not meant to be. They are
# candidate NEON kernels; a scalar fallback for them would be a fiction that
# proves nothing. So the two lab variants are AArch64-only.
# ---------------------------------------------------------------------------
# Order matters. On an Apple Silicon host CMAKE_SYSTEM_PROCESSOR reads arm64
# even when CMAKE_OSX_ARCHITECTURES asks for an x86_64 cross-build, so the
# explicit target settings have to be consulted first or an x86_64 build would
# try to compile the NEON labs.
if(DEFINED ANDROID_ABI AND NOT ANDROID_ABI STREQUAL "")
	set(NB_TARGET_ARCH "${ANDROID_ABI}")
elseif(DEFINED CMAKE_OSX_ARCHITECTURES AND NOT CMAKE_OSX_ARCHITECTURES STREQUAL "")
	set(NB_TARGET_ARCH "${CMAKE_OSX_ARCHITECTURES}")
elseif(MSVC AND DEFINED CMAKE_GENERATOR_PLATFORM AND NOT CMAKE_GENERATOR_PLATFORM STREQUAL "")
	set(NB_TARGET_ARCH "${CMAKE_GENERATOR_PLATFORM}")
else()
	set(NB_TARGET_ARCH "${CMAKE_SYSTEM_PROCESSOR}")
endif()

if(NB_TARGET_ARCH MATCHES "^(aarch64|arm64|ARM64|arm64-v8a|arm64e)$")
	set(NB_AARCH64 ON)
else()
	set(NB_AARCH64 OFF)
endif()

# A universal or multi-arch build would compile the lab sources for every slice,
# including the ones with no NEON. Refuse rather than fail deep in a header.
list(LENGTH NB_TARGET_ARCH _nb_arch_count)
if(_nb_arch_count GREATER 1)
	message(FATAL_ERROR
		"the conformance build targets one architecture at a time, got '${NB_TARGET_ARCH}'.\n"
		"  Configure once per architecture instead.")
endif()

message(STATUS "NAMBench conformance: target arch '${NB_TARGET_ARCH}', AArch64 = ${NB_AARCH64}")

# ---------------------------------------------------------------------------
# Flags shared by every variant.
#
# Optimisation is forced on in all configurations, mirroring the rule
# project.yml states for the Xcode build: a Debug build must still exercise
# optimised engine code. For conformance the reason is different but no weaker
# — an unoptimised build does not exercise the vectoriser, and the vectoriser is
# where a portability bug would live.
# ---------------------------------------------------------------------------
add_library(nb_conformance_flags INTERFACE)

if(MSVC)
	target_compile_options(nb_conformance_flags INTERFACE
		/O2 /EHsc /bigobj /utf-8 /W3
		# Eigen and nlohmann are noisy under MSVC's conformance warnings and
		# neither is ours to fix.
		/wd4267 /wd4244 /wd4305)
	target_compile_definitions(nb_conformance_flags INTERFACE
		NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
else()
	target_compile_options(nb_conformance_flags INTERFACE
		-O3 -Wall -Wextra -Wno-unused-parameter)
endif()

target_compile_definitions(nb_conformance_flags INTERFACE
	NDEBUG=1
	NAM_ENABLE_A2_FAST=1)

# ---------------------------------------------------------------------------
# One function builds all four variants, so the only things that can differ
# between them are the arguments — which is the same discipline the Engine
# target template enforces in project.yml.
#
#   nb_add_variant(<target> VENDOR <upstream|fused> PREFIX <sym> NAME <display>
#                  [DEFINES ...] [SOURCES ...])
# ---------------------------------------------------------------------------
function(nb_add_variant target)
	cmake_parse_arguments(V "" "VENDOR;PREFIX;NAME" "DEFINES;SOURCES" ${ARGN})

	set(_tree "${NB_VENDOR}/${V_VENDOR}")

	file(GLOB _nam_sources CONFIGURE_DEPENDS
		"${_tree}/NAM/*.cpp"
		"${_tree}/NAM/*/*.cpp")

	add_executable(${target}
		"${NB_ROOT}/Tools/nam_conformance.cpp"
		"${NB_ROOT}/Sources/Shim/nam_bench_shim.cpp"
		${V_SOURCES}
		${_nam_sources})

	target_include_directories(${target} PRIVATE
		"${_tree}"
		"${NB_ROOT}/Sources/Shim"
		"${NB_ROOT}/Sources/SlimEngines"
		"${NB_ROOT}/Sources/FullEngines")

	# Eigen and nlohmann as SYSTEM, so -Wall -Wextra stays about our code.
	target_include_directories(${target} SYSTEM PRIVATE
		"${_tree}/Dependencies/nlohmann"
		"${NB_VENDOR}/eigen")

	# NB_VARIANT_NAME is passed unquoted and stringized inside the shim, exactly
	# as project.yml does it, so both builds agree on the reported name.
	target_compile_definitions(${target} PRIVATE
		NB_PREFIX=${V_PREFIX}
		NB_VARIANT_NAME=${V_NAME}
		${V_DEFINES})

	target_link_libraries(${target} PRIVATE nb_conformance_flags)

	set_target_properties(${target} PROPERTIES
		CXX_VISIBILITY_PRESET hidden
		VISIBILITY_INLINES_HIDDEN ON)

	# std::filesystem needs an explicit library on the GCC versions that still
	# ship it separately.
	if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
		target_link_libraries(${target} PRIVATE stdc++fs)
	endif()
endfunction()

# ---------------------------------------------------------------------------
# Which variants to build.
#
# `upstream` and `planar` are the live comparison — the reference against the
# kernels proposed in Core PR #313 — and are always built. Everything else is
# regression insurance on code that no longer changes, and it is not cheap:
# measured on an M2, all six conformance targets take 256 s against 32 s for
# these two alone. Most of that is the kernel labs compiling 67 NEON candidates
# at -O3, which is also what makes a CI job sit silent at 95% long enough to
# look hung.
#
# NAMBENCH_ALL_VARIANTS brings back `fused` and both labs. What it buys:
#
#   fused     the reference the full lab's fu* candidates are validated
#             against, and — off AArch64 — the only thing that exercises the
#             generic engine, because there the upstream-vs-fused comparison
#             is a2_fast against the generic reference implementation.
#   labs      bit-identity checks on the verbatim ports, which are what make
#             every other lab number mean anything.
#
# Neither is worthless; both are checks on static code. They belong on a
# deliberate run rather than on every push.
# ---------------------------------------------------------------------------
option(NAMBENCH_ALL_VARIANTS
	"Also build the superseded fused engine and the kernel labs" OFF)

# --- Variant 1: sdatkinson/NeuralAmpModelerCore, routed to a2_fast ----------
nb_add_variant(nam_conformance_upstream
	VENDOR upstream
	PREFIX nb_upstream
	NAME a2_fast)

# ---------------------------------------------------------------------------
# The planar NEON kernels, as proposed to Core in PR #313.
#
# The checkout is the head of that draft PR, so the thing measured here and the
# thing proposed there cannot drift apart.
#
# NAM_A2_PLANAR — the switch that decides whether the kernels exist at all — is
# defined inside a2_planar.h for any target that defines __aarch64__. Off
# AArch64 the translation unit compiles to an object with no symbols and this
# variant is plain a2_fast. That is not a build failure and must not be treated
# as one: the driver asks which engine it actually got, and reports a2_fast
# honestly.
#
# Nothing needs forcing. The gate used to be __APPLE__ && __aarch64__, and this
# file used to carry a NAMBENCH_FORCE_A2_PLANAR option to open it; the gate has
# since been widened upstream on the evidence that option was added to collect,
# so both are gone.
# ---------------------------------------------------------------------------
nb_add_variant(nam_conformance_planar
	VENDOR planar
	PREFIX nb_planar
	NAME a2_planar
	DEFINES NB_VENDOR_HAS_PLANAR=1)

# --- The superseded fused engine, on request only ---------------------------
if(NAMBENCH_ALL_VARIANTS)
	nb_add_variant(nam_conformance_fused
		VENDOR fused
		PREFIX nb_fused
		NAME fused
		DEFINES NAM_ENABLE_FUSED=1)
endif()

# The labs need more than AArch64. They are written for the Clang/GCC dialect —
# `#pragma clang loop`, `#pragma clang fp contract`, __builtin_prefetch — which
# MSVC proper does not speak. That is not a portability defect to fix: those
# pragmas are how a candidate kernel pins down the codegen it is testing, and a
# version with them stripped out would be a different kernel. clang-cl is fine
# and is not excluded here, because it understands all of it.
if(NAMBENCH_ALL_VARIANTS AND NB_AARCH64 AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
	set(NB_BUILD_LABS ON)
else()
	set(NB_BUILD_LABS OFF)
endif()
message(STATUS "NAMBench conformance: all variants = ${NAMBENCH_ALL_VARIANTS}, kernel labs = ${NB_BUILD_LABS}")

if(NB_BUILD_LABS)
	file(GLOB _slim_sources CONFIGURE_DEPENDS "${NB_ROOT}/Sources/SlimEngines/*.cpp")
	file(GLOB _full_sources CONFIGURE_DEPENDS "${NB_ROOT}/Sources/FullEngines/*.cpp")

	# --- Variant 3: the slimmed-path lab, 3-channel submodel ---------------
	nb_add_variant(nam_conformance_slim
		VENDOR upstream
		PREFIX nb_slim
		NAME slim
		DEFINES NB_ENABLE_SLIM_LAB=1
		SOURCES ${_slim_sources})

	# --- Variant 4: the full-path lab, 8-channel submodel ------------------
	nb_add_variant(nam_conformance_full
		VENDOR fused
		PREFIX nb_full
		NAME full
		DEFINES NAM_ENABLE_FUSED=1 NB_ENABLE_FULL_LAB=1
		SOURCES ${_full_sources})
endif()

# ---------------------------------------------------------------------------
# The benchmark, optional and off by default.
#
# Everything above answers "is it correct here". This answers "how fast is it
# here", and it exists for one reason: the numbers this project publishes come
# from the Xcode build, and the Xcode build does not run on a Raspberry Pi.
#
# It is built differently from the conformance targets, and has to be. A
# benchmark comparing two engines must run them in ONE process — the app and the
# CLI do, and measuring across two processes would put the allocator, the page
# cache and the scheduler between the two numbers. So this reproduces the Xcode
# layout rather than the conformance layout: one shared library per variant,
# each hiding its own nam:: symbols behind -fvisibility=hidden and exporting only
# its prefixed C API, all four loaded into one driver.
#
# Not available under MSVC. That layout depends on hidden visibility plus a
# default-visibility attribute on the exports, which is not how Windows DLLs
# work — they need __declspec(dllexport) and an explicit import library. Windows
# gets conformance, which is what it is for.
# ---------------------------------------------------------------------------
option(NAMBENCH_BUILD_BENCHMARK "Build the portable benchmark driver" OFF)

if(NAMBENCH_BUILD_BENCHMARK AND CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
	message(FATAL_ERROR
		"NAMBENCH_BUILD_BENCHMARK needs hidden-visibility shared libraries, which "
		"MSVC does not provide. Use the conformance targets on Windows.")
endif()

if(NAMBENCH_BUILD_BENCHMARK)
	# Same arguments as nb_add_variant, but produces a shared library carrying
	# only the shim's prefixed C API.
	function(nb_add_engine_library target)
		cmake_parse_arguments(V "" "VENDOR;PREFIX;NAME" "DEFINES;SOURCES" ${ARGN})

		set(_tree "${NB_VENDOR}/${V_VENDOR}")
		file(GLOB _nam_sources CONFIGURE_DEPENDS
			"${_tree}/NAM/*.cpp"
			"${_tree}/NAM/*/*.cpp")

		add_library(${target} SHARED
			"${NB_ROOT}/Sources/Shim/nam_bench_shim.cpp"
			${V_SOURCES}
			${_nam_sources})

		target_include_directories(${target} PRIVATE
			"${_tree}"
			"${NB_ROOT}/Sources/Shim"
			"${NB_ROOT}/Sources/SlimEngines"
			"${NB_ROOT}/Sources/FullEngines")

		target_include_directories(${target} SYSTEM PRIVATE
			"${_tree}/Dependencies/nlohmann"
			"${NB_VENDOR}/eigen")

		target_compile_definitions(${target} PRIVATE
			NB_PREFIX=${V_PREFIX}
			NB_VARIANT_NAME=${V_NAME}
			${V_DEFINES})

		target_link_libraries(${target} PRIVATE nb_conformance_flags)

		# The whole arrangement rests on this: without hidden visibility both
		# checkouts would export nam:: and the dynamic linker would bind one
		# library's calls to the other's code, silently measuring the same engine
		# twice.
		set_target_properties(${target} PROPERTIES
			CXX_VISIBILITY_PRESET hidden
			VISIBILITY_INLINES_HIDDEN ON)
	endfunction()

	nb_add_engine_library(nam_engine_upstream
		VENDOR upstream PREFIX nb_upstream NAME a2_fast)

	nb_add_engine_library(nam_engine_planar
		VENDOR planar PREFIX nb_planar NAME a2_planar
		DEFINES NB_VENDOR_HAS_PLANAR=1)

	set(NB_ENGINE_LIBRARIES nam_engine_upstream nam_engine_planar)

	# `fused` left the measured line-up when the planar kernels superseded it, so
	# it is only built when asked for — either to reproduce a published fused
	# number with --with-fused, or as the reference the full lab's fu* candidates
	# are validated against.
	if(NAMBENCH_ALL_VARIANTS)
		nb_add_engine_library(nam_engine_fused
			VENDOR fused PREFIX nb_fused NAME fused
			DEFINES NAM_ENABLE_FUSED=1)
		list(APPEND NB_ENGINE_LIBRARIES nam_engine_fused)
	endif()

	if(NB_BUILD_LABS)
		nb_add_engine_library(nam_engine_slim
			VENDOR upstream PREFIX nb_slim NAME slim
			DEFINES NB_ENABLE_SLIM_LAB=1
			SOURCES ${_slim_sources})

		nb_add_engine_library(nam_engine_full
			VENDOR fused PREFIX nb_full NAME full
			DEFINES NAM_ENABLE_FUSED=1 NB_ENABLE_FULL_LAB=1
			SOURCES ${_full_sources})

		list(APPEND NB_ENGINE_LIBRARIES nam_engine_slim nam_engine_full)
	endif()

	add_executable(nam_benchmark "${NB_ROOT}/Tools/nam_benchmark.cpp")

	target_include_directories(nam_benchmark PRIVATE "${NB_ROOT}/Sources/Shim")
	target_link_libraries(nam_benchmark PRIVATE nb_conformance_flags ${NB_ENGINE_LIBRARIES})

	if(NB_BUILD_LABS)
		target_compile_definitions(nam_benchmark PRIVATE NAMBENCH_HAVE_LABS=1)
	endif()
	if(NAMBENCH_ALL_VARIANTS)
		target_compile_definitions(nam_benchmark PRIVATE NAMBENCH_HAVE_FUSED=1)
	endif()

	# Find the engine libraries next to the executable, so the build tree can be
	# copied to the machine under test as a unit.
	if(APPLE)
		set_target_properties(nam_benchmark PROPERTIES
			BUILD_RPATH "@loader_path"
			INSTALL_RPATH "@loader_path")
	else()
		set_target_properties(nam_benchmark PROPERTIES
			BUILD_RPATH "$ORIGIN"
			INSTALL_RPATH "$ORIGIN")
	endif()
endif()

# ---------------------------------------------------------------------------
# The test.
#
# The model is upstream's own example_models/A2.nam, which fetch-vendor.sh
# brings in with the rest of the checkout. It is a genuine A2 SlimmableContainer
# — a 3-channel nano submodel at max_value 0.5 and an 8-channel full submodel at
# max_value 1.0, 23 layers each — so it exercises every path the benchmark does.
# Using it means CI needs no capture, which matters because the captures this
# repository benchmarks are other people's work and are not committed.
# ---------------------------------------------------------------------------
enable_testing()

set(NB_CONFORMANCE_MODEL "${NB_VENDOR}/upstream/example_models/A2.nam"
	CACHE FILEPATH "The .nam the conformance run loads")

find_package(Python3 COMPONENTS Interpreter)

set(NB_RUN_TARGETS nam_conformance_upstream nam_conformance_planar)
if(NAMBENCH_ALL_VARIANTS)
	list(APPEND NB_RUN_TARGETS nam_conformance_fused)
endif()
if(NB_BUILD_LABS)
	list(APPEND NB_RUN_TARGETS nam_conformance_slim nam_conformance_full)
endif()

foreach(_target ${NB_RUN_TARGETS})
	add_test(NAME run.${_target}
		COMMAND ${_target}
			--model "${NB_CONFORMANCE_MODEL}"
			--out "${CMAKE_CURRENT_BINARY_DIR}/conformance")
endforeach()

if(Python3_Interpreter_FOUND)
	add_test(NAME compare
		COMMAND "${Python3_EXECUTABLE}" "${NB_ROOT}/Scripts/compare-conformance.py"
			"${CMAKE_CURRENT_BINARY_DIR}/conformance")
	# Every variant must have written its records before anything is compared.
	set_tests_properties(compare PROPERTIES DEPENDS "${NB_RUN_TARGETS}")
	foreach(_target ${NB_RUN_TARGETS})
		set_tests_properties(run.${_target} PROPERTIES FIXTURES_SETUP conformance_data)
	endforeach()
	set_tests_properties(compare PROPERTIES FIXTURES_REQUIRED conformance_data)
else()
	message(WARNING "Python 3 not found; the parity comparison will not run")
endif()
