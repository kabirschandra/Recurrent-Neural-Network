#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
CXX="${CXX:-g++}"
CXXFLAGS=(-std=c++17 -O3 -Wall -Wextra -pedantic)

usage() {
  cat <<'HELP'
Usage: ./run_tests.sh <command> [arguments]
  help                       Show commands
  compile [sine|smoke|all]   Compile tests (default: all)
  run [epochs lr T step]     Run sine test (auto-compiles if needed)
  smoke                      Run a forward/MSE/backward sanity test
  plot [--show]              Plot CSV results with Python/matplotlib
  all [epochs lr T step]     Compile all, run smoke + sine, then plot
  clean                      Delete compiled files and generated data

Default sine arguments: epochs=10000, lr=0.0001, T=5, step=0.1
CXX=clang++ ./run_tests.sh compile  # Choose compiler
HELP
}
compile_one() {
  local name="$1"
  mkdir -p build
  case "$name" in
    sine) "$CXX" "${CXXFLAGS[@]}" tests/Sine/main.c++ -o build/sine_test ;;
    smoke) "$CXX" "${CXXFLAGS[@]}" tests/Smoke/main.c++ -o build/smoke_test ;;
    *) echo "Unknown test: $name" >&2; exit 2 ;;
  esac
  echo "Compiled $name"
}
compile() {
  case "${1:-all}" in
    all) compile_one smoke; compile_one sine ;;
    sine|smoke) compile_one "$1" ;;
    *) usage; exit 2 ;;
  esac
}
run_sine() {
  [[ -x build/sine_test ]] || compile_one sine
  ./build/sine_test "$@"
}
run_smoke() {
  [[ -x build/smoke_test ]] || compile_one smoke
  ./build/smoke_test
}
cmd="${1:-help}"
shift || true
case "$cmd" in
  help|-h|--help) usage ;;
  compile) compile "${1:-all}" ;;
  run) run_sine "$@" ;;
  smoke) run_smoke ;;
  plot) python3 tests/Sine/plot_sine.py "$@" ;;
  all) compile all; run_smoke; run_sine "$@"; python3 tests/Sine/plot_sine.py ;;
  clean) rm -rf build tests/Sine/{loss.csv,predictions.csv,loss.png,predictions.png}; echo "Cleaned" ;;
  *) usage; exit 2 ;;
esac
