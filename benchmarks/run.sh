#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------
# W# Benchmark Suite — W# vs Lua vs Bun vs Julia
# ------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="$(dirname "$SCRIPT_DIR")"
BUN="$HOME/.bun/bin/bun"
JULIA="$HOME/.juliaup/bin/julia"
LUA="/usr/bin/lua"
RUNS=5

cd "$WORKSPACE"

# Colours
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

# Time a command, return elapsed seconds (3 decimal places).
# Runs $RUNS times, picks the median.
bench() {
    local label="$1"; shift
    local times=()

    for ((r = 1; r <= RUNS; r++)); do
        local start end elapsed
        start=$(date +%s%N)
        "$@" >/dev/null 2>&1
        end=$(date +%s%N)
        elapsed=$(awk "BEGIN{printf \"%.3f\", ($end - $start) / 1000000000}")
        times+=("$elapsed")
    done

    # Sort and pick median
    IFS=$'\n' sorted=($(sort -g <<<"${times[*]}")); unset IFS
    local mid=$(( RUNS / 2 ))
    echo "${sorted[$mid]}"
}

compile_ws() {
    local src="$1" name="$2"
    cargo run -q -p wsharp_cli -- build "$src" -o "$SCRIPT_DIR/$name.o" -O2
    cc "$SCRIPT_DIR/$name.o" -o "$SCRIPT_DIR/$name" -lm
}

# ------------------------------------------------------------------
# Build W# benchmarks
# ------------------------------------------------------------------

echo -e "${BOLD}${CYAN}Building W# benchmarks...${NC}"
compile_ws "$SCRIPT_DIR/fib.ws"    fib_ws
compile_ws "$SCRIPT_DIR/sum.ws"    sum_ws
compile_ws "$SCRIPT_DIR/primes.ws" primes_ws
echo -e "${GREEN}Build OK${NC}\n"

# ------------------------------------------------------------------
# Verify correctness (all languages must produce the same exit code)
# ------------------------------------------------------------------

verify() {
    local bench_name="$1"
    shift
    local expected=""
    for cmd in "$@"; do
        # cmd is a string, eval it
        set +e
        eval "$cmd" >/dev/null 2>&1
        local code=$?
        set -e
        if [ -z "$expected" ]; then
            expected=$code
        elif [ "$code" -ne "$expected" ]; then
            echo -e "${RED}MISMATCH in $bench_name: expected $expected, got $code ($cmd)${NC}"
            return 1
        fi
    done
    echo -e "  ${GREEN}$bench_name: all return exit code $expected${NC}"
}

echo -e "${BOLD}${CYAN}Verifying correctness...${NC}"
verify "fib(35)" \
    "$SCRIPT_DIR/fib_ws" \
    "$LUA $SCRIPT_DIR/fib.lua" \
    "$BUN $SCRIPT_DIR/fib.js" \
    "$JULIA --startup-file=no $SCRIPT_DIR/fib.jl"

verify "sum(100M)" \
    "$SCRIPT_DIR/sum_ws" \
    "$LUA $SCRIPT_DIR/sum.lua" \
    "$BUN $SCRIPT_DIR/sum.js" \
    "$JULIA --startup-file=no $SCRIPT_DIR/sum.jl"

verify "primes(100K)" \
    "$SCRIPT_DIR/primes_ws" \
    "$LUA $SCRIPT_DIR/primes.lua" \
    "$BUN $SCRIPT_DIR/primes.js" \
    "$JULIA --startup-file=no $SCRIPT_DIR/primes.jl"
echo ""

# ------------------------------------------------------------------
# Run benchmarks
# ------------------------------------------------------------------

echo -e "${BOLD}${CYAN}Running benchmarks ($RUNS runs each, median shown)...${NC}\n"

declare -A results

for bm in fib sum primes; do
    results["${bm}_ws"]=$(bench    "W#"    "$SCRIPT_DIR/${bm}_ws")
    results["${bm}_lua"]=$(bench   "Lua"   "$LUA" "$SCRIPT_DIR/${bm}.lua")
    results["${bm}_bun"]=$(bench   "Bun"   "$BUN" "$SCRIPT_DIR/${bm}.js")
    results["${bm}_julia"]=$(bench "Julia"  "$JULIA" --startup-file=no "$SCRIPT_DIR/${bm}.jl")
done

# ------------------------------------------------------------------
# Print results table
# ------------------------------------------------------------------

printf "\n${BOLD}%-20s %10s %10s %10s %10s${NC}\n" "Benchmark" "W#" "Lua" "Bun" "Julia"
printf "%-20s %10s %10s %10s %10s\n"               "--------" "---" "---" "---" "-----"

for bm in fib sum primes; do
    case $bm in
        fib)    label="Fibonacci(35)" ;;
        sum)    label="Sum(100M)" ;;
        primes) label="Primes(100K)" ;;
    esac
    printf "%-20s %9ss %9ss %9ss %9ss\n" \
        "$label" \
        "${results[${bm}_ws]}" \
        "${results[${bm}_lua]}" \
        "${results[${bm}_bun]}" \
        "${results[${bm}_julia]}"
done

echo ""

# ------------------------------------------------------------------
# Cleanup
# ------------------------------------------------------------------

rm -f "$SCRIPT_DIR"/fib_ws "$SCRIPT_DIR"/fib_ws.o
rm -f "$SCRIPT_DIR"/sum_ws "$SCRIPT_DIR"/sum_ws.o
rm -f "$SCRIPT_DIR"/primes_ws "$SCRIPT_DIR"/primes_ws.o

echo -e "${GREEN}Done.${NC}"
