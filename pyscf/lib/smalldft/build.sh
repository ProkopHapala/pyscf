#!/bin/bash
# Quick build of libsmalldft.so (no full PySCF cmake required).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/smalldft/small_grid.c"
OUT="$ROOT/libsmalldft.so"
INC="$ROOT"
# Fallback: pip-installed np_helper for linking
for d in "$HOME/.local/lib/python"*/site-packages/pyscf/lib \
         /usr/local/lib/python*/site-packages/pyscf/lib; do
    if [[ -f "$d/libnp_helper.so" ]]; then
        LIBDIR="$d"
        break
    fi
done
if [[ -z "${LIBDIR:-}" ]]; then
    echo "libnp_helper.so not found; build pyscf/lib first or set LIBDIR" >&2
    exit 1
fi
OPENBLAS=$(ls "$LIBDIR"/libopenblas*.so 2>/dev/null | head -1)
BLAS_EXTRA=()
if [[ -n "$OPENBLAS" ]]; then
    BLAS_EXTRA=("$OPENBLAS")
else
    BLAS_EXTRA=(-lopenblas)
fi
TILE_FLAG=()
if [[ -n "${SMALLDFT_TILE:-}" ]]; then
    TILE_FLAG=(-DTILE="$SMALLDFT_TILE")
fi
gcc -shared -fPIC -O3 -fopenmp -std=c99 \
    -I"$INC" \
    "${TILE_FLAG[@]}" \
    "$SRC" -o "$OUT" \
    -L"$LIBDIR" -lnp_helper -Wl,-rpath,"$LIBDIR" -lgomp \
    "${BLAS_EXTRA[@]}"
echo "built $OUT"
