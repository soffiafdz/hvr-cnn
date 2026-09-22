#!/usr/bin/env bash
# End-to-end smoke test of a built image: same command under podman/docker or
# apptainer. Usage: tests/smoke.sh <image-ref | file.sif> <fixture_dir> <out_dir> [reference_dir]
# fixture_dir must contain the stx2 MINC used for the baseline; nothing is
# checked on the host, everything runs inside the container.
set -euo pipefail
image=$1; fixtures=$(readlink -f "$2"); out=$(readlink -f -m "$3"); ref=${4:-}
mkdir -p "$out"
run() {  # run <args...> with fixtures, out (and ref) mounted at the same paths
    if [[ "$image" == *.sif ]]; then
        apptainer run --containall --no-home -B "$fixtures":"$fixtures":ro -B "$out":"$out" \
            ${ref:+-B "$ref":"$ref":ro} "$image" "$@"
    else
        local engine=podman; command -v podman >/dev/null || engine=docker
        $engine run --rm --read-only --network none --user "$(id -u):$(id -g)" --tmpfs /tmp \
            -v "$fixtures":"$fixtures":ro -v "$out":"$out" ${ref:+-v "$ref":"$ref":ro} "$image" "$@"
    fi
}
run selftest
stx2=$(find "$fixtures" -name 'stx2_*_t1.mnc' | head -1)
[[ -n "$stx2" ]] || { echo "no stx2_*_t1.mnc under $fixtures" >&2; exit 2; }
run run -i "$stx2" -o "$out/mnc" --threads "${THREADS:-4}"
run check "$out/mnc" ${ref:+--reference "$ref"}
echo "smoke test passed: $out"
