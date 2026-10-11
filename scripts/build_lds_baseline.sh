#!/usr/bin/env bash
# Compatibility entry point: no replacement scalar codec is built.
set -euo pipefail
exec bash "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/build_hipans_lds.sh"
