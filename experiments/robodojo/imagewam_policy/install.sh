#!/bin/bash
# Install XPolicyLab (websocket policy server + utils) into the ImageWAM venv so the
# policy server can import XPolicyLab.policy.imagewam_policy and client_server.ws.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV="${IMAGEWAM_VENV:-/storage/yukaichengLab/mazijian/wth/ImageWAM/.venv}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python -m pip install -e "${XPL_ROOT}"
python -c "import XPolicyLab, client_server.ws.model_server, msgpack_numpy, websockets; print('XPolicyLab OK', websockets.__version__)"
