#!/usr/bin/env bash
set -euo pipefail

CONDA_BIN=/storage/yukaichengLab/mazijian/miniconda3/bin/conda
IMAGEWAM_ROOT=/storage/yukaichengLab/mazijian/wth/ImageWAM
STATE_DIR=/storage/yukaichengLab/share/datasets/.robodojo_state
WHEELHOUSE_TORCH=${STATE_DIR}/wheelhouse_torch
WHEELHOUSE_ISAACSIM=${STATE_DIR}/wheelhouse_isaacsim51

mkdir -p "${STATE_DIR}"

exec 9>"${STATE_DIR}/isaacsim_install.lock"
if ! flock -n 9; then
  echo "Another Isaac Sim install is already running" >&2
  exit 1
fi

export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONNOUSERSITE=1
export PIP_USER=0
export PIP_DEFAULT_TIMEOUT=300
export TERM=xterm-256color

eval "$("${CONDA_BIN}" shell.bash hook)"
conda activate RoboDojo

python -m pip install --upgrade pip
python -m pip install \
  numpy==1.26.0 \
  typing_extensions==4.12.2 \
  filelock==3.13.1

# The PyTorch CUDA wheel index only hosts PyTorch artifacts, while torch 2.7
# requires SymPy from PyPI. Install it from the configured cluster PyPI mirror
# before switching pip to the dedicated PyTorch index.
python -m pip install \
  sympy==1.14.0 \
  networkx \
  jinja2 \
  fsspec \
  setuptools

# download.pytorch.org is unreachable/throttled from the H800 login node.
# Pull the three +cu128 wheels from the Aliyun PyTorch mirror (find-links) and
# let every nvidia-*/triton dependency resolve from the campus PyPI mirror.
# ${WHEELHOUSE_TORCH} is pre-populated with `pip download` (retry loop) so a
# transient 504 from the campus mirror on a 700 MB nvidia-* wheel cannot abort
# the whole install. Remote sources stay as fallback for anything missing.
python -m pip install \
  torch==2.7.0+cu128 \
  torchvision==0.22.0+cu128 \
  torchaudio==2.7.0+cu128 \
  --find-links "${WHEELHOUSE_TORCH}" \
  --find-links https://mirrors.aliyun.com/pytorch-wheels/cu128/ \
  --index-url https://mirrors.westlake.edu.cn/pypi/simple/

# pypi.nvidia.com is ~2 KB/s from the H800 login node and its 5.1 wheels are
# tagged manylinux_2_35 (glibc 2.35) while H800 runs Rocky 9.4 / glibc 2.34.
# The wheels were fetched elsewhere, verified by sha256, retagged to
# manylinux_2_34 with `wheel tags` (only isaacsim.robot_motion.lula references
# GLIBC_2.35; RoboDojo uses CuRobo, not Lula) and staged in ${WHEELHOUSE_ISAACSIM}.
python -m pip install 'isaacsim[all,extscache]==5.1.0' \
  --find-links "${WHEELHOUSE_ISAACSIM}"

# glibc 2.34 fix-up: the shipped binaries bind `hypot` to GLIBC_2.35 (only
# libusd_ts.so and the Lula module).  Retarget that requirement to
# hypot@GLIBC_2.2.5, which the host libm provides, then prove the library
# actually loads with immediate binding.
python -m pip install pyelftools
ISAACSIM_PKG_DIR=$(python -c "import sysconfig, os; print(os.path.join(sysconfig.get_paths()['purelib'], 'isaacsim'))")
python "${IMAGEWAM_ROOT}/scripts/patch_isaacsim_glibc235.py" "${ISAACSIM_PKG_DIR}"
USD_LIB_DIR=$(ls -d "${ISAACSIM_PKG_DIR}"/extscache/omni.usd.libs-*/bin | head -n 1)
[[ -f "${USD_LIB_DIR}/libusd_ts.so" ]] || { echo "libusd_ts.so not found under ${ISAACSIM_PKG_DIR}" >&2; exit 1; }
LD_LIBRARY_PATH="${USD_LIB_DIR}:${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" python - "${USD_LIB_DIR}/libusd_ts.so" <<'PY'
import ctypes, os, sys
lib = sys.argv[1]
ctypes.CDLL(lib, mode=os.RTLD_NOW)
print("loads OK with RTLD_NOW:", lib)
PY

python -m pip install \
  numpy==1.26.0 \
  packaging==23.0 \
  typing_extensions==4.12.2 \
  filelock==3.13.1 \
  websockets==12.0 \
  click==8.1.7 \
  psutil==5.9.8 \
  wheel==0.45.1 \
  starlette==0.45.3 \
  scipy==1.15.3 \
  warp-lang==1.11.0 \
  'onnx>=1.18,<1.22' \
  'ipython<9' \
  virtualenv==20.30.0

python -m pip uninstall -y python-discovery 2>/dev/null || true

# Rocky 9 ships GCC 11's libstdc++ (GLIBCXX <= 3.4.29) but isaacsim/kit/libcarb.so
# needs GLIBCXX_3.4.30.  The env already carries libstdcxx 15.x; make the
# dynamic loader see it whenever the env is activated.
# Kit's MDL-SDK (rtx.neuraylib -> iray/libneuray.so) dlopens libGLU.so.1, which
# the H800 compute nodes do not ship; without it the RTX renderer fails to
# create ("failed creating scene renderer") and cameras never produce images.
"${CONDA_BIN}" install -n RoboDojo -y -c conda-forge libglu
# RoboDojo streams camera videos through `ffmpeg ... -vcodec libx264`
# (utils/save_file.py).  The ffmpeg that ends up in the env lacks libx264
# (openh264 only) so every episode dies with BrokenPipe.  imageio-ffmpeg
# (pulled in by IsaacLab) bundles a GPL static ffmpeg with libx264; put it
# first on the env PATH.
IMAGEIO_FFMPEG=$(python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
"${IMAGEIO_FFMPEG}" -hide_banner -encoders 2>/dev/null | grep -q " libx264 " || {
  echo "imageio-ffmpeg binary has no libx264 encoder: ${IMAGEIO_FFMPEG}" >&2
  exit 1
}
if [[ -e "${CONDA_PREFIX}/bin/ffmpeg" && ! -L "${CONDA_PREFIX}/bin/ffmpeg" ]]; then
  mv "${CONDA_PREFIX}/bin/ffmpeg" "${CONDA_PREFIX}/bin/ffmpeg.conda-openh264"
fi
ln -sfn "${IMAGEIO_FFMPEG}" "${CONDA_PREFIX}/bin/ffmpeg"
mkdir -p "${CONDA_PREFIX}/etc/conda/activate.d" "${CONDA_PREFIX}/etc/conda/deactivate.d"
cat >"${CONDA_PREFIX}/etc/conda/activate.d/zz_robodojo_libstdcxx.sh" <<'EOF'
# RoboDojo/Isaac Sim 5.1 on Rocky 9: libcarb.so needs GLIBCXX_3.4.30+, which the
# host /lib64/libstdc++.so.6 (GCC 11) lacks. Prefer the conda-provided libstdc++.
export _ROBODOJO_SAVED_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
EOF
cat >"${CONDA_PREFIX}/etc/conda/deactivate.d/zz_robodojo_libstdcxx.sh" <<'EOF'
export LD_LIBRARY_PATH="${_ROBODOJO_SAVED_LD_LIBRARY_PATH:-}"
[[ -z "${LD_LIBRARY_PATH}" ]] && unset LD_LIBRARY_PATH
unset _ROBODOJO_SAVED_LD_LIBRARY_PATH
EOF

python -m pip show isaacsim torch torchvision torchaudio \
  >"${STATE_DIR}/isaacsim_versions.txt"
printf 'complete\n' >"${STATE_DIR}/isaacsim_install.complete"
