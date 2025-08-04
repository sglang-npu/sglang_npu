ENV_DIR="/home/runner/.cache/env_sglang.sh"
WORK_DIR="/home/runner/actions-runner/_work/sglang_npu/sglang_npu/test/srt"
docker exec \
  -w "$WORK_DIR" \
  sglang_ci_a3 \
  /bin/bash -c "source $ENV_DIR && exec \"$*\""
