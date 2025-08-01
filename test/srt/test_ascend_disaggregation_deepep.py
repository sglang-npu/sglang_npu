import unittest
import socket
from types import SimpleNamespace
from urllib.parse import urlparse

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    run_bench_offline_throughput,
)

TEST_MODEL_MATRIX = {
    "/home/runner/.cache/weights/DeepSeek-R1-W8A8": {
        "accuracy": 0.35,
        "latency": 1000,
        "output_throughput": 6,
    },
}

PREFILL_NODE1_IP = "192.168.0.84"
PREFILL_NODE2_IP = "192.168.0.34"
DECODE_NODE1_IP = "192.168.0.93"
DECODE_NODE2_IP = "192.168.0.91"

def launch_decode_node()

def launch_router()


class TestAscend_DISAGGREGATION_DEEPEP(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        os.environ['USE_VLLM_CUSTOM_ALLREDUCE'] = 1
        cls.models = TEST_MODEL_MATRIX.keys()
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.url = urlparse(DEFAULT_URL_FOR_TEST)
        # Need IP to choose disaggregation mode and whether we start router on this node
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        cls.local_ip = s.getsockename()[0]
        s.close()
        if cls.local_ip == PREFILL_NODE1_IP:
            launch_prefill_node(PREFILL_NODE1_IP)
            launch_router()
        elif cls.local_ip == PREFILL_NODE2_IP:
            launch_prefill_node(PREFILL_NODE2_IP)
        elif cls.local_ip == DECODE_NODE1_IP or cls.local_ip == DECODE_NODE2_IP:
            launch_decode_node()

    def launch_prefill_node(prefill_node_ip)
        self.common_args = [
                "--disaggregation-mode",
                "prefill",
                "--host",
                prefill_node_ip,
                "--port",
                8000,
                "--disaggregation-bootstrap-port",
                8996,
                "--dist-init-addr",
                prefill_node_ip + ":5000",
                "--nnodes",
                1,
                "--node-rank",
                0,
                "--trust-remote-code",
                "--mem-fraction-static",
                0.8,
                "--attention-backend",
                "ascend",
                "--quantization",
                "w8a8_int8",
                "--tp-size",
                16,
                "--dp-size",
                1,
                "--device",
                "npu",
                "--disaggregation-transfer",
                "backend ascned"
            ]
            
    def test_a_gsm8k(self):
        if local_ip == PREFILL_NODE1_IP:
            try:
                args = SimpleNamespace(
                    num_shots=5,
                    data_path=None,
                    num_questions=1319,
                    max_new_tokens=512,
                    parallel=128,
                    host=f"http://{self.url.hostname}",
                    port=int(self.url.port),
                )

                metrics = run_eval_few_shot_gsm8k(args)
                self.assertGreaterEqual(
                    metrics["accuracy"],
                    TEST_MODEL_MATRIX[model]["accuracy"],
                )
                self.assertLessEqual(
                    metrics["latency"],
                    TEST_MODEL_MATRIX[model]["latency"],
                )
            finally:
                kill_process_tree(process.pid)

