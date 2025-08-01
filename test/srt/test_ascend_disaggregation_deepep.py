import os
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

def launch_prefill_node(prefill_node_ip):
    common_args = [
        "--disaggregation-mode",
        "prefill",
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
        "--disaggregation-transfer-backend",
        "ascend"
    ]
    for model in TEST_MODEL_MATRIX.keys():
        process = popen_launch_server(
            model,
            f"http://{prefill_node_ip}:{5000}",
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                *common_args,
            ],
            pd_separated=True
        )

def launch_decode_node(decode_node_ip):
    common_args = [
        "--disaggregation-mode",
        "decode",
        "--dist-init-addr",
        DECODE_NODE1_IP + ":5000",
        "--nnodes",
        2,
        "--node-rank",
        0,
        "--trust-remote-code",
        "--mem-fraction-static",
        0.9,
        "--max-running-requests",
        1536,
        "--attention-backend",
        "ascend",
        "--quantization",
        "w8a8_int8",
        "--tp-size",
        32,
        "--dp-size",
        16,
        "--device",
        "npu",
        "--enable-deepep-moe",
        "--enable-dp-attention"
        "--enable-dp-lm-head",
        "--moe-dense-tp-size",
        1,
        "--deeped-mode",
        "low_latency"
        "--disaggregation-transfer-backend",
        "ascned"
    ]
    for model in TEST_MODEL_MATRIX.keys():
        process = popen_launch_server(
            model,
            f"http://{decode_node_ip}:{8001}",
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                *common_args,
            ],
            pd_separated=True
        )

def launch_router():
    import time
    import requests
    import subprocess
    
    NODE_IP_LIST = [PREFILL_NODE1_IP, PREFILL_NODE2_IP, DECODE_NODE1_IP, DECODE_NODE2_IP]
    NODE_PORT = 5000

    def checkout_port(host, port , timeout=3):
        try: 
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except Exception as e:
            print(f"Error: {e}")
            return False

    def wait_server_ready(url, timeout=60):
        start_time = time.perf_counter()
        while True:
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    print(f"Server {url} is ready")
                    return
            except Exception:
                pass

            if time.perf_counter() - start_time > timeout:
                raise RuntimeError(f"Server {url} failed to start in {timeout}s")
            time.sleep(1)

    while True:
        success_nodes = 0
        for ip in NODE_IP_LIST:
            if checkout_port(ip, NODE_PORT):
                print(f"{ip=} {NODE_PORT} is open")
                success_nodes = success_nodes + 1
            else:
                print(f"{ip=} {NODE_PORT} is closed")
                time.sleep(1)
        if success_nodes == len(NODE_IP_LIST):
            break
    
    prefill_url = f"http://{PREFILL_NODE1_IP}:{NODE_PORT}"
    decode_url = f"http://{DECODE_NODE1_IP}:{NODE_PORT}"
    lb_host = PREFILL_NODE1_IP
    lb_port = 9000
    lb_url = f"http://{lb_host}:{lb_port}"

    lb_command = [
            "python3",
            "-m",
            "sglang.srt.disaggregation.mini_lb",
            "--prefill",
            prefill_url,
            "--decode",
            decode_url,
            "--host",
            lb_host,
            "--port",
            lb_port,
        ]

    print("Starting load balancer:", " ".join(lb_command))
    subprocess.Popen(
        lb_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    wait_server_ready(lb_url + "/health")()


class TestAscend_DISAGGREGATION_DEEPEP(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        os.environ['USE_VLLM_CUSTOM_ALLREDUCE'] = '1'
        cls.models = TEST_MODEL_MATRIX.keys()
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.url = urlparse(DEFAULT_URL_FOR_TEST)
        # Need IP to choose disaggregation mode and whether we start router on this node
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        cls.local_ip = s.getsockname()[0]
        s.close()
        if cls.local_ip == PREFILL_NODE1_IP:
            launch_prefill_node(PREFILL_NODE1_IP)
            launch_router()
        elif cls.local_ip == PREFILL_NODE2_IP:
            launch_prefill_node(PREFILL_NODE2_IP)
        elif cls.local_ip == DECODE_NODE1_IP or cls.local_ip == DECODE_NODE2_IP:
            launch_decode_node()
    
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

if __name__ == "__main__":
    unittest.main()

