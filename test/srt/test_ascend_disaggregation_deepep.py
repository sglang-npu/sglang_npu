import os
import socket
import subprocess
import time
import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

import requests

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

MODEL_PATH = "/home/runner/.cache/weights/DeepSeek-R1-W8A8"

PREFILL_NODE1_IP = "192.168.0.84"
PREFILL_NODE2_IP = "192.168.0.34"
DECODE_NODE1_IP = "192.168.0.93"
DECODE_NODE2_IP = "192.168.0.91"


def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    local_ip = s.getsockname()[0]
    s.close()
    return local_ip


class TestAscend_DISAGGREGATION_DEEPEP(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        cls.process = None
        cls.local_ip = get_local_ip()

    def launch_prefill_node(self):
        if self.local_ip == PREFILL_NODE1_IP:
            dist_init_addr = f"{PREFILL_NODE1_IP}:5000"
            disaggregation_bootstrap_port = 8995
        elif self.local_ip == PREFILL_NODE2_IP:
            dist_init_addr = f"{PREFILL_NODE2_IP}:5000"
            disaggregation_bootstrap_port = 8996

        common_args = [
            "--disaggregation-mode",
            "prefill",
            "--trust-remote-code",
            "--dist-init-addr",
            dist_init_addr,
            "--disaggregation-bootstrap-port",
            disaggregation_bootstrap_port,
            "--nnodes",
            1,
            "--node-rank",
            0,
            "--tp-size",
            16,
            "--dp-size",
            1,
            "--mem-fraction-static",
            0.8,
            "--attention-backend",
            "ascend",
            "--quantization",
            "w8a8_int8",
            "--disaggregation-transfer-backend",
            "ascend",
        ]

        self.process = popen_launch_server(
            MODEL_PATH,
            f"http://{self.local_ip}:{8000}",
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                *common_args,
            ],
        )

    def launch_decode_node(self):
        dist_init_addr = f"{DECODE_NODE1_IP}:5000"

        if self.local_ip == DECODE_NODE1_IP:
            node_rank = 0
        elif self.local_ip == DECODE_NODE2_IP:
            node_rank = 1

        common_args = [
            "--disaggregation-mode",
            "decode",
            "--trust-remote-code",
            "--dist-init-addr",
            dist_init_addr,
            "--nnodes",
            2,
            "--node-rank",
            node_rank,
            "--tp-size",
            32,
            "--dp-size",
            4,
            "--mem-fraction-static",
            0.9,
            "--attention-backend",
            "ascend",
            "--quantization",
            "w8a8_int8",
            "--disaggregation-transfer-backend",
            "ascend",
            "--enable-deepep-moe",
            "--enable-dp-attention",
            "--deepep-mode",
            "low_latency",
            "--enable-dp-lm-head",
            "--moe-dense-tp-size",
            1,
        ]

        self.process = popen_launch_server(
            MODEL_PATH,
            f"http://{self.local_ip}:{8001}",
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                *common_args,
            ],
        )

    def checkout_port(self, host, port, timeout=3):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except Exception as e:
            print(f"Error: {e}")
            return False

    def launch_router(self):
        NODE_IP_LIST = [PREFILL_NODE1_IP, PREFILL_NODE2_IP, DECODE_NODE1_IP]
        while True:
            success_nodes = 0
            for ip in NODE_IP_LIST:
                if ip == PREFILL_NODE1_IP or ip == PREFILL_NODE2_IP:
                    port = 8000
                elif ip == DECODE_NODE1_IP:
                    port = 8001
                if self.checkout_port(ip, port):
                    print(f"{ip=} {port} is open")
                    success_nodes = success_nodes + 1
                else:
                    print(f"{ip=} {port} is closed")
            if success_nodes == len(NODE_IP_LIST):
                break
            time.sleep(3)

        prefill1_url = f"http://{PREFILL_NODE1_IP}:8000"
        prefill2_url = f"http://{PREFILL_NODE2_IP}:8000"
        decode_url = f"http://{DECODE_NODE1_IP}:8001"

        lb_command = [
            "python3",
            "-m",
            "sglang.srt.disaggregation.mini_lb",
            "--prefill",
            prefill1_url,
            prefill2_url,
            "--decode",
            decode_url,
            "--host",
            PREFILL_NODE1_IP,
            "--port",
            "6688",
            "--prefill-bootstrap-ports",
            "8995",
            "8996",
        ]

        print(f"Starting router, {lb_command=}")
        subprocess.Popen(lb_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def wait_router_ready(self, url, timeout=300):
        start_time = time.perf_counter()
        while True:
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    print(f"Router {url} is ready!")
                    return
            except Exception:
                pass

            if time.perf_counter() - start_time > timeout:
                raise RuntimeError(f"Server {url} failed to start in {timeout}s")
            time.sleep(10)

    def test_a_gsm8k(self):
        import threading

        if self.local_ip == PREFILL_NODE1_IP:
            sglang_thread = threading.Thread(target=self.launch_prefill_node)
            sglang_thread.start()
            router_thread = threading.Thread(target=self.launch_router)
            router_thread.start()
            self.wait_router_ready(f"http://{PREFILL_NODE1_IP}:6688" + "/health")

            print(f"Starting benchmark ......")
            try:
                args = SimpleNamespace(
                    num_shots=5,
                    data_path=None,
                    num_questions=100,
                    max_new_tokens=512,
                    parallel=128,
                    host=f"http://{PREFILL_NODE1_IP}",
                    port=6688,
                )

                metrics = run_eval_few_shot_gsm8k(args)
                self.assertGreaterEqual(
                    metrics["accuracy"],
                    0.90,
                )
                self.assertLessEqual(
                    metrics["latency"],
                    60,
                )
            finally:
                kill_process_tree(self.process.pid)

        elif self.local_ip == PREFILL_NODE2_IP:
            self.launch_prefill_node()
            while True:
                if not self.checkout_port(PREFILL_NODE1_IP, 8000):
                    time.sleep(10)
                else:
                    break
            while True:
                if self.checkout_port(PREFILL_NODE1_IP, 8000):
                    time.sleep(10)
                else:
                    kill_process_tree(self.process.pid)
                    break

        elif self.local_ip == DECODE_NODE1_IP or self.local_ip == DECODE_NODE2_IP:
            self.launch_decode_node()
            while True:
                if not self.checkout_port(PREFILL_NODE1_IP, 8000):
                    time.sleep(10)
                else:
                    break
            while True:
                if self.checkout_port(PREFILL_NODE1_IP, 8000):
                    time.sleep(10)
                else:
                    kill_process_tree(self.process.pid)
                    break


if __name__ == "__main__":
    unittest.main()
