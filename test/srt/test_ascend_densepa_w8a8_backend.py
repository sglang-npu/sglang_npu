"""
Usage:
python3 -m unittest test_ascend_dense_pa_backend.TestAscendDensePaBackend.test_gsm8k
"""

import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_MODEL_NAME_FOR_TEST_MLA,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    run_bench_offline_throughput,
)

DEFAULT_MODEL_NAME_FOR_TEST = "/models/Qwen2.5-7B-Instruct"
DEFAULT_MODEL_NAME_FOR_TEST_MLA = "/models/DeepSeek-V2-Lite-W8A8"
class TestAscendDensePaBackend(CustomTestCase):
    def test_latency(self):
        output_throughput = run_bench_offline_throughput(
            DEFAULT_MODEL_NAME_FOR_TEST,
            [
                "--attention-backend",
                "ascend",
            ],
        )

        print(f"{output_throughput=}")

        if is_in_ci():
            self.assertGreater(output_throughput, 30)

    def test_gsm8k(self):
        model = DEFAULT_MODEL_NAME_FOR_TEST
        base_url = DEFAULT_URL_FOR_TEST
        url = urlparse(base_url)
        process = popen_launch_server(
            model,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend",
                "ascend",
                "--mem-fraction-static",
                0.8,
            ],
        )

        try:
            args = SimpleNamespace(
                num_shots=5,
                data_path=None,
                num_questions=1319,
                max_new_tokens=512,
                parallel=128,
                host=f"http://{url.hostname}",
                port=int(url.port),
            )

            metrics = run_eval_few_shot_gsm8k(args)
            self.assertGreaterEqual(metrics["accuracy"], 0.80)
            self.assertLessEqual(metrics["latency"], 150)
        finally:
            kill_process_tree(process.pid)

class TestAscendMLAW8A8Backend(CustomTestCase):
    def test_latency(self):
        output_throughput = run_bench_offline_throughput(
            DEFAULT_MODEL_NAME_FOR_TEST_MLA,
            [
                "--attention-backend",
                "ascend",
                "--mem-fraction-static",
                0.7,
                "--tp-size",
                "4",
                "--trust-remote-code",
                "--disable-cuda-graph",
                "--quantization",
                "w8a8_int8",
            ],
        )

        print(f"{output_throughput=}")

        if is_in_ci():
            self.assertGreater(output_throughput, 6)

    def test_gsm8k(self):
        model = DEFAULT_MODEL_NAME_FOR_TEST_MLA
        base_url = DEFAULT_URL_FOR_TEST
        url = urlparse(base_url)
        process = popen_launch_server(
            model,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend",
                "ascend",
                "--mem-fraction-static",
                0.7,
                "--tp-size",
                "4",
                "--trust-remote-code",
                "--disable-cuda-graph",
                "--quantization",
                "w8a8_int8",
            ],
        )

        try:
            args = SimpleNamespace(
                num_shots=5,
                data_path=None,
                num_questions=128,
                max_new_tokens=512,
                parallel=128,
                host=f"http://{url.hostname}",
                port=int(url.port),
            )

            metrics = run_eval_few_shot_gsm8k(args)
            self.assertGreaterEqual(metrics["accuracy"], 0.29)
            self.assertGreaterEqual(metrics["output_throughput"], 100)
        finally:
            kill_process_tree(process.pid)

if __name__ == "__main__":
    unittest.main()
