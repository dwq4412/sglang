import unittest

import torch

from minisgl.disaggregation.runtime import _tensor_bytes


class TestRuntimeTensorSerialization(unittest.TestCase):
    def test_bfloat16_serializes_as_raw_storage_bytes(self):
        tensor = torch.tensor([1.0, -2.5, 3.25], dtype=torch.bfloat16)

        raw = _tensor_bytes(tensor)
        restored = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).clone()

        self.assertEqual(len(raw), tensor.numel() * tensor.element_size())
        self.assertTrue(torch.equal(restored, tensor))


if __name__ == "__main__":
    unittest.main()