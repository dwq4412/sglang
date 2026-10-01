import json
import types
import unittest
from unittest.mock import patch

import torch

from minisgl.disaggregation.nixl_backend import NixlEndpoint


class _FakeAgent:
    def __init__(self, name, config):
        self.name = name
        self.states = ["PROC", "DONE"]
        self.cleaned = 0

    def create_backend(self, name, params):
        self.backend = name

    def register_memory(self, addresses, kind):
        self.registered = addresses
        return object()

    def get_agent_metadata(self):
        return b"agent-metadata"

    def add_remote_agent(self, metadata):
        self.remote = metadata

    def get_xfer_descs(self, addresses, kind):
        return addresses

    def initialize_xfer(self, direction, source, destination, peer):
        self.xfer = (direction, source, destination, peer)
        return object()

    def transfer(self, handle):
        return "PROC"

    def check_xfer_state(self, handle):
        return self.states.pop(0)

    def release_xfer_handle(self, handle):
        self.released = True

    def deregister_memory(self, handle):
        self.deregistered = True

    def cleanup(self):
        self.cleaned += 1


class _ModernFakeAgent(_FakeAgent):
    def create_xfer_req(self, operation, source, destination, peer):
        self.xfer = (operation, source, destination, peer)
        return object()


class TestNixlEndpoint(unittest.TestCase):
    def setUp(self):
        self.pool = types.SimpleNamespace(
            num_layers=2,
            buffer=torch.empty(2, 2, 4, 2, 1, 2, dtype=torch.float16),
        )
        self.agent = _FakeAgent("local", object())

    def _endpoint(self):
        return NixlEndpoint.__new__(NixlEndpoint)

    def test_metadata_validation_and_page_mapping(self):
        with patch(
            "minisgl.disaggregation.nixl_backend._load_api",
            return_value=(lambda name, config: self.agent, lambda **_: object()),
        ):
            endpoint = NixlEndpoint(self.pool, 1, 1.0)
            metadata = json.loads(endpoint.metadata())
            self.assertEqual(metadata["v"], 1)
            self.assertEqual(len(endpoint.pages([0, 2])), 8)
            with self.assertRaises(ValueError):
                endpoint.add_peer(b'{"v": 1}')
            peer = endpoint.add_peer(endpoint.metadata())
            self.assertEqual(peer, "local")
            endpoint.write(peer, endpoint.pages([0]), endpoint.pages([1]))
            endpoint.cleanup()
            endpoint.cleanup()
            self.assertEqual(self.agent.cleaned, 1)
            self.assertTrue(self.agent.released)

    def test_modern_transfer_api_uses_three_tuple_descriptors(self):
        agent = _ModernFakeAgent("local", object())
        with patch(
            "minisgl.disaggregation.nixl_backend._load_api",
            return_value=(lambda name, config: agent, lambda **_: object()),
        ), patch(
            "minisgl.disaggregation.nixl_backend._read_operation", return_value="READ"
        ):
            endpoint = NixlEndpoint(self.pool, 1, 1.0)
            endpoint.add_peer(endpoint.metadata())
            endpoint.write("local", endpoint.pages([0]), endpoint.pages([1]))

        operation, local_destination, remote_source, peer = agent.xfer
        self.assertEqual(operation, "READ")
        self.assertTrue(all(len(descriptor) == 3 for descriptor in local_destination))
        self.assertTrue(all(len(descriptor) == 3 for descriptor in remote_source))
        self.assertEqual(
            local_destination[0][0], endpoint.pages([1])[0].address
        )
        self.assertEqual(remote_source[0][0], endpoint.pages([0])[0].address)
        self.assertEqual(peer, "local")


if __name__ == "__main__":
    unittest.main()
