import os
import unittest
from unittest.mock import patch

from langchain_teddynote.logging import langsmith


class TestLangsmithEndpoint(unittest.TestCase):
    def test_preserves_configured_endpoint(self):
        for endpoint in (
            "https://apac.api.smith.langchain.com",
            "https://eu.api.smith.langchain.com",
            "http://localhost:1984",
        ):
            with self.subTest(endpoint=endpoint), patch.dict(
                os.environ,
                {
                    "LANGSMITH_API_KEY": "synthetic-test-key",
                    "LANGSMITH_ENDPOINT": endpoint,
                },
                clear=True,
            ):
                langsmith("test-project")
                self.assertEqual(os.environ["LANGSMITH_ENDPOINT"], endpoint)
                self.assertEqual(os.environ["LANGSMITH_TRACING"], "true")
                self.assertEqual(os.environ["LANGSMITH_PROJECT"], "test-project")

    def test_defaults_to_us_endpoint_when_unset(self):
        with patch.dict(os.environ, {"LANGSMITH_API_KEY": "synthetic-test-key"}, clear=True):
            langsmith("test-project")
            self.assertEqual(
                os.environ["LANGSMITH_ENDPOINT"], "https://api.smith.langchain.com"
            )

    def test_preserves_endpoint_with_legacy_api_key(self):
        endpoint = "https://eu.api.smith.langchain.com"
        with patch.dict(
            os.environ,
            {"LANGCHAIN_API_KEY": "synthetic-test-key", "LANGSMITH_ENDPOINT": endpoint},
            clear=True,
        ):
            langsmith("test-project")
            self.assertEqual(os.environ["LANGSMITH_ENDPOINT"], endpoint)

    def test_missing_api_key_does_not_modify_environment(self):
        environment = {"LANGSMITH_ENDPOINT": "https://apac.api.smith.langchain.com"}
        with patch.dict(os.environ, environment, clear=True):
            langsmith("test-project")
            self.assertEqual(dict(os.environ), environment)

    def test_disabling_tracing_preserves_endpoint(self):
        endpoint = "https://apac.api.smith.langchain.com"
        with patch.dict(os.environ, {"LANGSMITH_ENDPOINT": endpoint}, clear=True):
            langsmith("test-project", set_enable=False)
            self.assertEqual(os.environ["LANGSMITH_ENDPOINT"], endpoint)
            self.assertEqual(os.environ["LANGSMITH_TRACING"], "false")


if __name__ == "__main__":
    unittest.main()
