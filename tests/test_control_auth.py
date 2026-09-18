import os
import unittest
from unittest.mock import patch
from app import app
from modules.execution.auth import TradingControlAuth


class TestTradingControlAuth(unittest.TestCase):
    """
    Tests security hardening of the trading control plane:
    - POST /api/trading/order
    - POST /api/trading/halt
    - POST /api/trading/resume
    - POST /api/trading/reconcile
    """

    def setUp(self):
        self.client = app.test_client()
        self.secret = "super_secret_quant_token_987654321"

    def test_unauthenticated_request_rejected_when_secret_set(self):
        """When TRADING_API_SECRET is set, requests without token are rejected with 401."""
        with patch.dict(os.environ, {"TRADING_API_SECRET": self.secret, "TRADING_MODE": "paper"}):
            # Order endpoint
            resp = self.client.post("/api/trading/order", json={"symbol": "2330", "quantity": 1000})
            self.assertEqual(resp.status_code, 401)
            self.assertFalse(resp.json["success"])
            self.assertNotIn(self.secret, resp.get_data(as_text=True))

            # Resume endpoint
            resp = self.client.post("/api/trading/resume", json={"operator_id": "ALGO_ADMIN", "confirmation": "CONFIRM_RESUME", "reason": "Testing"})
            self.assertEqual(resp.status_code, 401)
            self.assertFalse(resp.json["success"])

            # Halt endpoint
            resp = self.client.post("/api/trading/halt", json={"reason": "Emergency"})
            self.assertEqual(resp.status_code, 401)

            # Reconcile endpoint
            resp = self.client.post("/api/trading/reconcile")
            self.assertEqual(resp.status_code, 401)

    def test_invalid_token_rejected(self):
        """Invalid token is rejected with 401 and constant-time check."""
        with patch.dict(os.environ, {"TRADING_API_SECRET": self.secret, "TRADING_MODE": "paper"}):
            headers = {"X-API-Key": "wrong_token_xyz"}
            resp = self.client.post("/api/trading/halt", json={"reason": "Emergency"}, headers=headers)
            self.assertEqual(resp.status_code, 401)
            self.assertIn("Unauthorized", resp.json["error"])
            self.assertNotIn(self.secret, resp.get_data(as_text=True))

    def test_valid_token_accepted_halt_and_bearer_auth(self):
        """Valid token via Authorization: Bearer or X-API-Key is accepted."""
        with patch.dict(os.environ, {"TRADING_API_SECRET": self.secret, "TRADING_MODE": "paper"}):
            headers = {"Authorization": f"Bearer {self.secret}"}
            resp = self.client.post("/api/trading/halt", json={"reason": "Manual drill", "operator_id": "SEC_ADMIN"}, headers=headers)
            self.assertEqual(resp.status_code, 200)
            self.assertTrue(resp.json["success"])

    def test_resume_requires_explicit_confirmation(self):
        """Resuming trading requires explicit 'CONFIRM_RESUME' token from authorized operator."""
        with patch.dict(os.environ, {"TRADING_API_SECRET": self.secret, "TRADING_MODE": "paper"}):
            headers = {"X-API-Key": self.secret}

            # Missing confirmation token
            resp = self.client.post(
                "/api/trading/resume",
                json={"operator_id": "SEC_ADMIN", "reason": "Drill ended"},
                headers=headers,
            )
            self.assertEqual(resp.status_code, 400)
            self.assertIn("CONFIRM_RESUME", resp.json["error"])

            # Correct confirmation token
            resp = self.client.post(
                "/api/trading/resume",
                json={"operator_id": "SEC_ADMIN", "confirmation": "CONFIRM_RESUME", "reason": "Drill ended"},
                headers=headers,
            )
            self.assertEqual(resp.status_code, 200)
            self.assertTrue(resp.json["success"])

    def test_live_mode_fails_closed_when_secret_unset(self):
        """If TRADING_MODE=live but TRADING_API_SECRET is unset, all control endpoints fail closed with 403."""
        env = {"TRADING_MODE": "live"}
        # Remove any secret from environment
        with patch.dict(os.environ, env, clear=False):
            if "TRADING_API_SECRET" in os.environ:
                del os.environ["TRADING_API_SECRET"]
            if "QUANT_TRADING_SECRET" in os.environ:
                del os.environ["QUANT_TRADING_SECRET"]

            resp = self.client.post("/api/trading/order", json={"symbol": "2330", "quantity": 1000})
            self.assertEqual(resp.status_code, 403)
            self.assertIn("Live trading control plane is locked", resp.json["error"])

    def test_live_mode_manual_order_disabled_by_default(self):
        """In LIVE mode, arbitrary manual orders are blocked unless ENABLE_MANUAL_LIVE_ORDERS=true."""
        with patch.dict(os.environ, {
            "TRADING_API_SECRET": self.secret,
            "TRADING_MODE": "live",
            "ENABLE_MANUAL_LIVE_ORDERS": "false"
        }):
            headers = {"X-API-Key": self.secret}
            resp = self.client.post(
                "/api/trading/order",
                json={"symbol": "2330", "side": "BUY", "quantity": 1000},
                headers=headers
            )
            self.assertEqual(resp.status_code, 403)
            self.assertIn("disabled in LIVE trading mode", resp.json["error"])

    def test_credentials_never_leaked_in_output(self):
        """Ensures that even on errors or unauthorized attempts, credentials are never printed."""
        with patch.dict(os.environ, {"TRADING_API_SECRET": self.secret, "TRADING_MODE": "paper"}):
            resp = self.client.post("/api/trading/order", headers={"X-API-Key": "attacker_probe"})
            body = resp.get_data(as_text=True)
            self.assertNotIn(self.secret, body)
            self.assertNotIn("attacker_probe", body)


if __name__ == "__main__":
    unittest.main()
