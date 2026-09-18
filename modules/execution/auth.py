import os
import hmac
import functools
import logging
from typing import Optional, Tuple
from flask import request, jsonify

logger = logging.getLogger("QuantPilot.Auth")


class TradingControlAuth:
    """
    Security and access control for trading control plane APIs.
    Enforces fail-closed semantics for live execution and timing-safe comparison.
    """

    @classmethod
    def get_secret(cls) -> Optional[str]:
        return os.getenv("TRADING_API_SECRET") or os.getenv("QUANT_TRADING_SECRET")

    @classmethod
    def get_trading_mode(cls) -> str:
        return os.getenv("TRADING_MODE", "paper").strip().lower()

    @classmethod
    def is_manual_live_order_enabled(cls) -> bool:
        return os.getenv("ENABLE_MANUAL_LIVE_ORDERS", "false").strip().lower() in ("true", "1", "yes")

    @classmethod
    def verify_token(cls, token: Optional[str]) -> Tuple[bool, str]:
        """
        Validates provided token using constant-time comparison.
        Returns (is_valid, reason).
        """
        trading_mode = cls.get_trading_mode()
        secret = cls.get_secret()

        # Fail closed for live mode if secret is unconfigured
        if trading_mode == "live" and not secret:
            logger.error("Security violation: Live trading active but TRADING_API_SECRET is not configured.")
            return False, "Live trading control plane is locked: TRADING_API_SECRET is unconfigured"

        # In paper mode, if no secret is set, check if auth is explicitly required
        if not secret:
            if os.getenv("REQUIRE_CONTROL_AUTH", "false").strip().lower() in ("true", "1", "yes"):
                return False, "Control plane authentication required: TRADING_API_SECRET is unconfigured"
            # Paper mode local development convenience
            return True, "Development mode (no secret configured)"

        if not token:
            return False, "Missing authorization credentials"

        # Timing-safe constant-time comparison
        try:
            is_valid = hmac.compare_digest(token.strip().encode("utf-8"), secret.strip().encode("utf-8"))
            if not is_valid:
                return False, "Invalid authentication credentials"
            return True, "Authenticated"
        except Exception:
            return False, "Authentication validation error"

    @classmethod
    def extract_token_from_request(cls) -> Optional[str]:
        """Extracts token from X-API-Key or Authorization header without logging it."""
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return api_key.strip()

        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            return auth_header[7:].strip()

        return None


def require_control_auth(f):
    """
    Decorator for sensitive control endpoints:
    - POST /api/trading/halt
    - POST /api/trading/resume
    - POST /api/trading/order
    - POST /api/trading/reconcile
    """
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        token = TradingControlAuth.extract_token_from_request()
        is_valid, reason = TradingControlAuth.verify_token(token)
        if not is_valid:
            # Note: Never return or log the token in the response or logs
            return jsonify({
                "success": False,
                "error": f"Unauthorized: {reason}",
            }), 401 if "Missing" in reason or "Invalid" in reason else 403

        return f(*args, **kwargs)
    return decorated_function
