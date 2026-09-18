"""
Shadow Session Report and Observability REST API Tests.
Verifies canonical JSON and Markdown report generation, telemetry metrics,
and read-only endpoints in app.py.
"""
from datetime import datetime, timedelta
import json
import os
import shutil
import tempfile
import unittest

from app import app, execution_engine
from modules.execution.events import BarEvent, SignalEvent, TickEvent
from modules.monitoring.session_report import ShadowSessionReporter


class TestShadowSessionReportAndAPI(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.reports_dir = os.path.join(self.test_dir, "reports")
        self.client = app.test_client()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_structured_report_generation_and_persistence(self):
        reporter = ShadowSessionReporter(reports_dir=self.reports_dir)

        start_time = datetime(2026, 9, 18, 9, 0, 0)
        end_time = datetime(2026, 9, 18, 13, 30, 0)

        report = reporter.build_report(
            session_id="SHADOW-20260918-001",
            symbols=["2330", "2454"],
            start_time=start_time,
            end_time=end_time,
            ticks_received=15000,
            ticks_valid=14995,
            ticks_rejected=5,
            bars_count=270,
            signals_count=8,
            risk_approvals=6,
            risk_rejections=2,
            orders_submitted=6,
            orders_filled=6,
            orders_rejected=0,
            orders_cancelled=0,
            total_fills=6,
            gross_pnl=52000.0,
            commission=1200.0,
            tax=3800.0,  # 0.15% qualifying day trade rate applied
            slippage=400.0,
            queue_max_depth=42,
            queue_overflow_count=0,
            disconnect_count=0,
            reconciliation_count=1,
            kill_switch_events=0,
            data_health_incidents=5,
        )

        json_path, md_path = reporter.persist_report(report)

        self.assertTrue(os.path.exists(json_path))
        self.assertTrue(os.path.exists(md_path))

        # Check JSON content
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.assertEqual(data["session_id"], "SHADOW-20260918-001")
            self.assertEqual(data["ticks_processed"], 14995)
            self.assertEqual(data["gross_pnl"], 52000.0)
            self.assertEqual(data["net_pnl"], 46600.0)  # 52000 - 1200 - 3800 - 400
            self.assertEqual(data["total_statutory_tax"], 3800.0)

        # Check Markdown content
        with open(md_path, "r", encoding="utf-8") as f:
            md_text = f.read()
            self.assertIn("# QuantPilot Shadow Trading Session Report", md_text)
            self.assertIn("SHADOW-20260918-001", md_text)
            self.assertIn("46,600.00 TWD", md_text)

    def test_observability_api_endpoints(self):
        # 1. GET /api/trading/health
        res = self.client.get("/api/trading/health")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertIn("is_healthy", data)
        self.assertIn("supervisor_state", data)
        self.assertIn("broker_connected", data)

        # 2. GET /api/trading/session
        res = self.client.get("/api/trading/session")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertIn("session", data)
        self.assertIn("trading_mode", data)

        # 3. GET /api/trading/latency
        res = self.client.get("/api/trading/latency")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertIn("stages", data)

        # 4. GET /api/trading/data-health
        res = self.client.get("/api/trading/data-health")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertIn("data_health", data)

        # 5. GET /api/trading/queue
        res = self.client.get("/api/trading/queue")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertIn("queue", data)
        self.assertIn("capacity", data["queue"])
        self.assertIn("current_depth", data["queue"])

        # 6. GET /api/trading/shadow/report
        res = self.client.get("/api/trading/shadow/report")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertIn("report", data)
        self.assertIn("session_id", data["report"])


if __name__ == "__main__":
    unittest.main()
