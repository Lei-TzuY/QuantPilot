"""
Integration test for scripts/run_shadow_soak.py.
Executes the soak runner and verifies report invariants, provenance, and counter consistency.
"""
import os
import unittest

from scripts.run_shadow_soak import run_soak_session


class TestShadowSoakRunner(unittest.TestCase):
    def test_run_soak_session_reproducibility_and_invariants(self):
        symbols = ["2330", "2454"]
        date = "2026-09-18"

        # Execute full accelerated session (270 minutes x 2 symbols)
        report = run_soak_session(
            symbols=symbols,
            session_date=date,
            mode="accelerated_simulation",
        )

        self.assertIsNotNone(report)
        self.assertEqual(report.soak_mode, "ACCELERATED_SIMULATION")
        self.assertEqual(sorted(report.symbols), sorted(symbols))
        self.assertEqual(report.session_date, date)

        # Total ticks: 270 minutes * 4 ticks * 2 symbols + 2 boundary ticks = 2,162 ticks
        self.assertEqual(report.market_data_received, 2162)
        self.assertEqual(report.ticks_processed, 2162)
        self.assertEqual(report.ticks_rejected, 0)

        # Bars generated: 270 minutes * 2 symbols = 540 finalized bars!
        self.assertEqual(report.bars_generated, 540, "bars_generated MUST be 540 total finalized bars, NOT 2!")

        # Report self-consistency invariants
        violations = report.validate_invariants()
        self.assertEqual(violations, [], f"Report must have 0 invariant violations: {violations}")

        # Orders & Fills consistency
        self.assertLessEqual(report.orders_filled, report.orders_submitted)
        self.assertGreaterEqual(report.total_fills, report.orders_filled)

        # Provenance checks
        self.assertTrue(len(report.git_commit_sha) > 0)
        self.assertIn("soak_mode", report.provenance)
        self.assertEqual(report.queue_mode, "asynchronous")


if __name__ == "__main__":
    unittest.main()
