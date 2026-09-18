"""
Latency Instrumentation and Shadow Session Supervisor Tests.
Verifies microsecond percentile distributions and institutional lifecycle state invariants.
"""
from datetime import datetime, timedelta
import unittest

from modules.execution.supervisor import ReadinessChecklist, ShadowSessionSupervisor, SupervisorState
from modules.monitoring.latency import LatencyTracker


class TestLatencyAndSupervisor(unittest.TestCase):

    def test_latency_tracker_percentiles_calculation(self):
        tracker = LatencyTracker(reservoir_size=1000)

        # Record 100 samples from 1.0ms to 100.0ms
        for i in range(1, 101):
            tracker.record_stage_latency("market_data", float(i))

        p = tracker.get_percentiles("market_data")
        self.assertEqual(p.count, 100)
        self.assertEqual(p.min_ms, 1.0)
        self.assertEqual(p.max_ms, 100.0)
        self.assertAlmostEqual(p.p50_ms, 50.5, delta=0.5)
        self.assertAlmostEqual(p.p95_ms, 95.0, delta=1.0)
        self.assertAlmostEqual(p.p99_ms, 99.0, delta=1.0)

    def test_latency_tick_pipeline_recording(self):
        tracker = LatencyTracker()
        t_ex = datetime(2026, 9, 18, 9, 0, 0, 0)
        t_recv = t_ex + timedelta(milliseconds=12)
        t_enq = t_recv + timedelta(milliseconds=2)
        t_deq = t_enq + timedelta(milliseconds=5)

        tracker.record_tick_latencies(
            exchange_ts=t_ex,
            receive_ts=t_recv,
            enqueue_ts=t_enq,
            dequeue_ts=t_deq,
        )

        p_mkt = tracker.get_percentiles("market_data")
        self.assertEqual(p_mkt.count, 1)
        self.assertEqual(p_mkt.p50_ms, 12.0)

        p_q = tracker.get_percentiles("queue")
        self.assertEqual(p_q.count, 1)
        self.assertEqual(p_q.p50_ms, 5.0)

    def test_supervisor_legal_lifecycle_transitions(self):
        sup = ShadowSessionSupervisor(session_id="TEST-SUPERVISOR-01")
        self.assertEqual(sup.current_state, SupervisorState.INITIALIZING)

        # INITIALIZING -> CONNECTING
        self.assertTrue(sup.transition_to(SupervisorState.CONNECTING, "Connecting broker"))
        self.assertEqual(sup.current_state, SupervisorState.CONNECTING)

        # CONNECTING -> SYNCING
        self.assertTrue(sup.transition_to(SupervisorState.SYNCING, "Syncing state"))
        self.assertEqual(sup.current_state, SupervisorState.SYNCING)

        # SYNCING -> READY
        self.assertTrue(sup.transition_to(SupervisorState.READY, "Ready to start"))
        self.assertEqual(sup.current_state, SupervisorState.READY)

        # Illegal transition: READY cannot jump directly to CLOSED
        with self.assertRaises(ValueError):
            sup.transition_to(SupervisorState.INITIALIZING)

    def test_supervisor_pre_running_checklist_invariants(self):
        sup = ShadowSessionSupervisor(session_id="TEST-INVARIANTS-01")
        sup.transition_to(SupervisorState.CONNECTING)
        sup.transition_to(SupervisorState.SYNCING)
        sup.transition_to(SupervisorState.READY)

        # Checklist NOT passed yet -> transition to RUNNING must fail!
        self.assertFalse(sup.can_transition_to(SupervisorState.RUNNING))
        with self.assertRaises(RuntimeError):
            sup.transition_to(SupervisorState.RUNNING)

        # Fulfill all checklist invariants
        sup.update_checklist(
            broker_connected=True,
            market_clock_valid=True,
            journal_available=True,
            strategies_warmed_up=True,
            risk_engine_ready=True,
            kill_switch_active=False,
            reconciliation_clean=True,
        )

        # Now all passed -> transition to RUNNING succeeds!
        self.assertTrue(sup.can_transition_to(SupervisorState.RUNNING))
        self.assertTrue(sup.transition_to(SupervisorState.RUNNING, "Starting trading"))
        self.assertEqual(sup.current_state, SupervisorState.RUNNING)

    def test_supervisor_degraded_state_and_recovery(self):
        sup = ShadowSessionSupervisor(session_id="TEST-DEGRADED-01")
        sup.update_checklist(
            broker_connected=True,
            market_clock_valid=True,
            journal_available=True,
            strategies_warmed_up=True,
            risk_engine_ready=True,
            kill_switch_active=False,
            reconciliation_clean=True,
        )
        sup.transition_to(SupervisorState.CONNECTING)
        sup.transition_to(SupervisorState.SYNCING)
        sup.transition_to(SupervisorState.READY)
        sup.transition_to(SupervisorState.RUNNING)

        # Data incident triggers DEGRADED
        sup.trigger_degraded("Market quote gap detected")
        self.assertEqual(sup.current_state, SupervisorState.DEGRADED)

        # Incident resolved -> recover to RUNNING
        sup.recover_to_running("Quote feed recovered")
        self.assertEqual(sup.current_state, SupervisorState.RUNNING)


if __name__ == "__main__":
    unittest.main()
