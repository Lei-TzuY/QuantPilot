"""
Shadow Session Structured Reporting
Generates audit-ready JSON and human-readable Markdown summaries.
Persists execution PnL, statutory taxes, broker fees, queue metrics,
latency percentiles, invariant validation, and git/environment provenance.
"""
from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import logging
import os
import subprocess
import threading
from typing import Any, Dict, List, Optional, Tuple

from modules.monitoring.latency import LatencySnapshot

logger = logging.getLogger("QuantPilot.SessionReport")


def get_git_provenance() -> Tuple[str, bool]:
    """Retrieves current git commit SHA and dirty working tree status safely."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL).decode().strip()
        return commit, len(status) > 0
    except Exception:
        return "unknown", False


@dataclass
class ShadowSessionData:
    session_id: str
    session_date: str
    symbols: List[str]
    start_time: str
    end_time: str
    market_session_start: str = ""
    market_session_end: str = ""
    process_start: str = ""
    process_end: str = ""
    first_tick_time: str = ""
    last_tick_time: str = ""
    market_data_received: int = 0
    ticks_processed: int = 0
    ticks_rejected: int = 0
    bars_generated: int = 0
    signals_generated: int = 0
    risk_approvals: int = 0
    risk_rejections: int = 0
    orders_submitted: int = 0
    orders_filled: int = 0
    orders_rejected: int = 0
    orders_cancelled: int = 0
    total_fills: int = 0
    gross_pnl: float = 0.0
    total_commission: float = 0.0
    total_statutory_tax: float = 0.0
    total_slippage: float = 0.0
    net_pnl: float = 0.0
    max_drawdown_pct: float = 0.0
    queue_max_depth: int = 0
    queue_overflow_count: int = 0
    disconnect_count: int = 0
    reconciliation_count: int = 0
    kill_switch_events: int = 0
    data_health_incidents: int = 0
    exceptions_count: int = 0
    git_commit_sha: str = ""
    dirty_working_tree: bool = False
    trading_mode: str = "shadow"
    data_source_type: str = "synthetic"
    clock_mode: str = "SystemClock"
    queue_mode: str = "asynchronous"
    soak_mode: str = "ACCELERATED_SIMULATION"
    provenance: Dict[str, Any] = field(default_factory=dict)
    latency_summary: Dict[str, Any] = field(default_factory=dict)

    def validate_invariants(self) -> List[str]:
        """Validates report self-consistency invariants."""
        violations = []
        if self.orders_filled > self.orders_submitted:
            violations.append(f"orders_filled ({self.orders_filled}) > orders_submitted ({self.orders_submitted})")
        if self.ticks_processed + self.ticks_rejected > self.market_data_received + 10:
            violations.append(
                f"processed+rejected ({self.ticks_processed + self.ticks_rejected}) > received ({self.market_data_received})"
            )
        if self.total_fills < self.orders_filled:
            violations.append(f"total_fills ({self.total_fills}) < orders_filled ({self.orders_filled})")
        return violations


class ShadowSessionReporter:
    """
    Collects metrics across session components and writes canonical JSON + Markdown reports.
    """

    def __init__(self, reports_dir: str = "data/reports"):
        self.reports_dir = reports_dir
        self._lock = threading.RLock()
        self._latest_report: Optional[ShadowSessionData] = None

    def build_report(
        self,
        session_id: str,
        symbols: List[str],
        start_time: datetime,
        end_time: datetime,
        market_session_start: Optional[datetime] = None,
        market_session_end: Optional[datetime] = None,
        process_start: Optional[datetime] = None,
        process_end: Optional[datetime] = None,
        first_tick_time: Optional[datetime] = None,
        last_tick_time: Optional[datetime] = None,
        ticks_received: int = 0,
        ticks_valid: int = 0,
        ticks_rejected: int = 0,
        bars_count: int = 0,
        signals_count: int = 0,
        risk_approvals: int = 0,
        risk_rejections: int = 0,
        orders_submitted: int = 0,
        orders_filled: int = 0,
        orders_rejected: int = 0,
        orders_cancelled: int = 0,
        total_fills: int = 0,
        gross_pnl: float = 0.0,
        commission: float = 0.0,
        tax: float = 0.0,
        slippage: float = 0.0,
        max_drawdown_pct: float = 0.0,
        queue_max_depth: int = 0,
        queue_overflow_count: int = 0,
        disconnect_count: int = 0,
        reconciliation_count: int = 0,
        kill_switch_events: int = 0,
        data_health_incidents: int = 0,
        exceptions_count: int = 0,
        trading_mode: str = "shadow",
        data_source_type: str = "synthetic",
        clock_mode: str = "SystemClock",
        queue_mode: str = "asynchronous",
        soak_mode: str = "ACCELERATED_SIMULATION",
        provenance: Optional[Dict[str, Any]] = None,
        latency_snapshot: Optional[LatencySnapshot] = None,
    ) -> ShadowSessionData:
        with self._lock:
            commit_sha, is_dirty = get_git_provenance()
            net_pnl = gross_pnl - commission - tax - slippage
            lat_dict = {}
            if latency_snapshot:
                for stage, p in latency_snapshot.stages.items():
                    lat_dict[stage] = {
                        "p50_ms": p.p50_ms,
                        "p95_ms": p.p95_ms,
                        "p99_ms": p.p99_ms,
                        "mean_ms": p.mean_ms,
                        "count": p.count,
                    }

            prov_dict = provenance or {}
            prov_dict.update({
                "git_commit": commit_sha,
                "git_dirty": is_dirty,
                "soak_mode": soak_mode,
                "queue_mode": queue_mode,
                "clock_mode": clock_mode,
                "data_source": data_source_type,
            })

            data = ShadowSessionData(
                session_id=session_id,
                session_date=start_time.strftime("%Y-%m-%d"),
                symbols=symbols,
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                market_session_start=market_session_start.isoformat() if market_session_start else "",
                market_session_end=market_session_end.isoformat() if market_session_end else "",
                process_start=process_start.isoformat() if process_start else "",
                process_end=process_end.isoformat() if process_end else "",
                first_tick_time=first_tick_time.isoformat() if first_tick_time else "",
                last_tick_time=last_tick_time.isoformat() if last_tick_time else "",
                market_data_received=ticks_received,
                ticks_processed=ticks_valid,
                ticks_rejected=ticks_rejected,
                bars_generated=bars_count,
                signals_generated=signals_count,
                risk_approvals=risk_approvals,
                risk_rejections=risk_rejections,
                orders_submitted=orders_submitted,
                orders_filled=orders_filled,
                orders_rejected=orders_rejected,
                orders_cancelled=orders_cancelled,
                total_fills=total_fills,
                gross_pnl=round(gross_pnl, 2),
                total_commission=round(commission, 2),
                total_statutory_tax=round(tax, 2),
                total_slippage=round(slippage, 2),
                net_pnl=round(net_pnl, 2),
                max_drawdown_pct=round(max_drawdown_pct, 4),
                queue_max_depth=queue_max_depth,
                queue_overflow_count=queue_overflow_count,
                disconnect_count=disconnect_count,
                reconciliation_count=reconciliation_count,
                kill_switch_events=kill_switch_events,
                data_health_incidents=data_health_incidents,
                exceptions_count=exceptions_count,
                git_commit_sha=commit_sha,
                dirty_working_tree=is_dirty,
                trading_mode=trading_mode,
                data_source_type=data_source_type,
                clock_mode=clock_mode,
                queue_mode=queue_mode,
                soak_mode=soak_mode,
                provenance=prov_dict,
                latency_summary=lat_dict,
            )
            self._latest_report = data
            return data

    def persist_report(self, report: ShadowSessionData) -> Tuple[str, str]:
        """Saves JSON and Markdown reports to disk."""
        with self._lock:
            os.makedirs(self.reports_dir, exist_ok=True)
            date_tag = report.session_date.replace("-", "")
            json_filename = f"shadow_session_{date_tag}_{report.session_id}.json"
            md_filename = f"shadow_session_{date_tag}_{report.session_id}.md"

            json_path = os.path.join(self.reports_dir, json_filename)
            md_path = os.path.join(self.reports_dir, md_filename)

            # 1. Write JSON
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(asdict(report), f, indent=2, ensure_ascii=False)

            # 2. Render and Write Markdown
            md_content = self.render_markdown(report)
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_content)

            logger.info(f"Persisted shadow session report to {json_path} and {md_path}")
            return json_path, md_path

    def render_markdown(self, report: ShadowSessionData) -> str:
        """Renders clean human-readable Markdown summary with provenance."""
        invariants = report.validate_invariants()
        inv_status = "PASSED" if not invariants else f"FAILED ({', '.join(invariants)})"

        lines = [
            f"# QuantPilot Shadow Trading Session Report",
            f"**Session ID**: `{report.session_id}` | **Date**: `{report.session_date}`",
            f"**Market Window**: `{report.market_session_start or report.start_time}` to `{report.market_session_end or report.end_time}`",
            f"**Symbols**: `{', '.join(report.symbols)}`",
            f"**Execution Mode**: `{report.soak_mode}` | **Trading Target**: `{report.trading_mode.upper()}`",
            f"**Provenance**: Commit `{report.git_commit_sha[:7]}` ({'dirty' if report.dirty_working_tree else 'clean'}) | Invariants: `{inv_status}`",
            "",
            "## 1. Executive PnL & Cost Summary",
            f"- **Gross PnL**: `{report.gross_pnl:+,.2f} TWD`",
            f"- **Statutory Tax (證券交易稅)**: `{report.total_statutory_tax:,.2f} TWD`",
            f"- **Brokerage Commission**: `{report.total_commission:,.2f} TWD`",
            f"- **Estimated Slippage**: `{report.total_slippage:,.2f} TWD`",
            f"- **Net Virtual PnL**: `{report.net_pnl:+,.2f} TWD`",
            f"- **Max Drawdown**: `{report.max_drawdown_pct*100:.2f}%`",
            "",
            "## 2. Market Data & Pipeline Ingestion",
            f"- **Ticks Received**: `{report.market_data_received:,}`",
            f"- **Ticks Processed**: `{report.ticks_processed:,}`",
            f"- **Ticks Rejected (Integrity Filter)**: `{report.ticks_rejected:,}`",
            f"- **Bars Generated (1-min Total Finalized)**: `{report.bars_generated:,}`",
            f"- **Queue Max Depth**: `{report.queue_max_depth}` (Overflows: `{report.queue_overflow_count}`)",
            f"- **Disconnects**: `{report.disconnect_count}` | **Data Incidents**: `{report.data_health_incidents}`",
            "",
            "## 3. Trading & Risk Decisions",
            f"- **Signals Generated**: `{report.signals_generated:,}`",
            f"- **Risk Engine Approvals**: `{report.risk_approvals:,}`",
            f"- **Risk Engine Rejections**: `{report.risk_rejections:,}`",
            f"- **Orders Submitted**: `{report.orders_submitted:,}`",
            f"- **Orders Filled**: `{report.orders_filled:,}` (Total Fills: `{report.total_fills:,}`)",
            f"- **Orders Rejected**: `{report.orders_rejected:,}`",
            f"- **Orders Cancelled**: `{report.orders_cancelled:,}`",
            f"- **Kill Switch Halts**: `{report.kill_switch_events}`",
            f"- **Broker Reconciliations**: `{report.reconciliation_count}`",
            "",
            "## 4. Latency Distribution (ms)",
            "| Stage | p50 (ms) | p95 (ms) | p99 (ms) | Mean (ms) | Count |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
        if report.latency_summary:
            for stage, m in report.latency_summary.items():
                lines.append(
                    f"| {stage} | {m.get('p50_ms', 0):.3f} | {m.get('p95_ms', 0):.3f} | {m.get('p99_ms', 0):.3f} | {m.get('mean_ms', 0):.3f} | {m.get('count', 0)} |"
                )
        else:
            lines.append("| No latency samples recorded | - | - | - | - | 0 |")

        lines.append("")
        return "\n".join(lines)

    def get_latest_report(self) -> Optional[ShadowSessionData]:
        with self._lock:
            return self._latest_report
