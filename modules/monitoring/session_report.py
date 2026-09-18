"""
Structured Shadow Session Reporting
Gathers comprehensive end-of-session telemetry across execution, risk, data health,
latency percentiles, order statistics, and PnL into JSON and human-readable Markdown summaries.
"""
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

from modules.monitoring.latency import LatencySnapshot

logger = logging.getLogger("QuantPilot.SessionReport")


@dataclass
class ShadowSessionData:
    session_id: str
    session_date: str
    symbols: List[str]
    start_time: str
    end_time: str
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
    latency_summary: Dict[str, Any] = field(default_factory=dict)


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
        latency_snapshot: Optional[LatencySnapshot] = None,
    ) -> ShadowSessionData:
        with self._lock:
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

            data = ShadowSessionData(
                session_id=session_id,
                session_date=start_time.strftime("%Y-%m-%d"),
                symbols=symbols,
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
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
        """Renders clean human-readable Markdown summary."""
        lines = [
            f"# QuantPilot Shadow Trading Session Report",
            f"**Session ID**: `{report.session_id}` | **Date**: `{report.session_date}`",
            f"**Window**: `{report.start_time}` to `{report.end_time}`",
            f"**Symbols**: `{', '.join(report.symbols)}`",
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
            f"- **Bars Generated (1-min)**: `{report.bars_generated:,}`",
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
