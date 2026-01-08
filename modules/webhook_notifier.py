"""
Webhook Notification Module
Supports Discord and Slack notifications for alerts and trading signals.
"""

import requests
import json
from typing import Dict, Optional, List
from datetime import datetime


class WebhookNotifier:
    """
    Send notifications to Discord and Slack webhooks.
    """
    
    def __init__(self, config_file: str = "data/webhook_config.json"):
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """Load webhook configuration."""
        import os
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "discord_webhook": "",
            "slack_webhook": "",
            "enabled": False,
            "notify_on": ["alert_triggered", "backtest_complete"]
        }
    
    def _save_config(self):
        """Save webhook configuration."""
        import os
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def configure(self, discord_webhook: str = None, slack_webhook: str = None, enabled: bool = None) -> Dict:
        """
        Configure webhook settings.
        """
        if discord_webhook is not None:
            self.config["discord_webhook"] = discord_webhook
        if slack_webhook is not None:
            self.config["slack_webhook"] = slack_webhook
        if enabled is not None:
            self.config["enabled"] = enabled
        
        self._save_config()
        return {"success": True, "config": self.get_config()}
    
    def get_config(self) -> Dict:
        """Get webhook configuration (without exposing full URLs)."""
        return {
            "discord_configured": bool(self.config.get("discord_webhook")),
            "slack_configured": bool(self.config.get("slack_webhook")),
            "enabled": self.config.get("enabled", False),
            "notify_on": self.config.get("notify_on", [])
        }
    
    def send_discord(self, title: str, message: str, color: int = 0x58A6FF, fields: List[Dict] = None) -> bool:
        """
        Send notification to Discord webhook.
        
        Args:
            title: Embed title
            message: Main message content
            color: Embed color (hex)
            fields: Optional list of {name, value, inline} dicts
        """
        webhook_url = self.config.get("discord_webhook")
        if not webhook_url:
            return False
        
        embed = {
            "title": title,
            "description": message,
            "color": color,
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": "QuantPilot Trading System"}
        }
        
        if fields:
            embed["fields"] = fields
        
        payload = {
            "embeds": [embed]
        }
        
        try:
            response = requests.post(webhook_url, json=payload, timeout=10)
            return response.status_code == 204
        except Exception as e:
            print(f"Discord webhook error: {e}")
            return False
    
    def send_slack(self, title: str, message: str, fields: List[Dict] = None) -> bool:
        """
        Send notification to Slack webhook.
        
        Args:
            title: Message title
            message: Main message content
            fields: Optional list of {title, value, short} dicts
        """
        webhook_url = self.config.get("slack_webhook")
        if not webhook_url:
            return False
        
        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": title}
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": message}
            }
        ]
        
        if fields:
            field_elements = []
            for f in fields:
                field_elements.append({
                    "type": "mrkdwn",
                    "text": f"*{f.get('name', f.get('title', ''))}*\n{f.get('value', '')}"
                })
            blocks.append({
                "type": "section",
                "fields": field_elements[:10]  # Slack max 10 fields
            })
        
        blocks.append({
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": f"_QuantPilot • {datetime.now().strftime('%Y-%m-%d %H:%M')}_"}]
        })
        
        payload = {"blocks": blocks}
        
        try:
            response = requests.post(webhook_url, json=payload, timeout=10)
            return response.status_code == 200
        except Exception as e:
            print(f"Slack webhook error: {e}")
            return False
    
    def notify(self, event_type: str, title: str, message: str, fields: List[Dict] = None):
        """
        Send notification to all configured webhooks.
        
        Args:
            event_type: Type of event (e.g., 'alert_triggered', 'backtest_complete')
            title: Notification title
            message: Notification message
            fields: Optional data fields
        """
        if not self.config.get("enabled"):
            return {"success": False, "reason": "Webhooks disabled"}
        
        if event_type not in self.config.get("notify_on", []):
            return {"success": False, "reason": f"Notifications disabled for {event_type}"}
        
        results = {"discord": False, "slack": False}
        
        # Color mapping for different event types
        colors = {
            "alert_triggered": 0xF85149,  # Red
            "backtest_complete": 0x3FB950,  # Green
            "trade_executed": 0x58A6FF,  # Blue
            "risk_warning": 0xF0883E  # Orange
        }
        color = colors.get(event_type, 0x58A6FF)
        
        if self.config.get("discord_webhook"):
            results["discord"] = self.send_discord(title, message, color, fields)
        
        if self.config.get("slack_webhook"):
            results["slack"] = self.send_slack(title, message, fields)
        
        return {"success": any(results.values()), "results": results}
    
    def notify_alert_triggered(self, alert: Dict):
        """Send notification for a triggered alert."""
        fields = [
            {"name": "Symbol", "value": alert.get("symbol", "N/A"), "inline": True},
            {"name": "Condition", "value": alert.get("condition", "N/A"), "inline": True},
            {"name": "Target", "value": str(alert.get("target_value", "N/A")), "inline": True},
            {"name": "Triggered Price", "value": str(alert.get("triggered_price", "N/A")), "inline": True}
        ]
        
        return self.notify(
            "alert_triggered",
            f"🔔 Alert Triggered: {alert.get('symbol', 'Unknown')}",
            f"Your price alert for **{alert.get('symbol')}** has been triggered!",
            fields
        )
    
    def notify_backtest_complete(self, result: Dict):
        """Send notification for completed backtest."""
        return_pct = result.get("return_pct", 0)
        emoji = "📈" if return_pct >= 0 else "📉"
        
        fields = [
            {"name": "Strategy", "value": result.get("strategy", "N/A"), "inline": True},
            {"name": "Return", "value": f"{return_pct:.2f}%", "inline": True},
            {"name": "Sharpe Ratio", "value": f"{result.get('sharpe_ratio', 0):.2f}", "inline": True},
            {"name": "Max Drawdown", "value": f"{result.get('max_drawdown_pct', 0):.2f}%", "inline": True},
            {"name": "Win Rate", "value": f"{result.get('win_rate', 0):.1f}%", "inline": True},
            {"name": "Total Trades", "value": str(result.get("total_trades", 0)), "inline": True}
        ]
        
        return self.notify(
            "backtest_complete",
            f"{emoji} Backtest Complete",
            f"Strategy **{result.get('strategy')}** finished with **{return_pct:.2f}%** return.",
            fields
        )
