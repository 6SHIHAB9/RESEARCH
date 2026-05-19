from dataclasses import dataclass, field


@dataclass
class TradeRecord:
    tick: int
    from_agent: str
    from_name: str
    to_agent: str
    to_name: str
    gave: dict      # {item: amount}
    received: dict  # {item: amount}


class Market:
    """Tracks emergent trade history. No prices — value is what agents agree on."""

    def __init__(self):
        self.trades: list[TradeRecord] = []
        self.trade_counts: dict = {}    # agent_id → number of trades
        self.item_demand: dict = {}     # item → how many times requested

    def record_trade(self, tick: int, from_id: str, from_name: str,
                     to_id: str, to_name: str, gave: dict, received: dict):
        record = TradeRecord(tick, from_id, from_name, to_id, to_name, gave, received)
        self.trades.append(record)
        self.trade_counts[from_id] = self.trade_counts.get(from_id, 0) + 1
        self.trade_counts[to_id]   = self.trade_counts.get(to_id, 0) + 1
        if len(self.trades) > 200:
            self.trades = self.trades[-200:]

    def record_demand(self, item: str):
        self.item_demand[item] = self.item_demand.get(item, 0) + 1

    def most_traded(self) -> list[tuple]:
        return sorted(self.item_demand.items(), key=lambda x: -x[1])[:5]

    def to_dict(self) -> dict:
        return {
            "total_trades": len(self.trades),
            "most_demanded": self.most_traded(),
            "top_traders": sorted(self.trade_counts.items(), key=lambda x: -x[1])[:5],
            "recent_trades": [
                {
                    "tick": t.tick,
                    "from": t.from_name,
                    "to": t.to_name,
                    "gave": t.gave,
                    "received": t.received,
                }
                for t in self.trades[-10:]
            ],
        }
