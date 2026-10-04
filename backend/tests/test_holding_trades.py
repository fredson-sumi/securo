"""Provider trade histories mirrored into synced holdings' ledgers."""

from __future__ import annotations

import uuid
from datetime import date
from decimal import Decimal
from typing import Literal, cast

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.asset import Asset
from app.models.asset_transaction import AssetTransaction
from app.models.user import User
from app.providers.base import BankProvider, HoldingData, HoldingTradeData
from app.providers.pluggy import _build_trade_data
from app.services.connection_service import _reconciled_trades, _sync_holding_trades


def _holding(quantity: str, *, withdrawn: bool = False, external_id: str = "h1") -> HoldingData:
    return HoldingData(
        external_id=external_id,
        name="Holding",
        currency="BRL",
        current_value=Decimal("100"),
        quantity=Decimal(quantity),
        is_withdrawn=withdrawn,
    )


def _trade(
    trade_id: str, kind: Literal["buy", "sell"], day: int, quantity: str, price: str = "1"
) -> HoldingTradeData:
    return HoldingTradeData(
        external_id=trade_id,
        holding_external_id="h1",
        kind=kind,
        date=date(2026, 8, day),
        quantity=Decimal(quantity),
        price=Decimal(price),
    )


def test_reconciled_trades_must_add_up_to_the_reported_position():
    trades = [_trade("b", "buy", 2, "228"), _trade("c", "buy", 28, "81")]
    assert _reconciled_trades(_holding("309"), trades) == trades
    # A missing trade leaves the ledger short of the position: not used.
    assert _reconciled_trades(_holding("309"), trades[:1]) is None


def test_reconciled_trades_drop_a_repeated_provider_row():
    """Pluggy listed the same 08-25 redemption twice under two ids."""
    trades = [
        _trade("buy", "buy", 4, "500000", "0.01"),
        _trade("s1", "sell", 25, "196961", "0.01007794"),
        _trade("s2", "sell", 25, "196961", "0.01007794"),
        _trade("s3", "sell", 25, "248947", "0.01007794"),
    ]
    reconciled = _reconciled_trades(_holding("54092"), trades)
    assert reconciled is not None
    assert [t.external_id for t in reconciled] == ["buy", "s1", "s3"]


def test_reconciled_trades_close_a_withdrawn_holding_at_zero():
    trades = [_trade("b", "buy", 1, "10"), _trade("s", "sell", 20, "10")]
    assert _reconciled_trades(_holding("10", withdrawn=True), trades) == trades


def test_pluggy_trade_keeps_cash_amount_exact_and_skips_income():
    trade = _build_trade_data(
        "inv-1",
        {
            "id": "t1",
            "type": "SELL",
            "tradeDate": "2026-08-25T00:00:00.000Z",
            "quantity": 196961,
            "value": 0.01007794,
            "amount": 1984.96,
            "netAmount": 1977.95,
            "expenses": {"brokerageFee": 1.5, "incomeTax": 7.01},
        },
    )
    assert trade is not None
    assert trade.kind == "sell"
    assert trade.quantity * trade.price == pytest.approx(Decimal("1984.96"))
    assert trade.fee == Decimal("1.5")
    assert (
        _build_trade_data(
            "inv-1",
            {"id": "t2", "type": "INTEREST", "date": "2026-09-11", "quantity": 11, "amount": 1.49},
        )
        is None
    )


@pytest.mark.asyncio
async def test_sync_mirrors_reconciled_trades_and_leaves_manual_rows(
    session: AsyncSession, test_user: User, test_workspace
):
    asset = Asset(
        id=uuid.uuid4(),
        user_id=test_user.id,
        workspace_id=test_workspace.id,
        name="AUPO11",
        type="investment",
        currency="BRL",
        valuation_method="manual",
        source="pluggy",
        external_id="h1",
    )
    session.add(asset)
    await session.flush()
    session.add_all(
        [
            AssetTransaction(
                asset_id=asset.id,
                workspace_id=test_workspace.id,
                kind="buy",
                quantity=Decimal("1"),
                price=Decimal("1"),
                date=date(2026, 8, 1),
                source="pluggy",
                external_id="stale",
            ),
            AssetTransaction(
                asset_id=asset.id,
                workspace_id=test_workspace.id,
                kind="buy",
                quantity=Decimal("5"),
                price=Decimal("1"),
                date=date(2026, 8, 1),
                source="manual",
            ),
        ]
    )
    await session.flush()

    class Provider:
        async def get_holding_trades(self, credentials, holdings, *, full_history=False):
            return [_trade("b", "buy", 2, "228", "109.69"), _trade("c", "buy", 28, "81", "110.78")]

    await _sync_holding_trades(
        session,
        cast(BankProvider, Provider()),
        {},
        [_holding("309")],
        {"h1": asset},
        "pluggy",  # type: ignore[arg-type]
    )
    await session.flush()
    rows = (
        await session.execute(
            select(AssetTransaction.source, AssetTransaction.external_id, AssetTransaction.quantity)
            .where(AssetTransaction.asset_id == asset.id)
            .order_by(AssetTransaction.source, AssetTransaction.external_id)
        )
    ).all()
    assert [(source, external_id, float(qty)) for source, external_id, qty in rows] == [
        ("manual", None, 5.0),
        ("pluggy", "b", 228.0),
        ("pluggy", "c", 81.0),
    ]


@pytest.mark.asyncio
async def test_sync_backfills_history_a_recent_days_fetch_leaves_out(
    session: AsyncSession, test_user: User, test_workspace
):
    """A provider whose scheduled sync covers only days since the last sync."""
    asset = Asset(
        id=uuid.uuid4(),
        user_id=test_user.id,
        workspace_id=test_workspace.id,
        name="SPYL",
        type="investment",
        currency="USD",
        valuation_method="manual",
        source="broker",
        external_id="h1",
    )
    session.add(asset)
    await session.flush()
    first = _trade("t1", "buy", 24, "30.8512", "18.96")
    second = _trade("t2", "buy", 28, "39.969", "19.08")

    class Provider:
        def __init__(self):
            self.calls: list[bool] = []

        async def get_holding_trades(self, credentials, holdings, *, full_history=False):
            self.calls.append(full_history)
            return [first, second] if full_history else [second]

    async def stored_ids() -> list[str]:
        rows = await session.execute(
            select(AssetTransaction.external_id)
            .where(AssetTransaction.asset_id == asset.id)
            .order_by(AssetTransaction.external_id)
        )
        return [external_id for (external_id,) in rows]

    provider = Provider()
    await _sync_holding_trades(
        session,
        cast(BankProvider, provider),
        {},
        [_holding("70.8202")],
        {"h1": asset},
        "broker",  # type: ignore[arg-type]
    )
    await session.flush()
    assert provider.calls == [False, True]
    assert await stored_ids() == ["t1", "t2"]

    # Later recent-days syncs combine with the stored ledger: no refetch.
    provider = Provider()
    await _sync_holding_trades(
        session,
        cast(BankProvider, provider),
        {},
        [_holding("70.8202")],
        {"h1": asset},
        "broker",  # type: ignore[arg-type]
    )
    await session.flush()
    assert provider.calls == [False]
    assert await stored_ids() == ["t1", "t2"]


@pytest.mark.asyncio
async def test_sync_backfills_a_holding_with_no_recent_trades(
    session: AsyncSession, test_user: User, test_workspace
):
    """SXLP: bought once, weeks before the recent-days window."""
    asset = Asset(
        id=uuid.uuid4(),
        user_id=test_user.id,
        workspace_id=test_workspace.id,
        name="SXLP",
        type="investment",
        currency="USD",
        valuation_method="manual",
        source="broker",
        external_id="h1",
    )
    session.add(asset)
    await session.flush()
    buy = _trade("t1", "buy", 24, "3.2274", "47.0")

    class Provider:
        async def get_holding_trades(self, credentials, holdings, *, full_history=False):
            return [buy] if full_history else []

    await _sync_holding_trades(
        session,
        cast(BankProvider, Provider()),
        {},
        [_holding("3.2274")],
        {"h1": asset},
        "broker",  # type: ignore[arg-type]
    )
    await session.flush()
    count = await session.scalar(
        select(func.count())
        .select_from(AssetTransaction)
        .where(AssetTransaction.asset_id == asset.id)
    )
    assert count == 1
