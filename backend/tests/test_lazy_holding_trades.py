"""Provider trades are only fetched where the Performance tab is used.

Fetching trades costs one provider request per holding, so a bank sync
skips them for workspaces that never opened Performance, and otherwise only
asks for holdings whose position moved or that have no trades stored yet.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Optional

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.asset import Asset
from app.models.asset_value import AssetValue
from app.models.bank_connection import BankConnection
from app.models.user import User
from app.providers import register_provider
from app.providers.base import (
    AccountData,
    BankProvider,
    ConnectionData,
    HoldingData,
    HoldingTradeData,
    TransactionData,
)
from app.services import portfolio_performance_service
from app.services.connection_service import _position_changed, _sync_holdings


class _TradesProvider(BankProvider):
    holdings: list[HoldingData] = []
    trade_requests: list[list[str]] = []

    @property
    def name(self) -> str:
        return "trades-mock"

    async def get_oauth_url(self, redirect_uri, state, flow_params=None):  # pragma: no cover
        return "http://mock"

    async def handle_oauth_callback(self, code) -> ConnectionData:  # pragma: no cover
        raise NotImplementedError

    async def get_accounts(self, credentials) -> list[AccountData]:  # pragma: no cover
        return []

    async def get_transactions(
        self, credentials, account_external_id, since=None, payee_source="auto"
    ) -> list[TransactionData]:  # pragma: no cover
        return []

    async def refresh_credentials(self, credentials):  # pragma: no cover
        return credentials

    async def get_holdings(self, credentials) -> list[HoldingData]:
        return list(_TradesProvider.holdings)

    async def get_holding_trades(self, credentials, holdings, *, full_history=False):
        _TradesProvider.trade_requests.append(sorted(h.external_id for h in holdings))
        # One purchase per holding that adds up to its reported position.
        return [
            HoldingTradeData(
                external_id=f"buy-{holding.external_id}-{holding.quantity}",
                holding_external_id=holding.external_id,
                kind="buy",
                date=date(2026, 1, 2),
                quantity=holding.quantity or Decimal("1"),
                price=Decimal("10"),
            )
            for holding in holdings
        ]


@pytest.fixture(autouse=True)
def _register_provider():
    register_provider("trades-mock", _TradesProvider)
    _TradesProvider.holdings = []
    _TradesProvider.trade_requests = []


@pytest_asyncio.fixture
async def connection(session: AsyncSession, test_user: User, test_workspace) -> BankConnection:
    conn = BankConnection(
        id=uuid.uuid4(),
        user_id=test_user.id,
        workspace_id=test_workspace.id,
        provider="trades-mock",
        external_id="item-trades",
        institution_name="Mock Broker",
        credentials={"item_id": "item-trades"},
        status="active",
        created_at=datetime.now(timezone.utc),
    )
    session.add(conn)
    await session.commit()
    return conn


def _fund(external_id: str, quantity: str, value: str) -> HoldingData:
    return HoldingData(
        external_id=external_id,
        name=f"Fund {external_id}",
        currency="BRL",
        current_value=Decimal(value),
        quantity=Decimal(quantity),
        metadata={"status": "ACTIVE", "type": "MUTUAL_FUND"},
    )


async def _sync(session: AsyncSession, user: User, connection: BankConnection) -> None:
    await _sync_holdings(session, user.id, connection, {"item_id": "item-trades"})
    await session.commit()


async def _age_valuations(session: AsyncSession, workspace_id: uuid.UUID) -> None:
    """Move today's valuations to yesterday, as if the next sync is a day later."""
    asset_ids = select(Asset.id).where(Asset.workspace_id == workspace_id)
    await session.execute(
        update(AssetValue)
        .where(AssetValue.asset_id.in_(asset_ids))
        .values(date=AssetValue.date - timedelta(days=1))
    )
    await session.commit()


@pytest.mark.asyncio
async def test_no_trade_requests_where_performance_is_unused(
    session: AsyncSession, test_user: User, connection: BankConnection
):
    _TradesProvider.holdings = [_fund("a", "10", "100"), _fund("b", "5", "50")]
    await _sync(session, test_user, connection)
    _TradesProvider.holdings = [_fund("a", "12", "130"), _fund("b", "5", "55")]
    await _sync(session, test_user, connection)

    assert _TradesProvider.trade_requests == []


@pytest.mark.asyncio
async def test_trades_only_for_new_or_moved_positions_once_performance_is_used(
    session: AsyncSession, test_user: User, test_workspace, connection: BankConnection
):
    await portfolio_performance_service.record_usage(session, test_workspace.id)

    # First sync: nothing is stored yet, so every holding is fetched.
    _TradesProvider.holdings = [_fund("a", "10", "100"), _fund("b", "5", "50")]
    await _sync(session, test_user, connection)
    assert _TradesProvider.trade_requests == [["a", "b"]]

    # A day later only prices moved: no trade requests at all.
    await _age_valuations(session, test_workspace.id)
    _TradesProvider.holdings = [_fund("a", "10", "104"), _fund("b", "5", "52")]
    await _sync(session, test_user, connection)
    assert _TradesProvider.trade_requests == [["a", "b"]]

    # Another day: more shares of "a" were bought, so only "a" is fetched.
    await _age_valuations(session, test_workspace.id)
    _TradesProvider.holdings = [_fund("a", "12", "130"), _fund("b", "5", "53")]
    await _sync(session, test_user, connection)
    assert _TradesProvider.trade_requests == [["a", "b"], ["a"]]


@pytest.mark.parametrize(
    ("quantity", "value", "previous_units", "previous_amount", "expected"),
    [
        ("10", "120", Decimal("10"), Decimal("100"), False),  # price move only
        ("12", "120", Decimal("10"), Decimal("100"), True),  # shares bought
        (None, "100", None, Decimal("100"), False),  # balance unchanged
        (None, "150", None, Decimal("100"), True),  # balance moved, cause unknown
        (None, "100", None, None, True),  # never valued before
        ("10.0000004", "100", Decimal("10"), Decimal("100"), False),  # beyond stored scale
    ],
)
def test_position_changed(
    quantity: Optional[str],
    value: str,
    previous_units: Optional[Decimal],
    previous_amount: Optional[Decimal],
    expected: bool,
):
    holding = HoldingData(
        external_id="h",
        name="Holding",
        currency="BRL",
        current_value=Decimal(value),
        quantity=Decimal(quantity) if quantity is not None else None,
    )
    assert _position_changed(holding, previous_units, previous_amount) is expected


@pytest.mark.asyncio
async def test_opening_performance_records_usage(
    client: AsyncClient, auth_headers: dict, session: AsyncSession, test_workspace
):
    assert not await portfolio_performance_service.is_in_use(session, test_workspace.id)
    response = await client.get(
        "/api/assets/performance", params={"period": "3m"}, headers=auth_headers
    )
    assert response.status_code == 200
    assert await portfolio_performance_service.is_in_use(session, test_workspace.id)
    # Recording it again is a no-op.
    await portfolio_performance_service.record_usage(session, test_workspace.id)
    assert await portfolio_performance_service.is_in_use(session, test_workspace.id)
