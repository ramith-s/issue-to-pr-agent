"""Core invoice subtotal, discount, and tax calculations."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP

TWO_PLACES = Decimal("0.01")


@dataclass(frozen=True)
class LineItem:
    sku: str
    unit_price: Decimal
    quantity: int


@dataclass(frozen=True)
class InvoiceSummary:
    subtotal: Decimal
    discounted_subtotal: Decimal
    tax: Decimal
    total: Decimal


def _money(value: Decimal) -> Decimal:
    return value.quantize(TWO_PLACES, rounding=ROUND_HALF_UP)


def calculate_invoice_total(
    items: list[LineItem],
    tax_rate: Decimal,
    flat_discount: Decimal = Decimal("0.00"),
) -> InvoiceSummary:
    """Compute invoice totals with flat promotional discounts applied before tax."""
    if tax_rate < Decimal("0.00"):
        raise ValueError("tax_rate cannot be negative")
    if flat_discount < Decimal("0.00"):
        raise ValueError("flat_discount cannot be negative")

    subtotal = _money(
        sum((item.unit_price * item.quantity for item in items), Decimal("0.00"))
    )
    discounted_subtotal = subtotal - flat_discount
    tax = _money(discounted_subtotal * tax_rate)
    total = _money(discounted_subtotal + tax)
    return InvoiceSummary(
        subtotal=subtotal,
        discounted_subtotal=_money(discounted_subtotal),
        tax=tax,
        total=total,
    )
