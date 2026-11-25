"""Existing baseline unit tests for invoice_engine."""

from decimal import Decimal
import pytest

from invoice_engine.billing import LineItem, calculate_invoice_total


def test_standard_invoice_with_tax():
    items = [
        LineItem(sku="WIDGET-A", unit_price=Decimal("19.99"), quantity=2),
        LineItem(sku="WIDGET-B", unit_price=Decimal("10.02"), quantity=1),
    ]
    summary = calculate_invoice_total(items=items, tax_rate=Decimal("0.08"))
    assert summary.subtotal == Decimal("50.00")
    assert summary.discounted_subtotal == Decimal("50.00")
    assert summary.tax == Decimal("4.00")
    assert summary.total == Decimal("54.00")


def test_flat_discount_smaller_than_subtotal():
    items = [LineItem(sku="PLAN-PRO", unit_price=Decimal("100.00"), quantity=1)]
    summary = calculate_invoice_total(
        items=items,
        tax_rate=Decimal("0.10"),
        flat_discount=Decimal("20.00"),
    )
    assert summary.discounted_subtotal == Decimal("80.00")
    assert summary.tax == Decimal("8.00")
    assert summary.total == Decimal("88.00")


def test_negative_discount_raises():
    with pytest.raises(ValueError, match="flat_discount"):
        calculate_invoice_total([], tax_rate=Decimal("0.10"), flat_discount=Decimal("-5.00"))
