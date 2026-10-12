import logging

logger = logging.getLogger(__name__)

# Allow-list of approved, parameterized governed-metric queries.
# The metric's stored calculation_template (in Neo4j) is documentation; execution
# uses ONLY these vetted queries — never a raw model- or graph-supplied string.
# Each query takes exactly one bound parameter: line_of_business.
GOVERNED_METRIC_SQL = {
    "solvency_ii_ratio": (
        "SELECT eligible_own_funds / scr AS ratio "
        "FROM quarterly_capital WHERE line_of_business = %s"
    ),
    "cargo_loss_ratio": (
        "SELECT incurred_losses / earned_premium AS ratio "
        "FROM cargo_experience WHERE line_of_business = %s"
    ),
}


def compute_governed_metric(pg, metric_name: str, line_of_business: str) -> dict:
    """Execute an approved governed metric against Aurora and return its value.

    Governed: only registered metrics run (allow-list), the parameter is bound
    (never interpolated), and the transaction is read-only. Returns
    ``{"metric", "line_of_business", "value"}`` (``value`` is ``None`` if the
    line has no row).
    """
    sql = GOVERNED_METRIC_SQL.get(metric_name)
    if sql is None:
        raise ValueError(f"Unknown/unapproved governed metric: {metric_name}")

    with pg.conn.cursor() as cur:
        cur.execute("SET TRANSACTION READ ONLY;")   # governed: read-only
        cur.execute(sql, (line_of_business,))        # governed: bound param
        row = cur.fetchone()

    value = float(row[0]) if row and row[0] is not None else None
    logger.info("compute_governed_metric %s(%s) -> %s", metric_name, line_of_business, value)
    return {"metric": metric_name, "line_of_business": line_of_business, "value": value}
