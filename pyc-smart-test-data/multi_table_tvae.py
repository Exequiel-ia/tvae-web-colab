"""Four-table TVAE experiment: one synthesizer per entity, plus validated relational assembly.

Not a learned multi-table model. Keys, cardinalities and accounting constraints are
managed explicitly. Demo-only, fictitious data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import TVAESynthesizer


def training_frames(tables):
    sales = tables["ventas"]
    invoices = tables["facturas"]
    payments = tables["pagos"]
    client = tables["clientes"][["segmento", "region", "nivel_riesgo", "edad"]].copy()
    selling = sales[["canal", "monto_venta"]].copy()
    link = invoices.merge(sales[["venta_id", "monto_venta"]], on="venta_id", how="left")
    denom = link["monto_venta"].astype(float).clip(lower=0.01)
    inv = pd.DataFrame({
        "factor_facturacion": (link["monto_total"].astype(float) / denom).clip(0.01, 10).astype(float),
        "estado": link["estado"].astype(str),
    })
    paid = payments.merge(invoices[["factura_id", "monto_total"]], on="factura_id", how="left")
    factor = (paid["monto_pago"].astype(float) / paid["monto_total"].astype(float).clip(lower=0.01)).clip(0, 1)
    payment = pd.DataFrame({
        "fraccion_pago": factor.astype(float),
        "monto_pago": paid["monto_pago"].astype(float).clip(lower=0),
    })
    frames = {"clientes": client, "ventas": selling, "facturas": inv, "pagos": payment}
    for name, frame in frames.items():
        if frame.empty or frame.isna().any().any():
            raise ValueError("No hay suficientes datos completos para entrenar " + name)
    return frames


def train_models(tables, options, seed=42):
    """Train four independent TVAE models, atomically; caller stores only on success."""
    import torch
    frames = training_frames(tables)
    models = {}
    for idx, (name, frame) in enumerate(frames.items()):
        torch.manual_seed(int(seed) + idx)
        np.random.seed(int(seed) + idx)
        meta = SingleTableMetadata()
        meta.detect_from_dataframe(data=frame)
        model = TVAESynthesizer(
            meta,
            epochs=int(options["epochs"]),
            batch_size=int(options["batch"]),
            embedding_dim=int(options["embedding"]),
            compress_dims=tuple(options["dims"]),
            decompress_dims=tuple(options["dims"]),
            l2scale=float(options["l2"]),
            loss_factor=int(options["loss"]),
            enforce_min_max_values=bool(options["minmax"]),
            enforce_rounding=bool(options["rounding"]),
            enable_gpu=False,
        )
        model.fit(frame)
        models[name] = model
    return models


def generate_tables(original, models, customers_count, seed=42):
    """Use trained TVAE outputs for all four entities; explicitly enforce references.

    Cardinalities are sampled from observed counts; invoice states are synthesized;
    payment splits follow trained weights, but accounting totals are imposed.
    """
    from relational import verify_relations
    rng = np.random.default_rng(int(seed))
    source_sales = original["ventas"]
    source_invoices = original["facturas"]
    source_payments = original["pagos"]
    base_customers = original["clientes"]
    cust = models["clientes"].sample(num_rows=int(customers_count)).copy()
    cust["edad"] = pd.to_numeric(cust["edad"], errors="coerce").fillna(40).clip(18, 100).round().astype(int)
    cust["cliente_id"] = np.arange(1, len(cust) + 1)
    from demo_customers import fill_identities
    cust = fill_identities(cust, seed=seed + 1000)[base_customers.columns.tolist()]
    counts = source_sales.groupby("cliente_id").size().reindex(base_customers["cliente_id"], fill_value=0).to_numpy()
    counts = counts.astype(int)
    sizes = rng.choice(counts, size=len(cust)) if len(counts) else np.zeros(len(cust), dtype=int)
    total_sales = int(sizes.sum())
    if not total_sales:
        raise ValueError("No se pudieron preparar ventas para esta generación.")
    generated_sales = models["ventas"].sample(num_rows=total_sales)
    generated_invoices = models["facturas"].sample(num_rows=total_sales)
    per_sale_invoice_count = source_invoices.groupby("venta_id").size().reindex(source_sales["venta_id"], fill_value=0).to_numpy()
    payment_counts = source_payments.groupby("factura_id").size().reindex(source_invoices["factura_id"], fill_value=0).to_numpy()
    # Allocate a reasonable upper bound once, avoiding one SDV sample per payment.
    pool = models["pagos"].sample(num_rows=max(100, total_sales * 6))
    payment_weights = pd.to_numeric(pool["fraccion_pago"], errors="coerce").fillna(0.5).clip(0.01, 1).to_numpy()
    payments = []
    sales = []
    invoices = []
    cursor = 0
    pay_cursor = 0
    for cid, n in zip(cust["cliente_id"], sizes):
        for _ in range(int(n)):
            raw = generated_sales.iloc[cursor]
            item = generated_invoices.iloc[cursor]
            cursor += 1
            vid = len(sales) + 1
            amount = round(max(0.01, float(raw["monto_venta"])), 2)
            sales.append({"venta_id": vid, "cliente_id": int(cid), "canal": str(raw["canal"]), "monto_venta": amount})
            if len(per_sale_invoice_count) and int(rng.choice(per_sale_invoice_count)) <= 0:
                continue
            fid = len(invoices) + 1
            ratio = float(np.clip(item["factor_facturacion"], 0.01, 10))
            total = round(amount * ratio, 2)
            total = max(0.01, total)
            status = str(item["estado"])
            if status not in {"Pendiente", "Parcial", "Pagada"}:
                status = "Pendiente"
            paid = 0.0
            if status != "Pendiente":
                count = max(1, min(5, int(rng.choice(payment_counts)) if len(payment_counts) else 1))
                weights = []
                for _j in range(count):
                    weights.append(float(payment_weights[pay_cursor % len(payment_weights)]))
                    pay_cursor += 1
                fraction = 1.0 if status == "Pagada" else float(rng.uniform(0.1, 0.9))
                target = round(total * fraction, 2)
                if status == "Parcial" and target >= total:
                    target = round(max(0.0, total - 0.01), 2)
                if status == "Parcial" and target <= 0:
                    status = "Pendiente"
                else:
                    subtotal = 0.0
                    for j, weight in enumerate(weights):
                        p = round(target * weight / sum(weights), 2) if j < count - 1 else round(target - subtotal, 2)
                        p = max(0, p)
                        payments.append({"pago_id": len(payments)+1, "factura_id": fid,
                                         "cliente_id": int(cid), "monto_pago": p})
                        subtotal += p
                    paid = subtotal
            state = "Pagada" if abs(paid-total) < 0.02 else ("Parcial" if paid > 0 else "Pendiente")
            invoices.append({"factura_id": fid, "venta_id": vid, "cliente_id": int(cid),
                             "monto_total": total, "estado": state})
    result = {"clientes": cust,
              "ventas": pd.DataFrame(sales, columns=source_sales.columns).astype(source_sales.dtypes.to_dict()),
              "facturas": pd.DataFrame(invoices, columns=source_invoices.columns).astype(source_invoices.dtypes.to_dict()),
              "pagos": pd.DataFrame(payments, columns=source_payments.columns).astype(source_payments.dtypes.to_dict())}
    errors = verify_relations(result)
    if errors:
        raise ValueError("Integridad de entidades: " + "; ".join(errors))
    return result
