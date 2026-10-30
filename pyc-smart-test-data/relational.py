"""Relational demo generator.
Learns empirical distributions from the four source tables (not a multi-table TVAE).
No external real data should be used in public deployment.
"""
import numpy as np
import pandas as pd

def generate_children(real, customers, seed=42):
    rng=np.random.default_rng(int(seed))
    base_clients=real["clientes"].copy()
    source_sales=real["ventas"].copy()
    source_invoices=real["facturas"].copy()
    source_payments=real["pagos"].copy()
    sales_counts=source_sales.groupby("cliente_id").size().reindex(base_clients["cliente_id"],fill_value=0).to_numpy()
    sales_counts=sales_counts.astype(int)
    invoices_per_sale=source_invoices.groupby("venta_id").size().reindex(source_sales["venta_id"],fill_value=0).to_numpy()
    paid_amounts=source_payments.groupby("factura_id")["monto_pago"].sum()
    original_invoice=source_invoices.set_index("factura_id")
    fractions=[]
    for fid,group in source_payments.groupby("factura_id"):
        if fid in original_invoice.index and float(original_invoice.loc[fid,"monto_total"])>0:
            fractions.append(min(1,max(0,float(group["monto_pago"].sum())/float(original_invoice.loc[fid,"monto_total"]))))
    fractions=np.asarray(fractions,dtype=float)
    pay_count_dist=source_payments.groupby("factura_id").size().reindex(source_invoices["factura_id"],fill_value=0).astype(int).to_numpy()
    sales=[]; invoices=[]; payments=[]
    for cid in customers["cliente_id"]:
        n=int(rng.choice(sales_counts)) if len(sales_counts) else 0
        for _ in range(n):
            template=source_sales.iloc[int(rng.integers(len(source_sales)))]
            vid=len(sales)+1
            row=template.to_dict()
            row.update(venta_id=vid,cliente_id=int(cid))
            if "monto_venta" in row:
                row["monto_venta"]=round(max(0,float(row["monto_venta"])),2)
            sales.append(row)
            if len(invoices_per_sale) and int(rng.choice(invoices_per_sale))==0:
                continue
            fid=len(invoices)+1
            amount=round(float(row["monto_venta"])*1.18,2)
            count=int(rng.choice(pay_count_dist)) if len(pay_count_dist) else 0
            count=max(0,min(count,5))
            paid=0.0
            if count:
                fraction=float(rng.choice(fractions)) if len(fractions) else 1.
                paid=round(amount*fraction,2)
                slices=[round(paid/count,2)]*(count-1)
                slices.append(round(paid-sum(slices),2))
                for pamt in slices:
                    payments.append({"pago_id":len(payments)+1,"factura_id":fid,"cliente_id":int(cid),"monto_pago":max(0,pamt)})
            state="Pagada" if abs(paid-amount)<.02 else ("Parcial" if paid>0 else "Pendiente")
            invoices.append({"factura_id":fid,"venta_id":vid,"cliente_id":int(cid),"monto_total":amount,"estado":state})
    def frame(rows, source):
        if not rows:
            return pd.DataFrame(columns=source.columns)
        result=pd.DataFrame(rows)
        for col in source.columns:
            if col not in result:result[col]=pd.NA
        return result[source.columns]
    return {"clientes":customers.copy(),"ventas":frame(sales,source_sales),"facturas":frame(invoices,source_invoices),"pagos":frame(payments,source_payments)}

def verify_relations(t):
    cl=t["clientes"];v=t["ventas"];f=t["facturas"];p=t["pagos"]
    errors=[]
    for name,df,key in [("clientes",cl,"cliente_id"),("ventas",v,"venta_id"),("facturas",f,"factura_id"),("pagos",p,"pago_id")]:
        if df[key].duplicated().any() or df[key].isna().any():errors.append(name+": PK duplicada o nula")
    if not v["cliente_id"].isin(cl["cliente_id"]).all():errors.append("Venta sin cliente")
    if not f["venta_id"].isin(v["venta_id"]).all():errors.append("Factura sin venta")
    if not f["cliente_id"].isin(cl["cliente_id"]).all():errors.append("Factura sin cliente")
    if not p["factura_id"].isin(f["factura_id"]).all():errors.append("Pago sin factura")
    if not p["cliente_id"].isin(cl["cliente_id"]).all():errors.append("Pago sin cliente")
    if not f.empty and not v.empty:
        parent=v.set_index("venta_id")["cliente_id"]
        if not (f["cliente_id"].to_numpy()==f["venta_id"].map(parent).to_numpy()).all():errors.append("Factura con cliente distinto del de la venta")
    if not p.empty and not f.empty:
        parent=f.set_index("factura_id")["cliente_id"]
        if not (p["cliente_id"].to_numpy()==p["factura_id"].map(parent).to_numpy()).all():errors.append("Pago con cliente distinto del de la factura")
        totals=p.groupby("factura_id")["monto_pago"].sum()
        for row in f.itertuples(index=False):
            paid=float(totals.get(row.factura_id,0));amount=float(row.monto_total)
            status="Pagada" if abs(paid-amount)<.02 else ("Parcial" if paid>0 else "Pendiente")
            if paid>amount+.02 or paid<0 or status!=row.estado:
                errors.append("Monto/estado de factura inconsistente");break
    return errors
