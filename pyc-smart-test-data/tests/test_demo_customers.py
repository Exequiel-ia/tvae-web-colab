"""Regression tests for fictional customer columns in TVAE sample outputs."""
import pandas as pd
from demo_customers import fill_identities


def test_fill_identities_adds_missing_fields_for_tvae_output():
    """TVAE only learns four business fields; generated IDs/contacts must be added."""
    learned_columns = pd.DataFrame({
        "segmento": ["Personas", "Pyme", "Empresa"],
        "region": ["Lima", "Cusco", "Arequipa"],
        "nivel_riesgo": ["Bajo", "Medio", "Alto"],
        "edad": [29, 41, 56],
        "cliente_id": [1, 2, 3],
    })
    synthetic = fill_identities(learned_columns, seed=1100)
    expected = [
        "nombre_cliente", "direccion", "email", "telefono",
        "cliente_id", "segmento", "region", "nivel_riesgo", "edad",
    ]
    assert list(synthetic[expected]) == expected
    assert synthetic["nombre_cliente"].notna().all()
    assert synthetic["direccion"].str.contains("Zona Demo").all()
    assert synthetic["email"].str.endswith("@example.invalid").all()
    assert synthetic["telefono"].str.contains("X").all()


def test_fill_identities_replaces_demo_contacts_without_touching_business_fields():
    base = pd.DataFrame({
        "segmento": ["Personas"],
        "region": ["Lima"],
        "nivel_riesgo": ["Bajo"],
        "edad": [36],
        "cliente_id": [1],
        "nombre_cliente": ["Cliente Demostración 00001"],
        "direccion": ["Dirección antigua"],
        "email": ["old@example.invalid"],
        "telefono": ["000"],
    })
    synthetic = fill_identities(base, seed=1100)
    for col in ["segmento", "region", "nivel_riesgo", "edad", "cliente_id"]:
        assert synthetic.loc[0, col] == base.loc[0, col]
    assert synthetic.loc[0, "nombre_cliente"] != base.loc[0, "nombre_cliente"]
    assert synthetic.loc[0, "email"] != base.loc[0, "email"]
