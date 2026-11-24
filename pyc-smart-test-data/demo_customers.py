"""Realistic-looking but explicitly fictional demo identities.

No imported customer records, national identifiers or contactable email addresses.
All email domains use RFC 2606 .invalid to prevent delivery.
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd
from faker import Faker


def fill_identities(customers: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Replace demo contact fields with plausible fictitious Spanish names and addresses."""
    frame = customers.copy()
    fake = Faker("es_ES")
    Faker.seed(int(seed))
    rng = np.random.default_rng(int(seed))
    names, addresses, emails, phones = [], [], [], []
    for idx, row in enumerate(frame.itertuples(index=False), start=1):
        segment = str(getattr(row, "segmento", "Personas"))
        if segment == "Empresa":
            name = fake.company()
        elif segment == "Pyme":
            name = fake.company() if rng.random() < .6 else fake.name()
        else:
            name = fake.name()
        location = str(getattr(row, "region", "Lima"))
        street = fake.street_name()
        # "Zona Demo" makes it clear this is not a verified real residential address.
        addr = f"{street} {int(rng.integers(100, 9900))}, Zona Demo, {location}"
        slug = re.sub(r"[^a-z0-9]+", ".", name.lower()).strip(".")[:30] or "cliente"
        email = f"{slug}.{idx:05d}@example.invalid"
        # Intentionally non-dialable placeholders; never produce real phone numbers.
        phone = f"+51 9XX XXX {idx % 1000:03d}"
        names.append(name)
        addresses.append(addr)
        emails.append(email)
        phones.append(phone)
    for col, values in [("nombre_cliente", names), ("direccion", addresses),
                        ("email", emails), ("telefono", phones)]:
        if col in frame.columns:
            frame[col] = values
    return frame
