"""Network and optional PrimeKG helpers for the drug-information tool.

The agent owns FDA field selection. Importing this module never creates a model
client, reads credentials, or downloads a knowledge graph.
"""
from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any
# Re-exported for the drug tool's FDA URL construction.
from urllib.parse import quote as quote, urlencode as urlencode

import requests


_primekg = None


def _request_with_retry(method, url, max_retries=1, delay=1, **kwargs):
    """Make up to max_retries attempts with normal TLS verification enabled."""
    for attempt in range(max(1, max_retries)):
        try:
            response = requests.request(method, url, **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt + 1 < max_retries:
                time.sleep(delay)
    return None


def _get_primekg():
    """Load the optional PyTDC resource only when PrimeKG is requested."""
    global _primekg
    if _primekg is None:
        try:
            from tdc.resource import PrimeKG

            default_path = Path(__file__).resolve().parents[1] / '.runs/cache/primekg'
            cache_path = Path(os.environ.get('PRIMEKG_CACHE_DIR', default_path))
            _primekg = PrimeKG(path=str(cache_path))
        except (ImportError, OSError, RuntimeError):
            return None, None
    return _primekg, _primekg.df


def _query_primekg_drug_info(drug_name: str) -> dict[str, Any]:
    """Return direct drug relationships from an explicitly enabled PrimeKG."""
    _, frame = _get_primekg()
    if frame is None:
        return {}
    drug_lower = drug_name.lower()
    matches = frame[
        (frame['x_name'].str.lower() == drug_lower)
        | (frame['y_name'].str.lower() == drug_lower)
    ]
    if matches.empty:
        return {}

    info = {'targets': [], 'indications': [], 'interactions': [], 'source': 'PrimeKG'}
    for _, row in matches.iterrows():
        x_type, y_type = row['x_type'], row['y_type']
        x_name, y_name = row['x_name'], row['y_name']
        if any(kind in x_type or kind in y_type for kind in ('gene', 'protein')):
            target = x_name if x_name.lower() != drug_lower else y_name
            if target not in info['targets']:
                info['targets'].append(target)
        if 'disease' in x_type or 'disease' in y_type:
            disease = x_name if 'disease' in x_type else y_name
            if disease not in info['indications']:
                info['indications'].append(disease)
        if x_type == y_type == 'drug':
            other = x_name if x_name.lower() != drug_lower else y_name
            if other not in info['interactions']:
                info['interactions'].append(other)

    info['targets'] = info['targets'][:20]
    info['indications'] = info['indications'][:20]
    info['interactions'] = info['interactions'][:10]
    return info
