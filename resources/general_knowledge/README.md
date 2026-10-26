# General medical knowledge sample

This directory contains **16 source-linked records** about clinical pharmacy and medicines management. It contains no original benchmark questions, answer choices, answer labels, rationales, patient cases, or machine-specific paths.

`general_medical_knowledge.csv` has four columns:

| Column | Content |
| --- | --- |
| `entity_name` | Clinical-pharmacy or medicines-management topic |
| `knowledge` | Source-attributed summaries and a labeled verbatim excerpt |
| `source_url` | Public source associated with the record |
| `all_entity_info_json` | The same entity, content, and URL in the runtime's triple format |

The SQLite database contains these records, a BM25 index, and 3,072-dimensional `text-embedding-3-large` vectors generated from these 16 records. The FAISS index and document-ID file cover the same 16 records.

The manifest verifies source-file fingerprints. The query-embedding cache is empty. Runtime rebuilding is disabled; changing the CSV alone will invalidate the manifest.

This is a demonstration corpus with limited coverage, not the paper's experimental knowledge base. Consult the linked sources for current clinical context.
