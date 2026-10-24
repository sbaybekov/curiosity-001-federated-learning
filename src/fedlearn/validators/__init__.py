"""fedlearn.validators — validation & QC under partial visibility.

What a partner must provide (schema) vs. what a server is allowed to see
(aggregates). Every function here returns counts/statistics only — never raw
rows — and logs each check with a loud pass/fail verdict.
"""

from . import overlap, qc, schema  # noqa: F401
