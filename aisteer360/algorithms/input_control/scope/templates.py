"""Default prompt templates for the SCOPE meta-agents.

These templates are inspired by the rubrics in the SCOPE paper's Appendix F but rephrased to avoid licensing
concerns. They are working starting points, not optimized prompts; users should iterate on them via the corresponding
`*_template` fields on `SCOPEArgs`.

All placeholders use `str.format`-style braces.
"""
from __future__ import annotations


DEFAULT_GENERATOR_TEMPLATE = """\
You are improving a language model's behavior by synthesizing a guideline from an observed interaction.

Current guidelines:
{current_rules}

User input: {input_text}

Model response: {response_text}

Based on this interaction, propose ONE concise guideline (a single sentence) that would help the model produce better
responses to similar inputs in the future. Focus on either: (a) correcting an obvious mistake, or (b) reinforcing a
successful pattern. Do not duplicate existing guidelines.

Respond with ONLY the guideline text, no preamble.
"""


DEFAULT_SELECTOR_TEMPLATE = """\
You are choosing the best of several candidate guidelines.

Current guidelines:
{current_rules}

User input: {input_text}

Model response: {response_text}

Candidate guidelines:
{candidates}

Choose the candidate that would be most useful for guiding future responses. Respond with ONLY the integer index
(0-based) of your choice. No preamble.
"""


DEFAULT_CLASSIFIER_TEMPLATE = """\
You are classifying a guideline.

Current guidelines:
{current_rules}

Guideline to classify:
{guideline}

Decide whether this guideline is "strategic" (broadly applicable across many future tasks/sessions) or "tactical"
(useful only in the current session/context). Also assign a confidence score in [0, 1] reflecting how confident you are
in the strategic assignment.

Respond with a single JSON object on one line:
{{"category": "strategic" or "tactical", "confidence": <float in [0, 1]>}}

No preamble.
"""


DEFAULT_OPTIMIZER_CONFLICT_TEMPLATE = """\
You are resolving conflicts among guidelines.

Guidelines:
{rules}

Identify any guidelines that contradict each other and merge them into a single coherent guideline that captures the
intended behavior. Drop any guidelines that are no longer needed.

Respond with the resulting list of guidelines, one per line, no numbering or preamble.
"""


DEFAULT_OPTIMIZER_SUBSUMPTION_TEMPLATE = """\
You are pruning subsumed guidelines.

Guidelines:
{rules}

Drop any guideline that is fully covered by a more general one in the list. Keep the more general guideline.

Respond with the resulting list of guidelines, one per line, no numbering or preamble.
"""


DEFAULT_OPTIMIZER_CONSOLIDATION_TEMPLATE = """\
You are consolidating similar guidelines.

Guidelines:
{rules}

Merge guidelines that overlap in scope into single, comprehensive guidelines. Preserve any unique guidance.

Respond with the resulting list of guidelines, one per line, no numbering or preamble.
"""
