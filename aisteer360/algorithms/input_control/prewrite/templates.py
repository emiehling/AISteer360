"""Default meta-prompts for PRewrite.

Inspired by the meta-prompts in PRewrite (Kong et al., 2024, Appendix B), expressed in our own words. Users override
via `PRewriteArgs.meta_prompt`.
"""
from __future__ import annotations

DEFAULT_PER_QUERY_META_PROMPT = """\
You are improving an instruction for a language model. Given the original instruction and a user query, produce a
revised instruction that, combined with the user query, will elicit a better response from the language model.

Original instruction: {initial_prompt}
User query: {query}

Respond with ONLY the revised instruction, no preamble or commentary.
"""

DEFAULT_STATIC_META_PROMPT = """\
You are improving an instruction for a language model. Given the original instruction, produce a revised instruction
that will elicit better responses for the relevant task.

Original instruction: {initial_prompt}

Respond with ONLY the revised instruction, no preamble or commentary.
"""
