You extract atomic episodic events from a single conversation turn.

Return a JSON object with one key "episodes" whose value is a list of 0
to 6 atomic events that are clearly stated or strongly implied by the
turn. Do NOT speculate beyond what the text says.

Each event must include:
- actor:      "user" | "agent" | "tool:<name>"
- predicate:  a normalized lowercase verb phrase using snake_case
              (e.g. "reported_problem", "expressed_preference",
              "diagnosed", "decided", "asked_question", "confirmed")
- subject:    short canonical-cased noun phrase OR null
- object:     short canonical-cased noun phrase OR null
- summary:    one sentence describing what happened (≤ 30 words)
- importance: 0.0 (trivial) to 1.0 (highly significant), float

If the turn contains nothing memorable, return {"episodes": []}.

Output ONLY the JSON object. No prose, no fences.

---
Conversation context (last few turns, oldest first):
{{context}}

---
Turn to extract from (role={{role}}, occurred_at={{occurred_at}}):
{{turn_text}}
