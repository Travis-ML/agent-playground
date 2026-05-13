You generate higher-level insights ("reflections") from a related group
of atomic memory events.

Read the events. If there is a clear, well-supported higher-level
insight (a generalization, a pattern, a preference, a recurring theme),
write it as a single sentence ≤ 30 words. If nothing rises above the
particulars, return null.

Output JSON only:
  {"insight": "<sentence>" | null,
   "importance": 0.0..1.0,
   "supporting_event_ids": ["ep_...", ...]}

Events:
{{events}}
