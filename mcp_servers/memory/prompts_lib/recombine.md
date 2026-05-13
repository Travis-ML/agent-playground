You are dreaming. You'll be shown three memories from different parts
of a knowledge graph. Most triplets have nothing connecting them. Look
for a non-obvious, plausibly-true connection between them that would be
worth the agent investigating later.

If a connection is plausible: write a single statement ≤ 30 words.
If nothing rises above coincidence: output exactly the word `none`.

Output JSON only:
  {"statement": "<sentence>" | null, "confidence": 0.0..1.0}

Memories:
A) {{a}}
B) {{b}}
C) {{c}}
