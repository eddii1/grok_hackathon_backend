# Route eval

`uv run fleet-eval eval/cases.json` asks Grok for a route on each overview photo and compares it with a
fixed lawnmower sweep from the same start cell:

- **grok_steps / sweep_steps:** moves until the route enters the target cell (`null` = never).

Make cases at the venue: mark the flight area corners on the floor, take one photo from a high spot, place a
volunteer, and write down the cell they stand in. `corners` are pixel positions far-left;far-right;near-right;near-left.

```json
[{"image": "eval/photos/hall-1.jpg", "goal": "person in a red jacket", "start": "C5", "target": "D2",
  "corners": "120,400;1180,410;1270,1650;10,1640"}]
```
