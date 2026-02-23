# 🔴 Red Team Your Bridge

**Natural language → nonlinear FEA → adversarial analysis.**

> *Every bridge gets a peer review. Ours also gets a red team.*

Describe a bridge in plain English. Get a full OpenSees nonlinear model, hundreds of load cases (including the ones nobody thinks to run), and a ranked vulnerability report with real numbers — not opinions.

## Why

Traditional engineering saves the heavy analysis for last. By then, the design is locked, the schedule is set, and finding a problem means starting over.

**Red Team Your Bridge** flips the workflow:

```
Paragraph → FEA model → Find problems NOW → Design around them
```

AI doesn't make engineering easier. It makes rigorous engineering early.

## How It Works

```
"3-span continuous steel plate girder over the Kishwaukee River
 on I-39 in northern Illinois. 315-420-315 ft spans, 7 girders
 at 9.5' spacing. ILM erection."
         │
         ▼
   ┌─────────────┐
   │  NL Parser   │  Extracts: type, location, spans, material, method
   └──────┬──────┘
          ▼
   ┌─────────────┐
   │ Site Recon   │  GPS → seismic, scour, wind, soil, thermal, flood
   └──────┬──────┘
          ▼
   ┌─────────────────────────────────────────┐
   │         MCP Component Tools              │
   │                                          │
   │  foundation · substructure · bearings    │
   │  superstructure · connections · loads    │
   └──────────────┬──────────────────────────┘
                  ▼
   ┌─────────────┐
   │  Assembler   │  Stitches into complete OpenSees model
   └──────┬──────┘
          ▼
   ┌─────────────┐
   │  Red Team    │  Attacks the design from every angle
   └──────┬──────┘
          ▼
   ┌─────────────┐
   │   Report     │  Ranked findings: CRITICAL / WARNING / NOTE
   └─────────────┘
```

## MCP Tools

Each structural component is a self-contained MCP tool. The orchestrating AI agent parses natural language and calls the right tools. Each tool encapsulates domain expertise — the user says "drilled shaft," the tool decides nonlinear p-y curves.

| Tool | Purpose | Engineering Decisions Baked In |
|------|---------|-------------------------------|
| `site-recon` | GPS → full environmental profile | Seismic zone, site class, scour flood, wind speed, thermal range |
| `foundation` | Foundation modeling | **Always** nonlinear soil springs (p-y, t-z, Q-z). Upper/lower bounds automatic. |
| `substructure` | Columns, caps, walls | Mander confinement, P-Δ always on, cracked sections |
| `bearings` | Bearing behavior | Temperature-dependent friction, compression-only, uplift detection |
| `superstructure` | Deck and girder systems | Fiber sections, geometric nonlinearity for spans > 200 ft |
| `connections` | Joints and continuity | Pounding at expansion joints, shear key progressive failure |
| `loads` | Standard + adversarial loads | AASHTO + construction states + failure scenarios + extreme combos |
| `assembler` | Model assembly | Auto-mesh, Rayleigh damping, convergence handling |
| `red-team` | Adversarial analysis engine | DCR scan, failure cascade, sensitivity, robustness, history match |
| `report` | Findings delivery | Executive / technical / raw data tiers |

## Attack Vectors

The red team engine doesn't just check your design. It tries to **break** it:

- **DCR Scanner** — Flags everything above 0.85, not just 1.0. Catches the close calls.
- **Failure Cascade** — When something fails, removes it and re-runs. What breaks next?
- **Construction Vulnerability** — Finds the weakest moment during erection.
- **Sensitivity Sweep** — Which parameter ±20% swings the result most?
- **Extreme Combos** — Scour + seismic. Flood + vessel. The scenarios nobody models.
- **Robustness Check** — Remove one girder, one bearing, one column. Does the bridge survive?
- **History Match** — Compares against documented failures. "Your bridge looks like this one that failed."

## Bridge Types Supported

- Steel plate girders (composite/non-composite)
- Prestressed concrete I-girders (AASHTO, BT, NU)
- Prestressed segmental box girders
- CIP concrete box girders
- Steel trusses
- Concrete slab bridges
- Arches (concrete/steel)
- Cable-stayed bridges

## Tech Stack

- **FEA Engine:** [OpenSeesPy](https://openseespydoc.readthedocs.io/)
- **MCP Protocol:** Tool orchestration via [Model Context Protocol](https://modelcontextprotocol.io/)
- **Site Data APIs:** USGS, NOAA, NRCS, FEMA, ASCE 7
- **Failure Database:** NBI (National Bridge Inventory) + NTSB
- **Visualization:** SVG + PDF (WeasyPrint)
- **Language:** Python

## Project Structure

```
natural-language-builder/
├── README.md
├── ARCHITECTURE.md          # Detailed component mapping
├── LICENSE
├── pyproject.toml
├── src/
│   └── nlb/
│       ├── __init__.py
│       ├── mcp_server.py    # MCP server entry point
│       ├── tools/
│       │   ├── site_recon.py
│       │   ├── foundation.py
│       │   ├── substructure.py
│       │   ├── bearings.py
│       │   ├── superstructure.py
│       │   ├── connections.py
│       │   ├── loads.py
│       │   ├── assembler.py
│       │   ├── red_team.py
│       │   └── report.py
│       ├── opensees/
│       │   ├── materials.py  # Material library (Concrete01, Steel02, etc.)
│       │   ├── sections.py   # Fiber section builders
│       │   ├── elements.py   # Element type mappings
│       │   └── analysis.py   # Analysis sequence runners
│       ├── data/
│       │   ├── failures.json # Bridge failure database
│       │   └── defaults.json # Engineering defaults by bridge type
│       └── utils/
│           ├── geo.py        # Coordinate lookups
│           ├── parsers.py    # NL parsing helpers
│           └── viz.py        # SVG/PDF visualization
├── tests/
│   ├── test_site_recon.py
│   ├── test_foundation.py
│   └── ...
└── examples/
    ├── i39_kishwaukee.txt    # Example: steel plate girder
    ├── simple_prestressed.txt
    └── cable_stayed.txt
```

## Getting Started

```bash
# Clone
git clone https://github.com/mjamiv/natural-language-builder.git
cd natural-language-builder

# Install
pip install -e .

# Run MCP server
python -m nlb.mcp_server
```

## Status

🚧 **Under active development.**

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full component mapping and build plan.

## License

MIT

## Author

Michael Martello ([@MJAMIV](https://x.com/MJAMIV))
