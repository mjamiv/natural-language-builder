# Red Team Your Bridge — Architecture

## Philosophy

> AI doesn't make engineering easier. It makes rigorous engineering early.

Natural language → full nonlinear FEA → adversarial analysis → ranked findings with real numbers.

The program is bridge-agnostic. It works for any bridge someone can describe in English.

## Design Principle: MCP Tools as Components

Each structural component is an MCP tool. The orchestrating agent:
1. Parses natural language into component calls
2. Each tool returns OpenSees commands (nodes, elements, materials, loads)
3. Tools enforce engineering judgment internally — the user doesn't need to know soil spring formulation or element types
4. Assembly tool stitches components into a complete model
5. Analysis tool runs the attack vectors
6. Report tool synthesizes findings

**Why MCP tools?**
- Each tool is independently testable and improvable
- Encapsulates domain expertise (the user says "drilled shaft"; the tool decides nonlinear p-y curves)
- Agent can call them in any combination for any bridge type
- New bridge types = new tool combinations, not new code

---

## Component MCP Tools

### 1. `site-recon` — Site Intelligence from Coordinates

**Always runs first. Non-negotiable.**

**Input:** GPS coordinates (extracted from location description)

**Infers/Fetches:**
| Data | Source | Output |
|------|--------|--------|
| Seismic hazard | USGS Unified Hazard API | PGA, Ss, S1, site coefficients |
| Soil profile | USGS/NRCS Web Soil Survey API | Soil class, estimated SPT N-values, water table depth |
| Scour potential | NBI + FEMA flood maps | HEC-18 scour estimate, flood frequency |
| Wind speed | ASCE 7 wind map (by county) | V_ult, exposure category |
| Thermal range | NOAA climate normals | T_min, T_max, ΔT design range |
| Hydraulic data | USGS streamflow gauges | Q100, velocity, channel geometry |
| Frost depth | NOAA frost line data | Foundation minimum depth |

**Output:** Site profile JSON that all other tools consume.

**Fixed decisions (user doesn't choose):**
- Site class determination per AASHTO
- Seismic design category
- Scour design flood (Q100 for design, Q500 for check)
- Wind exposure category from terrain

---

### 2. `foundation` — Foundation Modeling

**Input:** Foundation type + site-recon output

**Types supported:**

| Type | OpenSees Elements | Key Parameters |
|------|------------------|----------------|
| Drilled shaft | `dispBeamColumn` + nonlinear p-y/t-z/Q-z springs (`PySimple1`, `TzSimple1`, `QzSimple1`) | Diameter, length, rebar, f'c, soil layers |
| Driven pile group | Same as shaft + pile cap rigid links | Pile size, count, spacing, layout, cap dims |
| Spread footing | Winkler springs (`zeroLength` with `ENT` material) | L, W, D, bearing capacity, subgrade modulus |
| Pile bent (trestle) | `dispBeamColumn` extending through soil with p-y springs | Pile size, embedded length, free height |
| Integral abutment piles | `dispBeamColumn` + p-y + abutment backfill springs (`HyperbolicGap`) | HP pile size, backfill properties |

**Fixed decisions:**
- Soil springs are ALWAYS nonlinear (p-y, t-z, Q-z) — no linear approximations
- Spring properties derived from site-recon soil profile using API/SPT correlations
- Upper/lower bound multipliers applied automatically (2x and 0.5x per AASHTO)
- Group effects applied per AASHTO for pile groups (p-multipliers)
- Scour adjusts spring locations (remove springs above scour depth)

**Inferred from type + site:**
- Minimum embedment (frost, scour, bearing)
- Expected capacity range
- Liquefaction susceptibility flag

**User provides:**
- Foundation type (or "recommend" and tool suggests based on soil + loads)
- Dimensions (or tool sizes based on preliminary demand estimate)

---

### 3. `substructure` — Columns, Caps, Walls

**Input:** Substructure type + geometry + foundation output (for base connectivity)

**Types supported:**

| Type | OpenSees Elements | Key Parameters |
|------|------------------|----------------|
| Single column | `dispBeamColumn` with fiber section | Shape (round/rect), diameter/dims, rebar, height, f'c |
| Multi-column bent | Multiple `dispBeamColumn` + cap beam | Column count, spacing, cap dims, column dims |
| Wall pier | `ShellMITC4` or equivalent plate elements | Height, width, thickness, rebar |
| Pile bent cap | Rigid links from cap to pile tops | Cap dimensions, pile connection type |
| Integral abutment | Backwall + wingwalls as beam elements + backfill springs | Seat width, backwall height, skew |
| Stub abutment on piles | Cap beam on pile foundation | Seat dims, bearing locations |

**Fixed decisions:**
- Concrete: `Concrete01` (confined + unconfined) with Mander confinement model
- Rebar: `Steel02` (Giuffré-Menegotto-Pinto) with strain hardening
- P-Δ effects always included (corotational transformation)
- Cracked section properties for capacity-protected elements

**Inferred:**
- Confinement ratio from transverse steel
- Plastic hinge length (Paulay & Priestley)
- Effective stiffness for elastic demand estimate

**User provides:**
- Type, dimensions, reinforcement (or "design for me" mode where tool sizes to demand)

---

### 4. `bearings` — Bearing Modeling

**Input:** Bearing type + locations

**Types supported:**

| Type | OpenSees Element | Behavior |
|------|-----------------|----------|
| Elastomeric (steel-reinforced) | `ElastomericBearingBoucWen` or `flatSliderBearing` | Shear flexibility, compression stiffness, rollover capacity |
| Pot bearing (guided) | `zeroLength` with `ElasticPP` in guided direction | Free in one direction, locked in other |
| Pot bearing (fixed) | `zeroLength` with high stiffness | Fixed in both horizontal directions |
| PTFE sliding | `flatSliderBearing` with friction model | μ = f(velocity, pressure, temperature) |
| Friction pendulum (single) | `SingleFPBearing` | Isolator with re-centering |
| Friction pendulum (triple) | `TripleFrictionPendulum` | Multi-stage isolation |
| Integral (monolithic) | Rigid connection | No relative displacement |
| Rocker/roller (steel) | `zeroLength` with `ENT` (compression only) + free translation | Vintage bearing type, uplift vulnerable |

**Fixed decisions:**
- Temperature-dependent friction (ALWAYS — not constant μ)
- Compression-only behavior for all non-integral bearings (uplift check automatic)
- Bearing stiffness bounds: upper bound (low temp, high friction) and lower bound (high temp, low friction)

**Inferred:**
- Stiffness properties from bearing dimensions + material
- Friction coefficients from PTFE/stainless standards
- Displacement capacity from bearing geometry

**User provides:**
- Type and location (or "recommend" based on bridge type + seismic demands)

---

### 5. `superstructure` — Deck and Girder Systems

**Input:** Superstructure type + span arrangement + cross-section

**Types supported:**

| Type | OpenSees Model Strategy | Key Parameters |
|------|------------------------|----------------|
| Steel plate girders (composite) | `dispBeamColumn` with fiber section (steel I + concrete slab) | Girder depth, flanges, web, slab thickness, haunch |
| Steel plate girders (non-composite) | `dispBeamColumn` fiber section (steel only) + mass for deck | Same minus composite action |
| Prestressed concrete I-girders | `dispBeamColumn` with fiber section (concrete + strands) | Girder type (AASHTO/BT/NU), strand pattern, f'ci, f'c |
| Prestressed concrete box (segmental) | `dispBeamColumn` with box fiber section | Cell geometry, tendon profile, segment joints |
| CIP concrete box girder | `dispBeamColumn` with multi-cell box section | Cell count, dims, PT profile |
| Steel truss | `truss` or `corotTruss` elements | Member sizes, panel geometry, connection type |
| Concrete slab bridge | `ShellMITC4` plate elements | Slab thickness, reinforcement |
| Arch (concrete/steel) | `dispBeamColumn` following arch geometry | Rise, shape, rib section |
| Cable-stayed | `corotTruss` for cables + beam for deck/tower | Cable layout, tensions, tower geometry |

**Fixed decisions:**
- Composite sections modeled with fiber sections (actual strain compatibility, not transformed section)
- Concrete: tension stiffening included for service checks
- Steel: residual stress pattern included for stability checks
- Prestress: time-dependent losses (creep, shrinkage, relaxation) per AASHTO
- Geometric nonlinearity for all cases > 200 ft span

**Inferred:**
- Live load distribution factors from girder spacing + span (AASHTO 4.6.2.2)
- Impact factor from span length
- Section classification (compact/noncompact/slender)

**User provides:**
- Type, span arrangement, cross-section description (or "typical for this span range")

---

### 6. `connections` — Joints and Continuity

**Input:** Connection type at each support

**Types supported:**
- Expansion joint (gap element with pounding)
- Link slab (cracked concrete tension element)
- Integral connection (rigid)
- Construction joint (friction + dowel)
- Shear key (breakaway element for seismic)
- Splice (bolted or welded — affects fatigue category)

**Fixed decisions:**
- Expansion joints always model pounding (gap + Hertz contact stiffness)
- Shear keys model progressive failure (elastic → yield → rupture)
- All splices flag fatigue category automatically

---

### 7. `loads` — Standard + Adversarial Load Generation

**Input:** site-recon output + bridge geometry

**Standard loads (AASHTO):**
- DC (components, wearing surface, barriers, utilities)
- DW (future wearing surface)
- HL-93 (truck, tandem, lane — moving load analysis)
- Permit vehicles (state-specific based on location)
- PS (prestress)
- CR, SH (creep, shrinkage — time-dependent)
- TU, TG (uniform temp, thermal gradient)
- WS, WL (wind on structure, wind on live load)
- BR (braking)
- CE (centrifugal — if curved)
- EQ (seismic — response spectrum or time-history from site-recon)
- SC (scour — modify foundation springs)
- CV (vessel collision — if over navigable water)
- IC (ice — if in cold climate)
- CT (vehicular collision — if piers near traffic)

**Adversarial loads (the red team):**
- Construction state loads (partially erected, temporary supports, equipment)
- Extreme event combos (scour + seismic, flood + vessel, fire + dead)
- Component failure scenarios (lost bearing, severed tendon, buckled brace)
- Degradation scenarios (section loss, reinforcement corrosion, concrete deterioration)
- "What if" combinations the code doesn't explicitly require

**Fixed decisions:**
- Load factors per AASHTO Table 3.4.1-1 for all limit states
- Moving load analysis uses influence lines (not static placement)
- Seismic: response spectrum minimum, time-history if SDC C or D
- ALL load combinations generated — Strength I through Extreme Event II

---

### 8. `assembler` — Model Assembly

**Input:** Outputs from all component tools

**Does:**
- Assigns global node numbering
- Connects components (foundation tops → column bases → cap → bearings → superstructure)
- Resolves coordinate systems and skew
- Applies constraints (rigid diaphragms, multi-point constraints)
- Generates recorder commands for all critical responses
- Creates analysis sequence (gravity → prestress → live load → pushover → time-history)
- Writes complete OpenSees .tcl or Python script

**Fixed decisions:**
- Mesh density: auto-refine at connections, plastic hinge zones, and midspan
- Mass: lumped at nodes (translational + rotational for seismic)
- Damping: Rayleigh (2% at T1 and 0.2T1) unless user specifies
- Convergence: Newton-Raphson with adaptive step, fallback to BFGS
- Output: envelope forces, displacements, reactions at every element for every load case

---

### 9. `red-team` — Adversarial Analysis Engine

**Input:** Assembled model + analysis results

**Attack vectors:**

| Vector | What It Does |
|--------|-------------|
| **DCR Scanner** | Reports every element/load case with DCR > 0.85 (not just > 1.0 — flags the close calls) |
| **Failure Cascade** | For each DCR > 1.0, removes that element and re-runs. What fails next? |
| **Construction Vulnerability** | Analyzes each construction stage. Finds the weakest state. |
| **Sensitivity Sweep** | Varies key parameters ±20% (soil stiffness, f'c, bearing friction, scour depth). Which parameter swings the result most? |
| **Extreme Combo** | Runs the adversarial load combinations. Reports any that govern over standard AASHTO. |
| **History Match** | Compares bridge characteristics against failure database. Flags similarities. |
| **Robustness Check** | Alternate load path analysis — remove one girder, one bearing, one column. Does the bridge survive? |

---

### 10. `report` — Red Team Report Generator

**Input:** All analysis results + red-team findings

**Output tiers:**
- **Executive summary** — 1-page: bridge description, top 3 findings, overall risk rating (GREEN/YELLOW/RED)
- **Technical report** — Full findings with deformed shapes, DCR plots, sensitivity tornados, failure chain diagrams
- **Raw data** — JSON export of every result for further processing

---

## Model Inference Strategy

The goal is **minimum user input, maximum model fidelity.**

| Information | How We Get It |
|------------|---------------|
| Location/environment | GPS → `site-recon` APIs |
| Bridge type | NL parsing ("steel plate girder" → superstructure tool selection) |
| Geometry | NL parsing ("3 spans, 315-420-315") |
| Foundations | Site-recon soil + bridge demands → tool recommends type + size |
| Bearings | Bridge type + seismic demands → tool recommends |
| Reinforcement/sections | Preliminary demand → tool sizes (or user overrides) |
| Load cases | Automatic from site + bridge type + AASHTO |
| Construction sequence | Inferred from erection method (or user describes) |

**The agent's job:** Convert "3-span steel bridge over a river in Illinois" into 10 MCP tool calls that produce a complete nonlinear FEA model with 500+ load cases and a ranked vulnerability report.

---

## Tech Stack

- **MCP Server:** Node.js or Python (mcporter compatible)
- **FEA Engine:** OpenSeesPy (already installed)
- **APIs:** USGS, NOAA, NRCS, FEMA, ASCE 7 data
- **Failure DB:** NBI (National Bridge Inventory) + NTSB reports
- **Output:** SVG visualizations + PDF report (WeasyPrint)

---

## Build Order

1. `site-recon` — proves the concept of auto-enrichment from coordinates
2. `foundation` — most complex component, highest engineering value
3. `superstructure` — steel plate girder first (most common)
4. `substructure` — columns and caps
5. `bearings` — boundary conditions
6. `connections` — joints
7. `loads` — standard + adversarial generation
8. `assembler` — stitch it all together
9. `red-team` — the attack engine
10. `report` — findings delivery

---

*"Every bridge gets a peer review. Ours also gets a red team."*
