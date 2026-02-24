#!/usr/bin/env python3
"""Natural Language Builder CLI — Red Team Your Bridge.

Usage:
    nlb "3-span steel plate girder over the Kishwaukee River on I-39
         in northern Illinois. 315-420-315 ft spans."

    nlb --input bridge.txt --output ./report/
    nlb site-recon --lat 42.28 --lon -89.09
    nlb --interactive
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import os
from pathlib import Path


# ── Pretty output helpers ─────────────────────────────────────────────

class Colors:
    RED = "\033[91m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

def status(icon: str, msg: str):
    print(f"{Colors.CYAN}{icon}{Colors.RESET} {msg}")

def success(msg: str):
    print(f"{Colors.GREEN}✓{Colors.RESET} {msg}")

def warn(msg: str):
    print(f"{Colors.YELLOW}⚠{Colors.RESET} {msg}")

def error(msg: str):
    print(f"{Colors.RED}✗{Colors.RESET} {msg}", file=sys.stderr)

def header(msg: str):
    print(f"\n{Colors.BOLD}{msg}{Colors.RESET}")

def risk_badge(rating: str) -> str:
    if rating == "RED":
        return f"{Colors.RED}🔴 RED{Colors.RESET}"
    elif rating == "YELLOW":
        return f"{Colors.YELLOW}🟡 YELLOW{Colors.RESET}"
    else:
        return f"{Colors.GREEN}🟢 GREEN{Colors.RESET}"

def severity_icon(sev: str) -> str:
    return {"CRITICAL": f"{Colors.RED}🔴", "WARNING": f"{Colors.YELLOW}⚠️", "NOTE": f"{Colors.DIM}📝"}.get(sev, "  ")


# ── NL Parser (extract bridge parameters from description) ───────────

def parse_description(text: str) -> dict:
    """Extract bridge parameters from natural language description.
    
    This is deliberately simple — it extracts what it can from patterns
    and leaves the rest for the user to confirm or the tools to infer.
    """
    import re
    
    params = {
        "description": text.strip(),
        "bridge_type": None,
        "spans": [],
        "num_girders": None,
        "girder_spacing_ft": None,
        "deck_width_ft": None,
        "girder_depth_in": None,
        "erection_method": None,
        "location": {},
        "water_crossing": False,
        "constraints": [],
    }
    
    lower = text.lower()
    
    # Bridge type
    type_patterns = {
        "steel_plate_girder": r"steel\s+(?:plate\s+)?girder",
        "prestressed_i": r"prestress(?:ed)?\s+(?:concrete\s+)?i[- ]?girder",
        "concrete_box": r"(?:cip|cast.in.place|concrete)\s+box\s+girder",
        "segmental_box": r"(?:segmental|pt\s+segmental)\s+box",
        "steel_truss": r"steel\s+truss",
        "concrete_slab": r"concrete\s+slab\s+bridge",
        "arch": r"arch\s+bridge",
        "cable_stayed": r"cable[- ]stayed",
    }
    for btype, pattern in type_patterns.items():
        if re.search(pattern, lower):
            params["bridge_type"] = btype
            break
    
    # Span lengths (e.g., "315-420-315 ft" or "315, 420, 315 feet" or "80 ft span")
    span_match = re.search(r"(\d+(?:\s*[-,]\s*\d+)*)\s*(?:ft|feet|foot)\s*(?:spans?)?", lower)
    if span_match:
        span_str = span_match.group(1)
        params["spans"] = [float(s.strip()) for s in re.split(r"[-,]\s*", span_str)]
    
    # Single span
    if not params["spans"]:
        single = re.search(r"(?:span\s+(?:of\s+)?|single\s+span\s+)(\d+)\s*(?:ft|feet|foot)", lower)
        if single:
            params["spans"] = [float(single.group(1))]
    
    # Number of girders
    girder_match = re.search(r"(\d+)\s+girder(?:\s+line)?s?", lower)
    if girder_match:
        params["num_girders"] = int(girder_match.group(1))
    
    # Girder spacing
    spacing_match = re.search(r"(?:at\s+|@\s*|spacing\s+(?:of\s+)?)(\d+(?:\.\d+)?)['\s]*(?:ft|feet|foot)?\s*(?:spacing|apart)?", lower)
    if spacing_match:
        params["girder_spacing_ft"] = float(spacing_match.group(1))
    
    # Deck width
    width_match = re.search(r"(\d+(?:\.\d+)?)['\s]*(?:ft|feet|foot)\s+(?:deck\s+)?width", lower)
    if width_match:
        params["deck_width_ft"] = float(width_match.group(1))
    
    # Girder depth
    depth_match = re.search(r"(\d+)[\"\s]*(?:in|inch)", lower)
    if depth_match:
        params["girder_depth_in"] = float(depth_match.group(1))
    
    # Erection method
    if "ilm" in lower or "incremental launch" in lower:
        params["erection_method"] = "ILM"
    elif "cantilever" in lower:
        params["erection_method"] = "cantilever"
    elif "crane" in lower:
        params["erection_method"] = "crane_erect"
    
    # Water crossing
    water_words = ["river", "creek", "stream", "channel", "waterway", "bayou", "inlet"]
    if any(w in lower for w in water_words):
        params["water_crossing"] = True
    
    # Location (state)
    states = {
        "alabama": (32.8, -86.8), "alaska": (64.0, -153.0), "arizona": (34.0, -111.0),
        "arkansas": (34.8, -92.2), "california": (36.8, -119.4), "colorado": (39.0, -105.5),
        "connecticut": (41.6, -72.7), "delaware": (39.0, -75.5), "florida": (27.6, -81.5),
        "georgia": (32.2, -83.4), "hawaii": (19.9, -155.6), "idaho": (44.1, -114.7),
        "illinois": (40.6, -89.4), "indiana": (40.3, -86.1), "iowa": (41.9, -93.1),
        "kansas": (39.0, -98.5), "kentucky": (37.8, -84.3), "louisiana": (30.5, -91.2),
        "maine": (45.4, -69.4), "maryland": (39.0, -76.6), "massachusetts": (42.4, -71.4),
        "michigan": (44.3, -84.5), "minnesota": (46.7, -94.7), "mississippi": (32.3, -89.4),
        "missouri": (38.6, -92.6), "montana": (46.9, -110.4), "nebraska": (41.1, -98.3),
        "nevada": (38.8, -116.4), "new hampshire": (43.5, -71.6), "new jersey": (40.1, -74.5),
        "new mexico": (34.5, -105.9), "new york": (43.0, -75.5), "north carolina": (35.8, -79.0),
        "north dakota": (47.5, -100.5), "ohio": (40.4, -82.9), "oklahoma": (35.0, -97.5),
        "oregon": (43.8, -120.6), "pennsylvania": (41.2, -77.2), "rhode island": (41.6, -71.5),
        "south carolina": (33.8, -81.2), "south dakota": (43.9, -99.9), "tennessee": (35.5, -86.6),
        "texas": (31.0, -100.0), "utah": (39.3, -111.1), "vermont": (44.0, -72.7),
        "virginia": (37.4, -78.2), "washington": (47.8, -120.7), "west virginia": (38.6, -80.6),
        "wisconsin": (43.8, -88.8), "wyoming": (43.1, -107.6),
    }
    for state_name, (lat, lon) in states.items():
        if state_name in lower:
            params["location"] = {"state": state_name.title(), "lat": lat, "lon": lon}
            break
    
    # Specific lat/lon from location mentions (rough — for well-known rivers)
    if "kishwaukee" in lower:
        params["location"] = {"state": "Illinois", "lat": 42.28, "lon": -89.09, "county": "Winnebago"}
    
    # Constraints
    constraint_patterns = [
        (r"no\s+equipment\s+in\s+(?:the\s+)?(?:river|water|valley|channel)", "No equipment in waterway"),
        (r"no\s+(?:explosives|blasting)", "No explosives"),
        (r"maintain\s+(?:traffic|one\s+lane)", "Maintain traffic during construction"),
        (r"navigable\s+(?:waterway|channel)", "Navigable waterway — vessel collision required"),
    ]
    for pattern, constraint in constraint_patterns:
        if re.search(pattern, lower):
            params["constraints"].append(constraint)
    
    return params


# ── Pipeline runner ───────────────────────────────────────────────────

def run_pipeline(params: dict, output_dir: Path | None = None, verbose: bool = False, dump_script: bool = False):
    """Run the full NLB pipeline from parsed parameters."""
    
    t0 = time.time()
    results = {}
    
    # ── Step 1: Site Recon ────────────────────────────────────────────
    status("🔍", "Site reconnaissance...")
    
    lat = params.get("location", {}).get("lat")
    lon = params.get("location", {}).get("lon")
    
    if lat and lon:
        try:
            from nlb.tools.site_recon import run_site_recon
            site = run_site_recon(lat, lon, params["description"])
            site_dict = site.to_dict() if hasattr(site, "to_dict") else site
            results["site"] = site_dict
            
            sdc = site_dict.get("seismic", {}).get("sdc", "?")
            wind = site_dict.get("wind", {}).get("v_ult", "?")
            scour = "scour zone" if site_dict.get("scour", {}).get("water_crossing") else "no scour"
            state = params["location"].get("state", "")
            county = params["location"].get("county", "")
            loc_str = f"{county + ', ' if county else ''}{state}"
            
            # Ensure soil layers exist for foundation tool
            if "layers" not in site_dict:
                site_dict["layers"] = [
                    {"soil_type": "stiff_clay", "top_depth_ft": 0, "thickness_ft": 15, "su_ksf": 1.5, "gamma_pcf": 120},
                    {"soil_type": "sand", "top_depth_ft": 15, "thickness_ft": 25, "phi_deg": 35, "gamma_pcf": 125, "N_spt": 30},
                    {"soil_type": "stiff_clay", "top_depth_ft": 40, "thickness_ft": 40, "su_ksf": 3.0, "gamma_pcf": 125},
                ]
                site_dict["gwt_depth_ft"] = 10.0
            
            success(f"Site: {loc_str} — SDC {sdc}, V={wind}mph, {scour}")
        except Exception as e:
            warn(f"Site recon failed: {e}")
            # Use conservative defaults
            site_dict = _default_site(params)
            results["site"] = site_dict
            success("Using conservative defaults")
    else:
        warn("No location found — using conservative defaults")
        site_dict = _default_site(params)
        results["site"] = site_dict
    
    # ── Step 2: Build Components ─────────────────────────────────────
    status("🏗️", "Building structural model...")
    
    num_spans = len(params.get("spans", [1]))
    num_supports = num_spans + 1  # abutments + piers
    
    # Superstructure
    try:
        from nlb.tools.superstructure import create_superstructure
        
        super_params = _build_superstructure_params(params)
        actual_num_girders = super_params.pop("_actual_num_girders", 5)
        superstructure = create_superstructure(**super_params)
        super_dict = _model_to_dict(superstructure)
        results["superstructure"] = super_dict
        results["superstructure"]["_actual_num_girders"] = actual_num_girders
        
        node_count = len(super_dict.get("nodes", []))
        elem_count = len(super_dict.get("elements", []))
        success(f"Superstructure: {node_count} nodes, {elem_count} elements")
    except Exception as e:
        error(f"Superstructure failed: {e}")
        if verbose:
            import traceback; traceback.print_exc()
        return None
    
    # Foundations
    try:
        from nlb.tools.foundation import create_foundation
        
        foundations = []
        for i in range(num_supports):
            fnd_params = _build_foundation_params(params, i, num_supports, site_dict)
            fnd = create_foundation(**fnd_params)
            foundations.append(_model_to_dict(fnd))
        results["foundations"] = foundations
        success(f"Foundations: {num_supports} supports")
    except Exception as e:
        warn(f"Foundation modeling failed: {e}")
        if verbose:
            import traceback; traceback.print_exc()
        foundations = [_mock_foundation(i) for i in range(num_supports)]
        results["foundations"] = foundations
    
    # Substructure
    try:
        from nlb.tools.substructure import create_substructure
        
        substructures = []
        for i in range(num_supports):
            sub_params = _build_substructure_params(params, i, num_supports)
            sub = create_substructure(**sub_params)
            substructures.append(_model_to_dict(sub))
        results["substructures"] = substructures
        success(f"Substructure: {num_supports} units")
    except Exception as e:
        warn(f"Substructure modeling failed: {e}")
        if verbose:
            import traceback; traceback.print_exc()
        substructures = [_mock_substructure(i) for i in range(num_supports)]
        results["substructures"] = substructures
    
    # Bearings
    try:
        from nlb.tools.bearings import create_bearing
        
        bearings = []
        for i in range(num_supports):
            brg_params = _build_bearing_params(params, i, num_supports)
            brg = create_bearing(**brg_params)
            bearings.append(_model_to_dict(brg))
        results["bearings"] = bearings
        success(f"Bearings: {num_supports} sets")
    except Exception as e:
        warn(f"Bearing modeling failed: {e}")
        if verbose:
            import traceback; traceback.print_exc()
        bearings = [_mock_bearing(i) for i in range(num_supports)]
        results["bearings"] = bearings
    
    # Loads
    try:
        from nlb.tools.loads import generate_loads
        
        load_params = _build_load_params(params, site_dict)
        loads = generate_loads(**load_params)
        load_dict = _model_to_dict(loads)
        results["loads"] = load_dict
        
        n_std = len(load_dict.get("cases", []))
        n_adv = len(load_dict.get("adversarial_cases", []))
        n_combo = load_dict.get("total_combinations", 0)
        success(f"Loads: {n_std} standard + {n_adv} adversarial cases, {n_combo} combinations")
    except Exception as e:
        warn(f"Load generation failed: {e}")
        if verbose:
            import traceback; traceback.print_exc()
        load_dict = {"cases": [], "combinations": [], "adversarial_cases": []}
        results["loads"] = load_dict
    
    # ── Step 3: Assemble ─────────────────────────────────────────────
    status("📐", "Assembling complete model...")
    
    try:
        from nlb.tools.assembler import assemble_model
        
        model = assemble_model(
            site=site_dict,
            foundations=foundations,
            substructures=substructures,
            bearings=bearings,
            superstructure=super_dict,
            loads=load_dict,
        )
        model_dict = _model_to_dict(model)
        results["model"] = model_dict
        
        success(f"Model: {model_dict.get('node_count', '?')} nodes, "
                f"{model_dict.get('element_count', '?')} elements, "
                f"{model_dict.get('load_combinations', '?')} load combos")
    except Exception as e:
        warn(f"Assembly failed: {e}")
        if verbose:
            import traceback; traceback.print_exc()
        model_dict = {"node_count": 0, "element_count": 0}
        results["model"] = model_dict
    
    # ── Dump script if requested ────────────────────────────────────
    if dump_script:
        try:
            from nlb.tools.assembler import generate_script, AssembledModel
            if isinstance(model, AssembledModel):
                script = generate_script(model)
                out_path = Path(output_dir) / "opensees_script.py"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(script)
                success(f"OpenSees script written to {out_path} ({len(script):,} chars)")
            else:
                warn("Model not assembled — cannot generate script")
        except Exception as e:
            warn(f"Script dump failed: {e}")
            if verbose:
                import traceback; traceback.print_exc()
        return results

    # ── Step 4: Run Analysis ─────────────────────────────────────────
    status("⚡", "Running analysis...")
    
    try:
        from nlb.tools.assembler import run_analysis, AssembledModel
        
        if isinstance(model, AssembledModel):
            analysis = run_analysis(model)
            analysis_dict = _model_to_dict(analysis)
        else:
            analysis_dict = _mock_analysis(model_dict)
        results["analysis"] = analysis_dict
        success("Analysis complete")
    except Exception as e:
        warn(f"Analysis failed (OpenSeesPy may not be available): {e}")
        if verbose:
            import traceback; traceback.print_exc()
        analysis_dict = _mock_analysis(model_dict)
        results["analysis"] = analysis_dict
    
    # ── Step 5: Red Team ─────────────────────────────────────────────
    status("🔴", "Red team attacking...")
    red_team_obj = None
    
    try:
        from nlb.tools.red_team import run_red_team
        
        # Pass raw analysis results + model to red team
        # It will auto-compute real DCRs from forces + section capacities
        model_dict = dict(results.get("model", {}))
        # Inject superstructure sections so red team can compute girder capacities
        super_sects = results.get("superstructure", {}).get("sections", [])
        if super_sects:
            model_dict["superstructure_sections"] = super_sects
        
        red_team_obj = run_red_team(
            analysis_results=analysis_dict,
            model=model_dict,
            site_constraints={"constraints": params.get("constraints", [])},
        )
        red_dict = _model_to_dict(red_team_obj)
        results["red_team"] = red_dict
    except Exception as e:
        warn(f"Red team failed: {e}")
        if verbose:
            import traceback; traceback.print_exc()
        red_dict = {"findings": [], "risk_rating": "UNKNOWN", "summary": "Red team analysis could not complete."}
        results["red_team"] = red_dict
    
    # ── Step 6: Report ───────────────────────────────────────────────
    elapsed = time.time() - t0
    
    print(f"\n{'━' * 60}")
    header(f"RED TEAM REPORT — {params.get('location', {}).get('state', 'Bridge')}")
    
    rating = red_dict.get("risk_rating", "UNKNOWN")
    print(f"Risk: {risk_badge(rating)}")
    print()
    
    findings = red_dict.get("findings", [])
    criticals = [f for f in findings if f.get("severity") == "CRITICAL"]
    warnings = [f for f in findings if f.get("severity") == "WARNING"]
    notes = [f for f in findings if f.get("severity") == "NOTE"]
    
    if criticals:
        for f in criticals[:3]:
            dcr_str = f" DCR = {f['dcr']:.2f}" if f.get('dcr') else ""
            print(f"  {severity_icon('CRITICAL')} CRITICAL:{Colors.RESET} {f.get('description', '')}{dcr_str}")
    
    if warnings:
        for f in warnings[:3]:
            dcr_str = f" DCR = {f['dcr']:.2f}" if f.get('dcr') else ""
            print(f"  {severity_icon('WARNING')} WARNING:{Colors.RESET} {f.get('description', '')}{dcr_str}")
    
    if notes:
        shown = min(3, 3 - len(criticals) - len(warnings))
        for f in notes[:max(0, shown)]:
            dcr_str = f" DCR = {f['dcr']:.2f}" if f.get('dcr') else ""
            print(f"  {severity_icon('NOTE')} NOTE:{Colors.RESET} {f.get('description', '')}{dcr_str}")
    
    if not findings:
        print(f"  {Colors.GREEN}No findings — design looks clean.{Colors.RESET}")
    
    remaining = len(findings) - 6
    if remaining > 0:
        print(f"  {Colors.DIM}... and {remaining} more findings{Colors.RESET}")
    
    print(f"\n{Colors.DIM}Analysis time: {elapsed:.1f}s{Colors.RESET}")
    
    # Save outputs
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Technical report
        try:
            from nlb.tools.report import generate_report
            # generate_report expects dicts — pass them through
            tech_report = generate_report(
                red_team=red_team_obj,
                model_info=model_dict,
                site=site_dict,
                tier="technical"
            )
            report_path = output_dir / "red-team-report.md"
            report_path.write_text(tech_report)
            print(f"📄 Technical report: {report_path}")
        except Exception as e:
            warn(f"Report generation failed: {e}")
        
        # Raw JSON
        raw_path = output_dir / "results.json"
        raw_path.write_text(json.dumps(results, indent=2, default=str))
        print(f"📊 Raw data: {raw_path}")
        
        # Executive summary
        try:
            from nlb.tools.report import generate_report
            exec_report = generate_report(
                red_team=red_team_obj,
                model_info=model_dict,
                site=site_dict,
                tier="executive"
            )
            exec_path = output_dir / "executive-summary.md"
            exec_path.write_text(exec_report)
            print(f"📋 Executive summary: {exec_path}")
        except Exception:
            pass
    
    print(f"{'━' * 60}")
    
    return results


# ── Parameter builders (map parsed NL → tool inputs) ─────────────────

def _build_superstructure_params(params: dict) -> dict:
    """Build superstructure tool parameters from parsed description."""
    spans = params.get("spans", [100.0])
    btype = params.get("bridge_type", "steel_plate_girder")
    
    type_map = {
        "steel_plate_girder": "steel_plate_girder_composite",
        "prestressed_i": "prestressed_i_girder",
        "concrete_box": "cip_box_girder",
        "segmental_box": "segmental_box_girder",
        "steel_truss": "steel_truss",
        "concrete_slab": "concrete_slab",
        "arch": "arch",
        "cable_stayed": "cable_stayed",
    }
    
    # Use single-line model (1 girder) for analysis.
    # Multi-girder models cause force amplification from penalty constraints.
    # Distribution factors handle transverse load distribution.
    p = {
        "bridge_type": type_map.get(btype, "steel_plate_girder_composite"),
        "span_lengths_ft": spans,
        "num_girders": 1,  # Single-line model; DFs applied in post-processing
        "girder_spacing_ft": params.get("girder_spacing_ft", 8.0),
        "_actual_num_girders": params.get("num_girders", 5),  # preserved for DF calc
    }
    
    if params.get("deck_width_ft"):
        p["deck_width_ft"] = params["deck_width_ft"]
    if params.get("girder_depth_in"):
        p["girder_depth_in"] = params["girder_depth_in"]
    
    return p


def _build_foundation_params(params: dict, index: int, total: int, site: dict) -> dict:
    """Build foundation parameters for a given support."""
    is_abutment = (index == 0 or index == total - 1)
    
    return {
        "foundation_type": "spread_footing" if is_abutment else "drilled_shaft",
        "params": {
            "diameter_ft": 4.0 if is_abutment else 7.0,
            "length_ft": 15.0 if is_abutment else 60.0,
            "width_ft": 20.0 if is_abutment else None,
            "depth_ft": 6.0 if is_abutment else None,
        },
        "site_profile": site,
    }


def _build_substructure_params(params: dict, index: int, total: int) -> dict:
    """Build substructure parameters for a given support."""
    is_abutment = (index == 0 or index == total - 1)
    
    if is_abutment:
        return {
            "sub_type": "seat_abutment",
            "seat_width_ft": 4.0,
            "backwall_height_ft": 7.0,
            "num_bearings": params.get("num_girders", 5),
            "bearing_spacing_ft": params.get("girder_spacing_ft", 8.0),
        }
    else:
        return {
            "sub_type": "multi_column_bent",
            "num_columns": 2,
            "column_diameter_ft": 7.0,
            "column_height_ft": 25.0,
            "column_spacing_ft": 20.0,
            "cap_width_ft": 4.0,
            "cap_depth_ft": 5.0,
            "fc_ksi": 4.0,
            "fy_ksi": 60.0,
        }


def _build_bearing_params(params: dict, index: int, total: int) -> dict:
    """Build bearing parameters for a given support."""
    is_abutment = (index == 0 or index == total - 1)
    
    return {
        "bearing_type": "elastomeric" if is_abutment else "ptfe_sliding",
        "vertical_capacity_kips": 500.0,
    }


def _build_load_params(params: dict, site: dict) -> dict:
    """Build load generation parameters."""
    from nlb.tools.loads import BridgeGeometry
    
    spans = params.get("spans", [100.0])
    n_girders = params.get("num_girders", 5)
    spacing = params.get("girder_spacing_ft", 8.0)
    deck_w = params.get("deck_width_ft") or (n_girders * spacing)
    depth = params.get("girder_depth_in") or 48.0
    
    # BridgeGeometry.span_ft expects a single float (max span for load generation)
    max_span = max(spans) if spans else 100.0
    
    geom = BridgeGeometry(
        span_ft=max_span,
        girder_spacing_ft=spacing,
        num_girders=n_girders,
        deck_width_ft=deck_w,
        girder_depth_in=depth,
    )
    
    return {
        "geom": geom,
        "site_profile": site,
    }


# ── Helpers ───────────────────────────────────────────────────────────

def _model_to_dict(obj) -> dict:
    """Convert a dataclass or object to dict."""
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dataclass_fields__"):
        import dataclasses
        return dataclasses.asdict(obj)
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
    return {}


def _default_site(params: dict) -> dict:
    """Conservative default site profile when recon fails."""
    return {
        "coordinates": params.get("location", {"lat": 40.0, "lon": -90.0}),
        "seismic": {"pga": 0.10, "ss": 0.25, "s1": 0.10, "sds": 0.20, "sd1": 0.13, "site_class": "D", "sdc": "B"},
        "wind": {"v_ult": 115, "exposure": "C"},
        "thermal": {"t_min": -10, "t_max": 110, "delta_t": 120},
        "scour": {"water_crossing": params.get("water_crossing", False), "design_flood": "Q100", "check_flood": "Q500"},
        "frost_depth_ft": 3.0,
        "soil": {"site_class": "D", "description": "Stiff soil (default)"},
        "climate_zone": "cold",
        "layers": [
            {"soil_type": "stiff_clay", "top_depth_ft": 0, "thickness_ft": 15, "su_ksf": 1.5, "gamma_pcf": 120},
            {"soil_type": "sand", "top_depth_ft": 15, "thickness_ft": 25, "phi_deg": 35, "gamma_pcf": 125, "N_spt": 30},
            {"soil_type": "stiff_clay", "top_depth_ft": 40, "thickness_ft": 40, "su_ksf": 3.0, "gamma_pcf": 125},
        ],
        "gwt_depth_ft": 10.0,
        "warnings": ["Using conservative defaults — no API data available"],
    }


def _mock_foundation(index: int) -> dict:
    return {"nodes": [], "elements": [], "springs": [], "materials": [], "top_node": index * 100 + 1, "base_node": index * 100, "cases": {}}


def _mock_substructure(index: int) -> dict:
    return {"nodes": [], "elements": [], "sections": [], "materials": [], "top_nodes": [index * 100 + 50], "base_nodes": [index * 100 + 1], "cap_nodes": [index * 100 + 50]}


def _mock_bearing(index: int) -> dict:
    return {"nodes": [], "elements": [], "materials": [], "constraints": [], "top_nodes": [index * 100 + 60], "bottom_nodes": [index * 100 + 50], "properties": {}, "cases": {}}


def _mock_analysis(model_dict: dict) -> dict:
    return {"envelopes": {}, "dcr": {}, "reactions": {}, "displacements": {}, "modal": {}, "controlling_cases": []}


# ── CLI entry point ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="nlb",
        description="Red Team Your Bridge — Natural Language → Nonlinear FEA → Adversarial Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  nlb "3-span steel plate girder, 315-420-315 ft, Illinois"
  nlb --input bridge.txt --output ./report/
  nlb site-recon --lat 42.28 --lon -89.09
        """,
    )
    
    parser.add_argument("description", nargs="*", help="Bridge description in natural language")
    parser.add_argument("--input", "-i", help="Read description from file")
    parser.add_argument("--output", "-o", default="./nlb-output", help="Output directory (default: ./nlb-output)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output and tracebacks")
    parser.add_argument("--json", action="store_true", help="Output raw JSON instead of pretty print")
    parser.add_argument("--parse-only", action="store_true", help="Only parse the description, don't run analysis")
    parser.add_argument("--dump-script", action="store_true", help="Dump generated OpenSees script to output dir (no analysis)")
    parser.add_argument("--site-recon", nargs=2, metavar=("LAT", "LON"), type=float, help="Run site recon only")
    
    args = parser.parse_args()
    
    # Handle site-recon mode
    if args.site_recon:
        from nlb.tools.site_recon import run_site_recon
        lat, lon = args.site_recon
        status("🔍", f"Site recon: {lat}, {lon}")
        desc = " ".join(args.description) if args.description else ""
        result = run_site_recon(lat, lon, desc)
        result_dict = result.to_dict() if hasattr(result, "to_dict") else _model_to_dict(result)
        print(json.dumps(result_dict, indent=2))
        return
    
    # Main pipeline
    if not args.description and not args.input:
        parser.print_help()
        return
    
    # Get description
    if args.input:
        text = Path(args.input).read_text()
    else:
        text = " ".join(args.description)
    
    print(f"\n{Colors.BOLD}🔴 RED TEAM YOUR BRIDGE{Colors.RESET}")
    print(f"{Colors.DIM}{'─' * 60}{Colors.RESET}\n")
    
    # Parse
    status("📝", "Parsing description...")
    params = parse_description(text)
    
    spans = params.get("spans", [])
    btype = (params.get("bridge_type") or "unknown").replace("_", " ")
    n_spans = len(spans)
    span_str = "-".join(str(int(s)) for s in spans) + " ft" if spans else "unknown"
    
    success(f"Type: {btype}, {n_spans} span(s): {span_str}")
    
    if params.get("num_girders"):
        success(f"Girders: {params['num_girders']} @ {params.get('girder_spacing_ft', '?')}' spacing")
    if params.get("location"):
        success(f"Location: {params['location'].get('state', '?')} ({params['location'].get('lat', '?')}, {params['location'].get('lon', '?')})")
    if params.get("erection_method"):
        success(f"Erection: {params['erection_method']}")
    if params.get("constraints"):
        for c in params["constraints"]:
            warn(f"Constraint: {c}")
    
    if args.parse_only:
        print(json.dumps(params, indent=2))
        return
    
    print()
    
    # Run pipeline
    results = run_pipeline(params, output_dir=Path(args.output), verbose=args.verbose, dump_script=getattr(args, 'dump_script', False))
    
    if args.json and results:
        print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
