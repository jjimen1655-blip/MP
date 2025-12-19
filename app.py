import os
import re
from dataclasses import dataclass
from typing import Literal, Optional, Set, Tuple

import streamlit as st
from openai import OpenAI
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics

MODEL_NAME = "gpt-5-mini"

# ---------- REFERENCE INFO ----------
def activity_factor_reference_table():
    st.markdown("### Activity Factor Reference")

    st.table([
        {
            "Modifier": "1.20",
            "Label": "Sedentary",
            "What this actually means":
                "Desk job + <5,000 steps/day; no structured exercise or ≤1x/week",
            "Typical patient examples":
                "Office workers, residents, drivers, remote workers",
        },
        {
            "Modifier": "1.30",
            "Label": "Lightly active",
            "What this actually means":
                "5,000-7,500 steps/day OR exercise 1-3x/week (≤30 min)",
            "Typical patient examples":
                "Dog walking, casual gym, yoga, light cycling",
        },
        {
            "Modifier": "1.40",
            "Label": "Moderately active",
            "What this actually means":
                "7,500-10,000 steps/day AND exercise 3-4x/week",
            "Typical patient examples":
                "Gym ~45 min, jogging, rec sports",
        },
        {
            "Modifier": "1.50",
            "Label": "Very active",
            "What this actually means":
                "10,000-12,500 steps/day AND hard exercise 5-6x/week",
            "Typical patient examples":
                "CrossFit, heavy lifting, distance running",
        },
        {
            "Modifier": "1.60-1.70",
            "Label": "Athlete / Labor",
            "What this actually means":
                "Manual labor OR 2-a-day training most days",
            "Typical patient examples":
                "Construction workers, competitive athletes",
        },
    ])


def insulin_resistance_reference():
    st.markdown("### A1C-based carb guidance (g/kg/day)")
    st.caption(
        "Use these as clinical guidance bands (not diagnostic advice). "
        "Cut points: Normal <5.7%, Prediabetes 5.7-6.4%, Diabetes >=6.5%. "
        "Within diabetes, the 'near goal / above goal / very high' split is an internal guidance bucket."
    )
    st.table([
        {
            "Band (A1C)": "Normal / no IR (<5.7%)",
            "Suggested carb range (g/kg/day)": "2.0-3.0",
            "Recommended cap (g/kg/day)": "3.0",
        },
        {
            "Band (A1C)": "Prediabetes / mild IR (5.7-6.4%)",
            "Suggested carb range (g/kg/day)": "1.5-2.5",
            "Recommended cap (g/kg/day)": "2.5",
        },
        {
            "Band (A1C)": "T2DM near goal (6.5-6.9%)",
            "Suggested carb range (g/kg/day)": "1.0-2.0",
            "Recommended cap (g/kg/day)": "2.0",
        },
        {
            "Band (A1C)": "T2DM above goal (7.0-8.4%)",
            "Suggested carb range (g/kg/day)": "0.8-1.5",
            "Recommended cap (g/kg/day)": "1.5",
        },
        {
            "Band (A1C)": "T2DM very high (>=8.5%)",
            "Suggested carb range (g/kg/day)": "0.5-1.0",
            "Recommended cap (g/kg/day)": "1.0",
        },
    ])


# ---------- TEXT NORMALIZATION ----------
def normalize_text_for_parsing(text: str) -> str:
    """
    Normalize common Unicode characters that frequently break parsing/PDF rendering:
    - Dash variants (– — − etc.) -> "-"
    - Curly quotes -> straight quotes
    - NBSP -> space
    - Ellipsis -> "..."
    - Bullet chars -> "-"
    """
    replacements = {
        # Dashes / hyphens / minus
        "\u2010": "-",  # hyphen
        "\u2011": "-",  # non-breaking hyphen
        "\u2012": "-",  # figure dash
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
        "\u2212": "-",  # minus sign
        "\u00AD": "-",  # soft hyphen

        # Quotes
        "\u2018": "'",  # left single quote
        "\u2019": "'",  # right single quote
        "\u201C": '"',  # left double quote
        "\u201D": '"',  # right double quote

        # Spaces / punctuation
        "\u00A0": " ",  # non-breaking space
        "\u2026": "...",  # ellipsis

        # Bullets
        "\u2022": "-",  # bullet
        "\u25CF": "-",  # black circle
        "\u25E6": "-",  # white bullet
        "\u2043": "-",  # hyphen bullet
    }

    for bad, good in replacements.items():
        text = text.replace(bad, good)

    # Normalize line starts that look like bullets but aren't "- "
    fixed_lines = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("-", "*")) and not stripped.startswith("- "):
            stripped = stripped[1:].lstrip()
            fixed_lines.append("- " + stripped)
        else:
            fixed_lines.append(line)
    return "\n".join(fixed_lines)


# ---------- MEALPLAN MACRO SANITIZATION (FIX) ----------
_MACROS_LINE_RE = re.compile(
    r'^(?P<prefix>Approx:|Aproximado:|Aprox:)\s*'
    r'(?P<kcal>-?\d+(?:\.\d+)?)\s*kcal,\s*'
    r'P:\s*(?P<p>-?\d+(?:\.\d+)?)\s*g,\s*'
    r'C:\s*(?P<c>-?\d+(?:\.\d+)?)\s*g,\s*'
    r'F:\s*(?P<f>-?\d+(?:\.\d+)?)\s*g,\s*'
    r'Na:\s*(?P<na>-?\d+(?:\.\d+)?)\s*mg\s*$',
    flags=re.IGNORECASE
)

def _safe_float(x: str, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def sanitize_and_rebalance_macro_lines(text: str) -> str:
    """
    Fixes AI-output macro lines like:
      Approx: 520 kcal, P: 40 g, C: -5 g, F: 28 g, Na: 900 mg

    Rules:
    - Clamp P/C/F/Na/kcal to >= 0
    - Rebalance FAT to match calories:
        fat_g := max(0, (kcal - 4P - 4C) / 9)
      Then recompute kcal := 4P + 4C + 9F to keep internal consistency.
    - Only modifies lines that match the strict "Approx/Aprox/Aproximado" pattern.
    """
    out_lines = []
    for raw in text.splitlines():
        line = raw.rstrip("\n")
        m = _MACROS_LINE_RE.match(line.strip())
        if not m:
            out_lines.append(line)
            continue

        prefix = m.group("prefix")
        kcal = max(0.0, _safe_float(m.group("kcal")))
        p = max(0.0, _safe_float(m.group("p")))
        c = max(0.0, _safe_float(m.group("c")))
        na = max(0.0, _safe_float(m.group("na")))

        # Rebalance fat based on kcal after clamping P/C
        fat = (kcal - (4.0 * p) - (4.0 * c)) / 9.0
        fat = max(0.0, fat)

        # Round to nice display values
        p_i = int(round(p))
        c_i = int(round(c))
        fat_i = int(round(fat))
        na_i = int(round(na))

        # Recompute kcal from rounded macros so the line is consistent
        kcal_i = int(round((4 * p_i) + (4 * c_i) + (9 * fat_i)))

        out_lines.append(f"{prefix} {kcal_i} kcal, P: {p_i} g, C: {c_i} g, F: {fat_i} g, Na: {na_i} mg")

    return "\n".join(out_lines)


def add_section_spacing(text: str) -> str:
    """
    Option A:
    Adds blank lines between major sections WITHOUT inserting blank lines inside Day blocks.
    Also removes stray "Meal Plan (English/Spanish)" headers if the model inserts them.
    """
    lines = text.splitlines()

    major_headers: Set[str] = {
        "Cost summary (estimates only)",
        "Cost summary (rough estimates only)",
        "Grocery list (scaled for ~1 person, 14 days)",
        "Grocery list (grouped by category)",
        "Cooking instructions for selected main meals",
        "PRICE DISCLAIMER:",
        "DISCLAIMER:",
        # Spanish
        "Resumen de costos (estimaciones aproximadas)",
        "Lista del súper (agrupada por categoría)",
        "Instrucciones de cocina para algunas comidas principales",
        "AVISO DE PRECIOS:",
        "DESCARGO DE RESPONSABILIDAD:",
    }

    grocery_headers: Set[str] = {
        "Produce:",
        "Protein:",
        "Dairy:",
        "Grains / Starches:",
        "Pantry:",
        "Frozen:",
        "Other:",
        # Spanish
        "Productos frescos:",
        "Proteínas:",
        "Lácteos:",
        "Granos / Almidones:",
        "Despensa:",
        "Congelados:",
        "Otros:",
    }

    banned_single_line_headers: Set[str] = {
        "Meal Plan (English)",
        "Meal Plan (Spanish)",
        "Meal Plan (Español)",
        "Plan de comidas (Español)",
        "Plan de comidas",
    }

    out = []
    in_day_block = False
    day_prefixes = ("Day ", "Día ", "Dia ")

    for raw in lines:
        line = raw.rstrip("\n")
        stripped = line.strip()

        if stripped in banned_single_line_headers:
            continue

        if any(stripped.startswith(p) for p in day_prefixes):
            in_day_block = True
            if out and out[-1].strip() != "":
                out.append("")
            out.append(line)
            continue

        if in_day_block and (stripped in major_headers or stripped in grocery_headers):
            in_day_block = False

        if not in_day_block and (stripped in major_headers or stripped in grocery_headers):
            if out and out[-1].strip() != "":
                out.append("")
            out.append(line)
            continue

        out.append(line)

    spaced = "\n".join(out)
    spaced = re.sub(r"\n{3,}", "\n\n", spaced).strip()
    return spaced


def add_recipe_spacing_and_dividers(text: str, divider_len: int = 48) -> str:
    """
    Adds spacing BETWEEN recipes and inserts a horizontal divider BETWEEN recipes
    in the "Cooking instructions..." section only.

    - Does NOT insert blank lines inside Day blocks.
    - Divider is ASCII hyphens only, safe for PDF parsing.
    """
    lines = text.splitlines()

    recipe_section_headers = {
        "Cooking instructions for selected main meals",
        "Instrucciones de cocina para algunas comidas principales",
    }

    in_recipe_section = False
    saw_first_recipe = False
    out = []

    divider = "-" * max(10, int(divider_len))

    for raw in lines:
        line = raw.rstrip("\n")
        stripped = line.strip()

        if stripped in recipe_section_headers:
            in_recipe_section = True
            saw_first_recipe = False
            out.append(line)
            continue

        if not in_recipe_section:
            out.append(line)
            continue

        is_recipe_header = stripped.startswith("Recipe:") or stripped.startswith("Receta:")

        if is_recipe_header:
            if saw_first_recipe:
                if out and out[-1].strip() != "":
                    out.append("")
                out.append(divider)
                out.append("")
            else:
                if out and out[-1].strip() != "":
                    out.append("")
                saw_first_recipe = True

            out.append(line)
            continue

        out.append(line)

    result = "\n".join(out)
    result = re.sub(r"\n{3,}", "\n\n", result).strip()
    return result


def format_end_sections(text: str) -> str:
    """
    Cleans up the post-Day-14 summary area so:
    - Daily targets summary is on its own line
    - Daily supplements becomes its own section, one-per-line bullets, with instructions preserved
    - Cost summary header is on its own line with spacing
    Works for both English and Spanish variants.
    """
    lines = text.splitlines()
    out = []
    in_supplements = False
    supplements_header = None  # "Daily supplements:" or "Suplementos diarios:" (preserve language)

    # Headers that end the supplements block
    section_starters = {
        "Cost summary (rough estimates only)",
        "Cost summary (estimates only)",
        "Grocery list (grouped by category)",
        "Cooking instructions for selected main meals",
        "PRICE DISCLAIMER:",
        "DISCLAIMER:",
        # Spanish
        "Resumen de costos (estimaciones aproximadas)",
        "Lista del súper (agrupada por categoría)",
        "Instrucciones de cocina para algunas comidas principales",
        "AVISO DE PRECIOS:",
        "DESCARGO DE RESPONSABILIDAD:",
    }

    def _emit_supplements_block(supp_str: str, header: str):
        out.append("")  # blank line before supplements section
        out.append(header)

        items = [x.strip() for x in supp_str.split(",") if x.strip()]
        for it in items:
            if it.startswith("- "):
                out.append(it)
            else:
                out.append(f"- {it}")

    for raw in lines:
        line = raw.rstrip("\n")
        s = line.strip()

        # Stop supplements mode if a new major section starts
        if in_supplements and s in section_starters:
            in_supplements = False
            supplements_header = None

        # --- Case A: legacy combined line ---
        # "Daily macro target (primary individual): ...; Daily supplements: ..."
        if s.startswith("Daily macro target (primary individual):"):
            if "; Daily supplements" in s:
                left, right = s.split("; Daily supplements", 1)
                out.append(left.strip())

                # Parse items after "Daily supplements"
                supp_str = right
                if ":" in supp_str:
                    supp_str = supp_str.split(":", 1)[1].strip()
                else:
                    supp_str = supp_str.strip()

                supplements_header = "Daily supplements:"
                _emit_supplements_block(supp_str, supplements_header)
                in_supplements = True
                continue

            out.append(line)
            in_supplements = False
            supplements_header = None
            continue

        # --- Case B: new combined line (English) ---
        # "Daily targets summary: ... Daily supplements: ..."
        if s.lower().startswith("daily targets summary:"):
            low = s.lower()
            if "daily supplements:" in low:
                idx = low.find("daily supplements:")
                targets_part = s[:idx].rstrip(" .")
                supp_part = s[idx + len("daily supplements:"):].strip()

                out.append(targets_part.strip() + ".")
                supplements_header = "Daily supplements:"
                _emit_supplements_block(supp_part, supplements_header)
                in_supplements = True
                continue

            out.append(line)
            continue

        # --- Case C: new combined line (Spanish) ---
        # "Resumen de objetivos diarios: ... Suplementos diarios: ..."
        if s.lower().startswith("resumen de objetivos diarios:"):
            low = s.lower()
            if "suplementos diarios:" in low:
                idx = low.find("suplementos diarios:")
                targets_part = s[:idx].rstrip(" .")
                supp_part = s[idx + len("suplementos diarios:"):].strip()

                out.append(targets_part.strip() + ".")
                supplements_header = "Suplementos diarios:"
                _emit_supplements_block(supp_part, supplements_header)
                in_supplements = True
                continue

            out.append(line)
            continue

        # If standalone supplements header exists, preserve and enter supplements mode
        if s in ("Daily supplements:", "Suplementos diarios:"):
            out.append(s)
            in_supplements = True
            supplements_header = s
            continue

        # Cost summary header cleanup
        if s in ("Cost summary (rough estimates only)", "Cost summary (estimates only)"):
            if out and out[-1].strip() != "":
                out.append("")
            out.append(s)
            continue

        out.append(line)

    cleaned = "\n".join(out)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


# ---------- OPENAI CLIENT SETUP ----------
api_key = None
try:
    api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found in Streamlit secrets or environment variables")

client = OpenAI(api_key=api_key)

# ---------- DATA STRUCTURES ----------
Intensity = Literal["Gentle", "Moderate", "Aggressive"]
GoalMode = Literal["Weight loss", "Maintenance", "Weight gain"]
TrainingVolume = Literal["Moderate (3–4 days/week)", "High volume (5–6 days/week)"]
FatMode = Literal["Auto", "Manual"]


@dataclass
class MacroResult:
    rmr: float
    tdee: float
    target_kcal: float
    protein_g: float
    fat_g: float
    carbs_g: float
    protein_pct: float
    fat_pct: float
    carbs_pct: float


# ---------- CALCULATION LOGIC ----------
def calculate_macros(
    sex: str,
    age: int,
    height_cm: float,
    weight_current_kg: float,
    weight_goal_kg: float,
    weight_source: Literal["Current", "Goal"],
    activity_factor: float,
    goal_mode: GoalMode,
    intensity: Optional[Intensity] = None,
    use_estimated_maintenance: bool = True,
    maintenance_kcal_known: Optional[float] = None,
    surplus_kcal: float = 300.0,

    protein_g_per_kg: float = 1.4,

    # Priority mode (carb-first)
    carbs_g_per_kg_cap: Optional[float] = None,
    carb_cap_basis: Literal["Current", "Macro weight"] = "Current",
    fat_mode: FatMode = "Manual",
    fat_g_per_kg_manual: float = 0.7,

    # weight gain only
    carbs_g_per_kg_gain: Optional[float] = None,
) -> MacroResult:
    """Calculate RMR, TDEE, target calories and macros using Mifflin-St Jeor."""

    if sex.upper() == "M":
        rmr = 10 * weight_current_kg + 6.25 * height_cm - 5 * age + 5
    else:
        rmr = 10 * weight_current_kg + 6.25 * height_cm - 5 * age - 161

    tdee = rmr * activity_factor

    if use_estimated_maintenance or (maintenance_kcal_known is None):
        maintenance_kcal = tdee
    else:
        maintenance_kcal = float(maintenance_kcal_known)

    if goal_mode == "Weight loss":
        deficit_map = {"Gentle": 250, "Moderate": 500, "Aggressive": 750}
        chosen_intensity: Intensity = intensity or "Moderate"
        target_kcal = max(maintenance_kcal - deficit_map[chosen_intensity], 1200)
    elif goal_mode == "Maintenance":
        target_kcal = max(maintenance_kcal, 1200)
    else:
        target_kcal = max(maintenance_kcal + float(surplus_kcal), 1200)

    weight_for_macros = weight_current_kg if weight_source == "Current" else weight_goal_kg
    carb_cap_weight_kg = weight_current_kg if carb_cap_basis == "Current" else weight_for_macros

    protein_g = weight_for_macros * float(protein_g_per_kg)
    kcal_protein = protein_g * 4

    kcal_carbs = 0.0
    kcal_fat = 0.0
    carbs_g = 0.0
    fat_g = 0.0

    # OPTION A: carb cap + Auto fat => carbs prioritized first, then protein, then fat remainder
    # Weight gain: desired carbs used but never exceed carb cap.
    if carbs_g_per_kg_cap is not None and fat_mode == "Auto":
        carb_cap_g = float(carbs_g_per_kg_cap) * float(carb_cap_weight_kg)

        if goal_mode == "Weight gain" and carbs_g_per_kg_gain is not None:
            desired_carbs_g = weight_for_macros * float(carbs_g_per_kg_gain)
            carbs_g = min(desired_carbs_g, carb_cap_g)
        else:
            carbs_g = carb_cap_g

        kcal_carbs = carbs_g * 4

        fat_min_g = 0.3 * weight_for_macros
        fat_max_g = (1.2 * weight_for_macros) if goal_mode == "Weight gain" else (1.5 * weight_for_macros)

        min_fat_kcal = fat_min_g * 9
        if (kcal_protein + kcal_carbs + min_fat_kcal) > target_kcal:
            kcal_carbs = max(target_kcal - kcal_protein - min_fat_kcal, 0.0)
            carbs_g = min(carbs_g, (kcal_carbs / 4.0) if kcal_carbs > 0 else 0.0)
            kcal_carbs = carbs_g * 4

        fat_kcal = max(target_kcal - (kcal_protein + kcal_carbs), 0.0)
        fat_g = (fat_kcal / 9.0) if fat_kcal > 0 else 0.0
        fat_g = max(fat_min_g, min(fat_g, fat_max_g))
        kcal_fat = fat_g * 9

        kcal_carbs = max(target_kcal - (kcal_protein + kcal_fat), 0.0)
        carbs_g = min(carb_cap_g, (kcal_carbs / 4.0) if kcal_carbs > 0 else 0.0)
        kcal_carbs = carbs_g * 4

    else:
        # Default behavior:
        # - Weight gain: carbs targeted (g/kg) if provided, fat remainder with clamp 0.6-1.0 g/kg
        # - Else: fat manual g/kg, carbs remainder
        # - If fat_mode Auto but no carb cap: fat fills after protein, carbs become remainder
        if goal_mode == "Weight gain" and carbs_g_per_kg_gain is not None:
            carbs_g = weight_for_macros * float(carbs_g_per_kg_gain)
            kcal_carbs = carbs_g * 4

            fat_kcal = target_kcal - (kcal_protein + kcal_carbs)
            fat_g = fat_kcal / 9 if fat_kcal > 0 else 0.0

            fat_min_g = 0.6 * weight_for_macros
            fat_max_g = 1.0 * weight_for_macros

            if fat_g < fat_min_g:
                fat_g = fat_min_g
                kcal_fat = fat_g * 9
                kcal_carbs = max(target_kcal - (kcal_protein + kcal_fat), 0.0)
                carbs_g = kcal_carbs / 4 if kcal_carbs > 0 else 0.0
            elif fat_g > fat_max_g:
                fat_g = fat_max_g
                kcal_fat = fat_g * 9
                kcal_carbs = max(target_kcal - (kcal_protein + kcal_fat), 0.0)
                carbs_g = kcal_carbs / 4 if kcal_carbs > 0 else 0.0
            else:
                kcal_fat = fat_g * 9

        else:
            if fat_mode == "Auto":
                fat_min_g = 0.3 * weight_for_macros
                fat_max_g = 1.5 * weight_for_macros
                remaining_kcal_after_protein = max(target_kcal - kcal_protein, 0.0)

                fat_g = min(max((remaining_kcal_after_protein * 0.35) / 9.0, fat_min_g), fat_max_g)
                kcal_fat = fat_g * 9
                kcal_carbs = max(target_kcal - (kcal_protein + kcal_fat), 0.0)
                carbs_g = kcal_carbs / 4 if kcal_carbs > 0 else 0.0

            else:
                fat_g = weight_for_macros * float(fat_g_per_kg_manual)
                kcal_fat = fat_g * 9
                kcal_carbs = max(target_kcal - (kcal_protein + kcal_fat), 0.0)
                carbs_g = kcal_carbs / 4 if kcal_carbs > 0 else 0.0

    if target_kcal > 0:
        protein_pct = kcal_protein / target_kcal * 100
        fat_pct = kcal_fat / target_kcal * 100
        carbs_pct = kcal_carbs / target_kcal * 100
    else:
        protein_pct = fat_pct = carbs_pct = 0.0

    return MacroResult(
        rmr=rmr,
        tdee=tdee,
        target_kcal=target_kcal,
        protein_g=protein_g,
        fat_g=fat_g,
        carbs_g=carbs_g,
        protein_pct=protein_pct,
        fat_pct=fat_pct,
        carbs_pct=carbs_pct,
    )


# ---------- AI MEAL PLAN GENERATION ----------
def build_mealplan_prompt(
    macros: MacroResult,
    goal_mode: GoalMode,
    using_glp1: bool,
    allergies: str,
    dislikes: str,
    preferred_store: str,
    weekly_budget: float,
    language: str,
    diet_pattern: str,
    fluid_limit_l,
    fast_food_chains,
    fast_food_percent: int,
    big_meals_per_day: int,
    snacks_per_day: int,
    prep_style: str,
    household_size: int,
    meal_prep_style: str,
    avg_prep_minutes: int,
    cooking_skill: str,
):
    if language == "Spanish":
        lang_note = (
            "REQUISITO DE IDIOMA (OBLIGATORIO):\n"
            "- TODO el texto debe estar en español.\n"
            "- Esto incluye encabezados, secciones, categorías de la lista del súper y avisos.\n"
            "- NO uses inglés, excepto nombres de marcas o medicamentos.\n"
            "- Usa SOLO guiones ASCII '-' para viñetas y rangos numéricos (por ejemplo: 500-1000).\n"
        )

        grocery_headers = """
Productos frescos:
- nombre del artículo — precio unitario — total de la línea

Proteínas:
- nombre del artículo — precio unitario — total de la línea

Lácteos:
- nombre del artículo — precio unitario — total de la línea

Granos / Almidones:
- nombre del artículo — precio unitario — total de la línea

Despensa:
- nombre del artículo — precio unitario — total de la línea

Congelados:
- nombre del artículo — precio unitario — total de la línea

Otros:
- nombre del artículo — precio unitario — total de la línea
""".strip()

        price_disclaimer = (
            "AVISO DE PRECIOS:\n"
            "Todos los precios son estimaciones únicamente y NO representan datos en tiempo real de tiendas. "
            "Los precios reales varían según la tienda y la región."
        )

        approx_rule_note = (
            '- La línea de macros debe iniciar exactamente con "Aprox:" o "Aproximado:" (en español) '
            'y debe ir en su propia línea.\n'
        )

        end_sections_header = f"""
DESPUÉS del Día 14, incluye SOLAMENTE estas 5 secciones (en este orden):

1) Resumen de objetivos diarios (persona principal) — UNA sola línea:
Resumen de objetivos diarios: <kcal> kcal, Proteína <g> g, Carbohidratos <g> g, Grasa <g> g.

2) Suplementos diarios (persona principal) — DEBE ser su propia sección:
Suplementos diarios:
- <suplemento>: <dosis> — <horario/instrucciones>
- <suplemento>: <dosis> — <horario/instrucciones>
(1 por línea; NO pongas suplementos en la misma línea que el resumen de macros.)

3) Resumen de costos (estimaciones aproximadas)
- Costo total de 14 días: $X
- Promedio por semana: $Y

4) Lista del súper (agrupada por categoría)

5) Instrucciones de cocina para algunas comidas principales
- Incluye instrucciones SOLO para las comidas principales más complejas que usaste (NO para cada comida).
- Apunta a 6-10 recetas en total.
- Para cada receta:
  Receta: <nombre tal como aparece en el plan>
  - Tiempo: ~X minutos (intenta respetar la guía de tiempo del usuario)
  - Porciones: suficiente para apoyar a la familia; aclara cómo porcionar para la persona principal
  - Ingredientes: lista corta
  - Pasos: 4-8 pasos cortos (muy fáciles si el nivel es principiante)

Esta lista del súper de 14 días está ajustada para alimentar aproximadamente a {household_size} persona(s).
""".strip()

    else:
        lang_note = (
            "IMPORTANT: Respond entirely in English. Use a clear, patient-friendly style. "
            "Use ONLY standard ASCII hyphens '-' for bullets and numeric ranges (e.g., 500-1000)."
        )

        grocery_headers = """
Produce:
- item name — unit price — line total

Protein:
- item name — unit price — line total

Dairy:
- item name — unit price — line total

Grains / Starches:
- item name — unit price — line total

Pantry:
- item name — unit price — line total

Frozen:
- item name — unit price — line total

Other:
- item name — unit price — line total
""".strip()

        price_disclaimer = (
            "PRICE DISCLAIMER:\n"
            "All prices are estimates only and NOT real-time retailer data. "
            "Actual prices vary by store and region."
        )

        approx_rule_note = (
            '- The "Approx:" line MUST start with exactly "Approx:" and must be on its own line.\n'
        )

        end_sections_header = f"""
AFTER Day 14, include ONLY these 5 sections (in this order):

1) Daily macro target summary (primary individual) — ONE line only:
Daily targets summary: <kcal> kcal, Protein <g> g, Carbs <g> g, Fat <g> g.

2) Daily supplements (primary individual) — MUST be its own section:
Daily supplements:
- <supplement>: <dose> — <timing/instructions>
- <supplement>: <dose> — <timing/instructions>
(1 per line; do NOT put supplements on the same line as the macro targets.)

3) Cost summary (rough estimates only)
- Total 14-day cost: $X
- Average per week: $Y

4) Grocery list (grouped by category)

5) Cooking instructions for selected main meals
- Include instructions ONLY for the more complex main meals you used (NOT for every meal).
- Aim for 6-10 recipes total.
- For each recipe:
  Recipe: <name as used in the plan>
  - Time: ~X minutes (try to respect the user's time guide)
  - Servings: enough to help feed the household; note how to portion for the primary individual
  - Ingredients: short list
  - Steps: 4-8 short steps (beginner-friendly if selected)

This 14-day grocery list is scaled to feed approximately {household_size} people.
""".strip()

    if goal_mode == "Maintenance":
        goal_note = """
GOAL MODE: MAINTENANCE
- The calorie target is meant to maintain current weight.
- Prioritize nutrient-dense meals to reduce risk of micronutrient deficiencies.
- Keep meals realistic for a busy person and do not introduce extreme restrictions unless a medical diet pattern is selected.
"""
    elif goal_mode == "Weight gain":
        goal_note = """
GOAL MODE: WEIGHT GAIN
- The calorie target already includes a surplus for weight gain.
- Prioritize lean mass support: adequate protein distribution across the day, sufficient carbohydrates for training performance, and fats for hormonal support.
- Avoid "dirty bulk" patterns (excess ultra-processed foods); keep choices mostly nutrient-dense.
"""
    else:
        goal_note = """
GOAL MODE: WEIGHT LOSS
- The calorie target already includes a deficit for weight loss.
- Keep meals nutrient-dense and high-protein to support satiety and preserve lean mass.
"""

    glp1_note = ""
    if using_glp1:
        glp1_note = """
GLP-1 RECEPTOR AGONIST-SPECIFIC CONSIDERATIONS:
The patient is actively using a GLP-1 receptor agonist (for diabetes and/or weight loss).

MACRONUTRIENT PRIORITY:
- Protein intake has been intentionally set higher to preserve lean mass during reduced caloric intake.
- Emphasize high-quality, leucine-rich protein sources distributed across the day.

SUPPLEMENTATION (INCLUDE IN THE END SECTIONS ONLY):
- Protein supplement: 20-40 g/serving (whey isolate or plant blend)
- Multivitamin once daily
- Vitamin B12 500-1,000 mcg daily (especially if also on metformin)
- Electrolytes/hydration support as needed
- Consider magnesium/fiber if constipation
"""

    clinical_note = ""
    if diet_pattern == "Cardiac (CHF / low sodium)":
        limit_txt = f"{fluid_limit_l:.1f} L/day" if fluid_limit_l else "1.5-2.0 L/day"
        clinical_note = f"""
CLINICAL DIET PATTERN: Cardiac diet for CHF with reduced ejection fraction.
- Sodium goal: generally < 2,000 mg/day.
- Emphasize high-fiber, low-sodium foods; minimize processed and canned foods.
- Avoid obviously salty foods (chips, fries, cured meats, canned soups, frozen dinners with high sodium).
- Fluid restriction: target total fluid intake of {limit_txt} per day (all beverages, soups, and liquid foods count).
- Include a simple suggested fluid schedule over the day.
"""
    elif diet_pattern == "Diabetic":
        clinical_note = """
CLINICAL DIET PATTERN: Diabetic diet for active diabetes.
- Emphasize consistent, moderate carbohydrate intake spread throughout the day.
- Prefer low-glycemic index carbohydrates (beans, lentils, whole grains, non-starchy vegetables).
- Avoid sugary drinks, juice, desserts; minimize refined carbs and added sugars.
- Pair carbohydrates with protein and/or fat to reduce postprandial glucose spikes.
"""
    elif diet_pattern == "Renal (ESRD / CKD 4-5)":
        clinical_note = """
CLINICAL DIET PATTERN: Renal diet for ESRD or CKD stage 4-5 (general guidance, not individualized).
- Limit sodium and highly processed foods.
- Avoid very high potassium foods in large amounts (bananas, oranges, potatoes, tomatoes, spinach, avocados, etc.).
- Limit high phosphorus foods (colas, many processed foods, some dairy, organ meats).
- Use moderate portions of protein; avoid extremely high-protein fad diets unless on dialysis and advised otherwise.
- Prefer lower-potassium fruits and vegetables and simple home-cooked meals over restaurant / fast-food when possible.
"""

    fast_food_note = ""
    if fast_food_chains and fast_food_percent > 0:
        chains_txt = ", ".join(fast_food_chains)
        fast_food_note = f"""
FAST-FOOD / TAKEOUT PATTERN (REAL MENU ITEMS ONLY):
- Patient is okay with using some meals from these fast-food chains: {chains_txt}.
- Aim for roughly {fast_food_percent}% of total weekly meals to be from fast-food or takeout.
- Use ONLY real menu items that actually exist or have existed on the standard menu at those chains.
- Prefer core, long-running menu items rather than limited-time specials to reduce error.
- For each fast-food meal, specify the restaurant and exact item name.
- For each item, provide approximate calories, protein, carbohydrates, fat, and sodium.
"""

    meal_timing_note = f"""
MEAL TIMING PREFERENCES:
- Target {big_meals_per_day} main meal(s) and {snacks_per_day} snack time(s) per day.
- Main meals should contain the majority of daily calories and protein.
- Snacks should be lighter and help fill in remaining macros without overshooting daily targets.
"""

    if prep_style == "Mostly premade / ready-to-eat from store":
        prep_note = """
COOKING VS PREMADE:
- Prioritize ready-to-eat or minimal-prep items (rotisserie chicken, pre-cooked grains, frozen vegetables, bagged salads).
- Avoid complicated recipes; most meals should be assembly or reheat rather than scratch cooking.
"""
    elif prep_style == "Mostly home-cooked meals":
        prep_note = """
COOKING VS PREMADE:
- Emphasize simple home-cooked meals using basic ingredients.
- Occasional premade or frozen items are okay, but most meals should involve basic cooking.
"""
    else:
        prep_note = """
COOKING VS PREMADE:
- Use a balanced mix of home-cooked meals, ready-to-eat items, and occasional fast-food/takeout.
- Reuse ingredients across meals to save time and reduce waste.
"""

    if meal_prep_style == "Bulk meal prep / repeat same meals for several days":
        variety_note = """
MEAL VARIETY VS BULK PREP:
- Repeat the SAME set of meals for 2-3 days in a row when practical.
- Prioritize meals that reheat well and can be cooked in large batches.
"""
    else:
        variety_note = """
MEAL VARIETY VS BULK PREP:
- Aim for reasonable variety across the 14 days, but you may still reuse some meals for practicality.
"""

    time_note = f"""
TIME AVAILABLE TO COOK (GUIDE, NOT STRICT):
- Average time available per MAIN meal: {int(avg_prep_minutes)} minutes.
- If time is 0-10 minutes, prioritize no-cook, microwave, air-fryer, rotisserie chicken, bagged salad kits, frozen steamable veggies, pre-cooked grains.
- Prefer recipes that reuse ingredients and avoid long simmer/bake times unless they can be batch-prepped quickly.
"""

    skill_note = f"""
COOKING SKILL LEVEL:
- Skill level: {cooking_skill}
- If Beginner: keep cooking methods very simple and include a short instruction section at the end ONLY for the more complex main meals you used.
"""

    household_note = ""
    portion_disclaimer = ""
    if household_size and household_size > 1:
        household_note = f"""
HOUSEHOLD / FAMILY MEAL SCALING:
- The calorie and macro targets apply ONLY to the primary individual.
- Meals and groceries should be planned to feed approximately {household_size} people.
- Grocery quantities and total cost MUST reflect feeding about {household_size} people.
- Macro estimates MUST reflect ONLY the primary individual's portion.
"""
        portion_disclaimer = f"""
IMPORTANT PORTION DISCLAIMER:
All calorie estimates, macro calculations, and portion recommendations in this plan apply ONLY to the primary individual.
Meals may be prepared in larger quantities to feed the household ({household_size} people), but macros apply only to the primary individual's portion.
"""

    two_week_budget = weekly_budget * 2.0
    pricing_note = f"""
PRICING AND GROCERY COST (ESTIMATES ONLY):
- All prices are approximate and must NOT use real-time data from any retailer or restaurant.
- Base grocery prices on typical U.S. supermarket averages.
- Weekly grocery budget is approximately ${weekly_budget:.2f} for the household (about {household_size} people).
- You are planning for 14 days (2 weeks), so try to keep the total 14-day food cost near ${two_week_budget:.2f}.
- For each grocery list item, include an estimated unit price and a line total.
- Provide an estimated total grocery cost for all 14 days and a rough per-week average cost.
"""

    non_negative_rule = """
MACRO VALIDITY RULE (MANDATORY):
- NO macro value may be negative. Never output negative grams or negative sodium.
- If a macro would be negative due to rounding, set it to 0 and rebalance FAT (and calories) so values remain realistic.
"""

    return f"""
{lang_note}

You are a registered dietitian and meal-planning assistant.

{goal_note}
{glp1_note}

MACRO TARGETS (PER DAY) FOR PRIMARY INDIVIDUAL:
- Daily calories: {macros.target_kcal:.0f} kcal
- Protein: {macros.protein_g:.0f} g/day
- Carbohydrates: {macros.carbs_g:.0f} g/day
- Fats: {macros.fat_g:.0f} g/day

PATIENT CONSTRAINTS:
- Allergies / must AVOID: {allergies or "none specified"}
- Foods to avoid / dislikes: {dislikes or "none specified"}
- Household size to feed with meals and groceries: {household_size}
- Weekly grocery budget: ${weekly_budget:.2f} (for the whole household)
- Preferred grocery store or market: {preferred_store or "generic US supermarket"}

{clinical_note}
{fast_food_note}
{meal_timing_note}
{prep_note}
{time_note}
{skill_note}
{variety_note}
{household_note}
{pricing_note}

{non_negative_rule}

MEAL PLAN TASK:
Create a 14-day meal plan for a single adult (the primary individual) based on the macro targets and constraints above.
Plan meals and grocery quantities so they can reasonably feed the whole household (about {household_size} people),
but keep all calorie and macro estimates focused on the primary individual's portion only.

STRUCTURE:
- For each day, include exactly {big_meals_per_day} main meal(s) and {snacks_per_day} snack time(s) per day.
- Label them clearly (for example: Breakfast, Lunch, Dinner, Snack 1, Snack 2).
- Distribute calories and macros so that totals for the day roughly match the macro targets for the primary individual.

Additional requirements:
- Keep recipes simple and realistic for a busy person.
- Reuse ingredients across meals to save cost and reduce waste.
- Keep daily totals reasonably close to the macro targets.
- Assume typical adult portion sizes; you may approximate macros.
- Respect the clinical diet pattern if one is specified.

{portion_disclaimer}

OUTPUT FORMAT (STRICT — MUST FOLLOW EXACTLY):
1) Use plain text only. No markdown tables.
2) DO NOT insert blank lines between items. (No empty lines inside a day.)
3) Each day must follow this exact pattern:

Day 1
- <Meal/ Snack label>: <meal description>
  Approx: <kcal> kcal, P: <g> g, C: <g> g, F: <g> g, Na: <mg> mg
- <Meal/ Snack label>: <meal description>
  Approx: <kcal> kcal, P: <g> g, C: <g> g, F: <g> g, Na: <mg> mg
(repeat until the day has exactly {big_meals_per_day + snacks_per_day} total items)

MANDATORY RULES FOR EVERY SINGLE ITEM:
- EVERY bullet line that begins with "- " MUST be followed immediately by an "Approx:" line on the NEXT line.
- This includes snacks, home-cooked meals, and ALL fast-food/takeout items.
{approx_rule_note}- Always include sodium as "Na: <mg> mg" (use a rough estimate). This is especially important for fast-food and cardiac diets.

FAST-FOOD NAMING RULE (when fast food is used):
- For fast-food meals, include the restaurant name in the description, e.g.:
  "- Lunch: Chick-fil-A Grilled Chicken Sandwich"
  "  Approx: ..."

NO EXTRA TEXT RULE:
- Do NOT add any commentary, tips, explanations, or extra headers inside days.
- Do NOT output a separate "Meal Plan (English)" header mid-document.
- Only the Day blocks + the required end sections.

{end_sections_header}

FORMAT RULES (MANDATORY):
- Category names MUST be plain text headers with NO leading dash.
- Grocery items MUST be bulleted with a leading "- ".
- Do NOT put a dash in front of category names.
- Do NOT use placeholders such as "...".
- Each category MUST contain at least one specific grocery item.

Now output the grocery list using EXACTLY this structure:

{grocery_headers}

{price_disclaimer}
""".strip()


def generate_meal_plan_with_ai(
    macros: MacroResult,
    goal_mode: GoalMode,
    using_glp1: bool,
    allergies: str,
    dislikes: str,
    preferred_store: str,
    weekly_budget: float,
    language: str,
    diet_pattern: str,
    fluid_limit_l,
    fast_food_chains,
    fast_food_percent: int,
    big_meals_per_day: int,
    snacks_per_day: int,
    prep_style: str,
    household_size: int,
    meal_prep_style: str,
    avg_prep_minutes: int,
    cooking_skill: str,
) -> str:
    prompt = build_mealplan_prompt(
        macros=macros,
        goal_mode=goal_mode,
        using_glp1=using_glp1,
        allergies=allergies,
        dislikes=dislikes,
        preferred_store=preferred_store,
        weekly_budget=weekly_budget,
        language=language,
        diet_pattern=diet_pattern,
        fluid_limit_l=fluid_limit_l,
        fast_food_chains=fast_food_chains,
        fast_food_percent=fast_food_percent,
        big_meals_per_day=big_meals_per_day,
        snacks_per_day=snacks_per_day,
        prep_style=prep_style,
        household_size=household_size,
        meal_prep_style=meal_prep_style,
        avg_prep_minutes=int(avg_prep_minutes),
        cooking_skill=str(cooking_skill),
    )

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise, practical meal-planning assistant for evidence-based weight management. "
                    "Use only standard ASCII hyphens '-' for bullets and numeric ranges."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )

    raw_text = completion.choices[0].message.content or ""
    return normalize_text_for_parsing(raw_text)


# ---------- PDF GENERATION (UNCHANGED) ----------
def create_pdf_from_text(text: str, title: str = "Meal Plan") -> bytes:
    """
    Your PDF function (unchanged).
    """
    from io import BytesIO
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase import pdfmetrics

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    page_width, page_height = letter

    left_margin = 40
    right_margin = 40
    top_margin = 40
    bottom_margin = 40

    usable_width = page_width - left_margin - right_margin

    title_font = "Helvetica-Bold"
    title_size = 14
    day_font = "Helvetica-Bold"
    day_size = 12
    table_header_font = "Helvetica-Bold"
    table_header_size = 9
    table_body_font = "Helvetica"
    table_body_size = 8
    body_font = "Helvetica"
    body_size = 10

    table_leading = 10
    body_leading = 14

    def wrap_text(text_line: str, font_name: str, font_size: int, max_width: float):
        words = text_line.split()
        if not words:
            return [""]
        lines = []
        current = words[0]
        for word in words[1:]:
            test = current + " " + word
            w = pdfmetrics.stringWidth(test, font_name, font_size)
            if w <= max_width:
                current = test
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return lines

    def parse_days_and_meals(text: str):
        lines = text.splitlines()
        used_indices = set()

        days = []
        current_day = None
        current_rows = []

        day_prefixes = ("Day ", "Día ", "Dia ")
        macros_prefixes = ("Approx:", "Aproximado:", "Aprox:")

        i = 0
        while i < len(lines):
            raw = lines[i]
            line = raw.strip()

            if any(line.startswith(p) for p in day_prefixes):
                if current_day is not None:
                    days.append({"day_title": current_day, "rows": current_rows})
                    current_rows = []
                current_day = line
                used_indices.add(i)
                i += 1
                continue

            if current_day is not None and line.startswith("- "):
                desc = line[2:].strip()

                macros_line = ""
                macros_idx = None
                j = i + 1
                while j < len(lines):
                    look = lines[j].strip()
                    if look.startswith("- ") or any(look.startswith(p) for p in day_prefixes):
                        break
                    if any(look.startswith(p) for p in macros_prefixes):
                        macros_line = look
                        macros_idx = j
                        break
                    j += 1

                if not macros_line:
                    i += 1
                    continue

                used_indices.add(i)
                if macros_idx is not None:
                    used_indices.add(macros_idx)

                meal_number = len(current_rows) + 1
                current_rows.append((f"Meal {meal_number}", desc, macros_line or ""))
                i += 1
                continue

            i += 1

        if current_day is not None:
            days.append({"day_title": current_day, "rows": current_rows})

        leftover_lines = [lines[idx] for idx in range(len(lines)) if idx not in used_indices]
        return days, leftover_lines

    def new_page_with_title():
        c.showPage()
        c.setFont(title_font, title_size)
        c.drawString(left_margin, page_height - top_margin, title)
        y = page_height - top_margin - 25
        return y

    c.setFont(title_font, title_size)
    c.drawString(left_margin, page_height - top_margin, title)
    current_y = page_height - top_margin - 25

    days, leftover_lines = parse_days_and_meals(text)

    col1_width = 70
    col3_width = 130
    col2_width = usable_width - col1_width - col3_width

    col1_x = left_margin
    col2_x = col1_x + col1_width
    col3_x = col2_x + col2_width

    for day in days:
        day_title = day["day_title"]
        rows = day["rows"]

        if not rows:
            if current_y <= bottom_margin + 40:
                current_y = new_page_with_title()
            c.setFont(day_font, day_size)
            c.drawString(left_margin, current_y, day_title)
            current_y -= 20
            continue

        if current_y <= bottom_margin + 60:
            current_y = new_page_with_title()

        c.setFont(day_font, day_size)
        c.drawString(left_margin, current_y, day_title)
        current_y -= 25

        header_height = table_leading + 4
        if current_y - header_height <= bottom_margin:
            current_y = new_page_with_title()
            c.setFont(day_font, day_size)
            c.drawString(left_margin, current_y, day_title + " (cont.)")
            current_y -= 25

        c.setFont(table_header_font, table_header_size)
        header_top_y = current_y
        header_bottom_y = current_y - header_height

        c.line(left_margin, header_top_y, page_width - right_margin, header_top_y)
        c.line(left_margin, header_bottom_y, page_width - right_margin, header_bottom_y)

        c.line(col1_x, header_top_y, col1_x, header_bottom_y)
        c.line(col2_x, header_top_y, col2_x, header_bottom_y)
        c.line(col3_x, header_top_y, col3_x, header_bottom_y)
        c.line(page_width - right_margin, header_top_y, page_width - right_margin, header_bottom_y)

        baseline_offset = (header_height - table_header_size) / 2
        text_y = header_top_y - baseline_offset - 5
        c.drawString(col1_x + 2, text_y, "Meal #")
        c.drawString(col2_x + 2, text_y, "Meal description")
        c.drawString(col3_x + 2, text_y, "Approx kcal & macros")

        current_y = header_bottom_y

        c.setFont(table_body_font, table_body_size)
        for meal_no, desc, macros_line in rows:
            meal_lines = wrap_text(meal_no, table_body_font, table_body_size, col1_width - 4)
            desc_lines = wrap_text(desc, table_body_font, table_body_size, col2_width - 4)
            macros_lines = wrap_text(macros_line, table_body_font, table_body_size, col3_width - 4)

            num_lines = max(len(meal_lines), len(desc_lines), len(macros_lines))
            row_height = num_lines * table_leading + 4

            if current_y - row_height <= bottom_margin:
                current_y = new_page_with_title()
                c.setFont(day_font, day_size)
                c.drawString(left_margin, current_y, f"{day_title} (cont.)")
                current_y -= 25

                header_height = table_leading + 4
                c.setFont(table_header_font, table_header_size)
                header_top_y = current_y
                header_bottom_y = current_y - header_height

                c.line(left_margin, header_top_y, page_width - right_margin, header_top_y)
                c.line(left_margin, header_bottom_y, page_width - right_margin, header_bottom_y)
                c.line(col1_x, header_top_y, col1_x, header_bottom_y)
                c.line(col2_x, header_top_y, col2_x, header_bottom_y)
                c.line(col3_x, header_top_y, col3_x, header_bottom_y)
                c.line(page_width - right_margin, header_top_y, page_width - right_margin, header_bottom_y)

                baseline_offset = (header_height - table_header_size) / 2
                text_y = header_top_y - baseline_offset - 5
                c.drawString(col1_x + 2, text_y, "Meal #")
                c.drawString(col2_x + 2, text_y, "Meal description")
                c.drawString(col3_x + 2, text_y, "Approx kcal & macros")

                current_y = header_bottom_y
                c.setFont(table_body_font, table_body_size)

            row_top_y = current_y
            row_bottom_y = current_y - row_height

            c.line(left_margin, row_top_y, page_width - right_margin, row_top_y)
            c.line(left_margin, row_bottom_y, page_width - right_margin, row_bottom_y)
            c.line(col1_x, row_top_y, col1_x, row_bottom_y)
            c.line(col2_x, row_top_y, col2_x, row_bottom_y)
            c.line(col3_x, row_top_y, col3_x, row_bottom_y)
            c.line(page_width - right_margin, row_top_y, page_width - right_margin, row_bottom_y)

            row_text_y = row_top_y - 2 - table_leading
            for idx in range(num_lines):
                if idx < len(meal_lines):
                    c.drawString(col1_x + 2, row_text_y, meal_lines[idx])
                if idx < len(desc_lines):
                    c.drawString(col2_x + 2, row_text_y, desc_lines[idx])
                if idx < len(macros_lines):
                    c.drawString(col3_x + 2, row_text_y, macros_lines[idx])
                row_text_y -= table_leading

            current_y = row_bottom_y

        current_y -= 18

    if leftover_lines:
        if current_y <= bottom_margin + 40:
            current_y = new_page_with_title()

        c.setFont(body_font, body_size)
        text_obj = c.beginText()
        text_obj.setTextOrigin(left_margin, current_y)
        text_obj.setFont(body_font, body_size)
        text_obj.setLeading(body_leading)

        for raw_line in leftover_lines:
            wrapped_lines = wrap_text(raw_line, body_font, body_size, usable_width)
            for line in wrapped_lines:
                if text_obj.getY() <= bottom_margin:
                    c.drawText(text_obj)
                    current_y = new_page_with_title()
                    text_obj = c.beginText()
                    text_obj.setTextOrigin(left_margin, current_y)
                    text_obj.setFont(body_font, body_size)
                    text_obj.setLeading(body_leading)
                text_obj.textLine(line)

        c.drawText(text_obj)
        current_y = text_obj.getY()

    if "Spanish" in title or "Español" in title:
        disclaimer_text = (
            "DESCARGO DE RESPONSABILIDAD:\n"
            "Este plan de alimentación es sólo para fines educativos y no constituye asesoramiento médico ni nutricional. "
            "El autor no es un dietista registrado ni un profesional de la nutrición autorizado. "
            "Las estimaciones de calorías, los cálculos de macronutrientes y las sugerencias de compras pueden ser "
            "inexactas o no apropiadas para personas con condiciones médicas específicas. "
            "Los pacientes deben consultar con un proveedor de atención médica autorizado o con un dietista registrado "
            "para recibir recomendaciones médicas o nutricionales personalizadas. "
            "Si tiene dudas sobre restricciones dietéticas, enfermedades crónicas, alergias, control de peso "
            "o necesidades nutricionales, hable con un dietista registrado."
        )
    else:
        disclaimer_text = (
            "DISCLAIMER:\n"
            "This meal plan is for educational purposes only and does not constitute medical or nutritional advice. "
            "The author is not a registered dietitian or licensed nutrition professional. "
            "Calorie estimates, macro calculations, and grocery suggestions may be inaccurate or inappropriate "
            "for individuals with specific medical conditions. "
            "Patients should consult with a licensed healthcare provider or registered dietitian for personalized "
            "medical or nutritional guidance. "
            "If you have concerns about dietary restrictions, chronic illness, allergies, weight management, "
            "or nutritional needs, please speak with a registered dietitian."
        )

    current_y -= 20
    if current_y <= bottom_margin + 40:
        current_y = new_page_with_title()

    c.setFont(body_font, body_size)
    text_obj = c.beginText()
    text_obj.setTextOrigin(left_margin, current_y)
    text_obj.setFont(body_font, body_size)
    text_obj.setLeading(body_leading)

    for line in disclaimer_text.split("\n"):
        wrapped = wrap_text(line, body_font, body_size, usable_width)
        for w in wrapped:
            if text_obj.getY() <= bottom_margin:
                c.drawText(text_obj)
                current_y = new_page_with_title()
                text_obj = c.beginText()
                text_obj.setTextOrigin(left_margin, current_y)
                text_obj.setFont(body_font, body_size)
                text_obj.setLeading(body_leading)
            text_obj.textLine(w)

    c.drawText(text_obj)

    c.save()
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


# ---------- STREAMLIT UI ----------
def main():
    st.set_page_config(page_title="Evidence-Based Macro & Meal Planner", layout="centered")

    st.title("Personalized Meal Planning for the busy clinician")
    st.write("Enter metrics and personal preferences below")

    # Initialize session state
    if "protein_g_per_kg" not in st.session_state:
        st.session_state["protein_g_per_kg"] = 1.4
    if "fat_g_per_kg" not in st.session_state:
        st.session_state["fat_g_per_kg"] = 0.7
    if "using_glp1" not in st.session_state:
        st.session_state["using_glp1"] = False

    # Defaults for carb-cap + fat mode
    if "enable_carb_cap" not in st.session_state:
        st.session_state["enable_carb_cap"] = False
    if "carbs_g_per_kg_cap" not in st.session_state:
        st.session_state["carbs_g_per_kg_cap"] = 1.2
    if "carb_cap_basis" not in st.session_state:
        st.session_state["carb_cap_basis"] = "Current"
    if "fat_mode" not in st.session_state:
        st.session_state["fat_mode"] = "Manual"

    # NEW: diabetic guidance state
    if "diabetic_carb_mode" not in st.session_state:
        st.session_state["diabetic_carb_mode"] = True
    if "a1c_band" not in st.session_state:
        st.session_state["a1c_band"] = "T2DM near goal (A1C 6.5-6.9%)"

    # A1C-based guidance bands (cutpoints + internal guidance buckets)
    IR_BANDS = {
        "Normal / no IR (A1C < 5.7%)": {"cap": 3.0, "default": 2.5, "min": 2.0, "max": 3.0},
        "Prediabetes / mild IR (A1C 5.7-6.4%)": {"cap": 2.5, "default": 2.0, "min": 1.5, "max": 2.5},
        "T2DM near goal (A1C 6.5-6.9%)": {"cap": 2.0, "default": 1.5, "min": 1.0, "max": 2.0},
        "T2DM above goal (A1C 7.0-8.4%)": {"cap": 1.5, "default": 1.25, "min": 0.8, "max": 1.5},
        "T2DM very high (A1C >= 8.5%)": {"cap": 1.0, "default": 0.9, "min": 0.5, "max": 1.0},
    }

    # 0) MODE SELECTOR
    goal_mode: GoalMode = st.selectbox(
        "Mode",
        options=["Weight loss", "Maintenance", "Weight gain"],
        index=0,
        help="Select weight loss, maintenance, or weight gain."
    )

    # 1) Patient / User Info
    st.subheader("1. Patient / User Info")
    col1, col2 = st.columns(2)

    with col1:
        sex = st.selectbox("Sex", options=["M", "F"])
        age = st.number_input("Age (years)", min_value=12, max_value=100, value=30)

        height_unit = st.radio("Height units", options=["cm", "ft/in"], index=1, horizontal=True)
        if height_unit == "cm":
            height_cm = st.number_input("Height (cm)", min_value=120.0, max_value=230.0, value=170.0)
        else:
            height_ft = st.number_input("Height (feet)", min_value=3, max_value=7, value=5)
            height_in = st.number_input("Height (inches)", min_value=0, max_value=11, value=6)
            height_cm = height_ft * 30.48 + height_in * 2.54
            st.caption(f"Calculated height: {height_cm:.1f} cm")

    with col2:
        weight_unit = st.radio("Weight units", options=["kg", "lbs"], index=1, horizontal=True)
        if weight_unit == "kg":
            weight_current_kg = st.number_input("Current weight (kg)", min_value=30.0, max_value=300.0, value=70.0)
            weight_goal_kg = st.number_input("Goal weight (kg)", min_value=30.0, max_value=300.0, value=65.0)
        else:
            weight_current_lbs = st.number_input("Current weight (lbs)", min_value=60.0, max_value=660.0, value=154.0)
            weight_goal_lbs = st.number_input("Goal weight (lbs)", min_value=60.0, max_value=660.0, value=143.0)
            weight_current_kg = weight_current_lbs / 2.20462
            weight_goal_kg = weight_goal_lbs / 2.20462
            st.caption(f"Current weight: {weight_current_kg:.1f} kg\nGoal weight: {weight_goal_kg:.1f} kg")

        weight_source = st.selectbox("Weight used for macros", options=["Current", "Goal"])

    # 2) Activity & Maintenance Settings
    st.subheader("2. Activity & Maintenance Settings")

    colA, colB = st.columns([10, 1])

    with colA:
        activity_factor = st.number_input(
            "Activity factor",
            min_value=1.1,
            max_value=2.5,
            value=1.4,
            step=0.025,
            help="Click the question mark for a detailed activity reference table."
        )

    with colB:
        with st.popover("❓"):
            activity_factor_reference_table()

    use_estimated_maintenance = st.selectbox(
        "Use estimated maintenance (TDEE)?",
        options=["Yes", "No"],
        index=0,
        help="If No, enter a known maintenance calorie value."
    ) == "Yes"

    maintenance_kcal_known = None
    if not use_estimated_maintenance:
        maintenance_kcal_known = st.number_input(
            "Known maintenance calories (kcal/day)",
            min_value=800.0,
            max_value=6000.0,
            value=2600.0,
            step=50.0,
        )

    intensity: Optional[Intensity] = None
    if goal_mode == "Weight loss":
        intensity = st.selectbox(
            "Weight-loss intensity",
            options=["Gentle", "Moderate", "Aggressive"],
            help="Gentle ≈250 kcal/day deficit, Moderate ≈500, Aggressive ≈750."
        )

    surplus_kcal = 300.0
    if goal_mode == "Weight gain":
        st.subheader("2b. Weight gain settings")
        surplus_kcal = st.number_input(
            "Daily calorie surplus (kcal/day)",
            min_value=100.0,
            max_value=900.0,
            value=300.0,
            step=50.0,
            help="Typical evidence-based surplus: +250–500 kcal/day."
        )
        est_gain_kg_per_week = (surplus_kcal * 7) / 7700.0
        st.caption(f"Estimated gain from surplus: ~{est_gain_kg_per_week:.2f} kg/week (rough estimate).")
        min_gain = float(weight_current_kg) * 0.0025
        max_gain = float(weight_current_kg) * 0.005
        st.caption(f"Recommended gain range: ~{min_gain:.2f} to {max_gain:.2f} kg/week (0.25–0.5% BW/week).")

    # 5) Clinical diet pattern
    st.subheader("5. Clinical diet pattern (optional)")
    diet_pattern = st.selectbox(
        "Apply a medical diet template",
        options=["None", "Cardiac (CHF / low sodium)", "Diabetic", "Renal (ESRD / CKD 4-5)"],
        help="Adds extra constraints to the meal plan. If you enable carb priority, it can also change macro math."
    )

    fluid_limit_l = None
    if "Cardiac" in diet_pattern:
        fluid_limit_l = st.number_input(
            "Daily fluid limit (liters)",
            min_value=0.5,
            max_value=4.0,
            value=1.5,
            step=0.25,
            help="Typical CHF fluid restriction is around 1.5–2.0 L/day; adjust per patient."
        )

    # Diabetic guidance panel (A1C bands)
    if diet_pattern == "Diabetic":
        st.markdown("### Diabetic carb guidance (A1C-based, optional)")

        diabetic_carb_mode = st.toggle(
            "Use A1C-based carb guidance (suggests carb range + recommended cap)",
            value=bool(st.session_state["diabetic_carb_mode"]),
            help="Uses standard A1C cut points plus internal 'control buckets' to suggest a carb g/kg range/cap. You can still override."
        )
        st.session_state["diabetic_carb_mode"] = bool(diabetic_carb_mode)

        if diabetic_carb_mode:
            a1c_band = st.selectbox(
                "A1C band (guides carb g/kg range)",
                options=list(IR_BANDS.keys()),
                index=list(IR_BANDS.keys()).index(st.session_state["a1c_band"])
                if st.session_state["a1c_band"] in IR_BANDS else 2,
                help="Normal <5.7, Prediabetes 5.7-6.4, Diabetes >=6.5. Within diabetes, these are guidance buckets."
            )
            st.session_state["a1c_band"] = a1c_band

            # Helpful defaults when diabetic guidance is on (but user can toggle back)
            if st.session_state.get("enable_carb_cap") is False:
                st.session_state["enable_carb_cap"] = True
            if st.session_state.get("fat_mode") != "Auto":
                st.session_state["fat_mode"] = "Auto"

    glp1_help = (
        "Check this if the patient is using a GLP-1 receptor agonist.\n\n"
        "Examples:\n"
        "- Semaglutide (Ozempic®, Wegovy®, Rybelsus®)\n"
        "- Tirzepatide (Mounjaro®, Zepbound®)\n"
        "- Liraglutide (Victoza®, Saxenda®)\n"
        "- Dulaglutide (Trulicity®)\n"
        "- Exenatide (Byetta®, Bydureon®)\n"
        "- Lixisenatide (Adlyxin®)\n"
    )

    using_glp1 = st.checkbox(
        "Using a GLP-1 receptor agonist (GLP-1RA)",
        key="using_glp1",
        help=glp1_help
    )

    if using_glp1:
        if st.session_state["protein_g_per_kg"] < 1.6:
            st.session_state["protein_g_per_kg"] = 1.6

    # 3) Macro Settings
    st.subheader("3. Macro Settings (priority mode + toggles)")

    if using_glp1:
        protein_min = 1.6
        protein_max = 2.0
        protein_help = "GLP-1RA selected: protein min 1.6 g/kg, adjustable up to 2.0 g/kg."
    else:
        protein_min = 0.8
        protein_max = 2.5
        if goal_mode == "Weight gain":
            protein_help = "Weight gain: typical evidence-based range 1.6–2.2 g/kg/day."
        elif goal_mode == "Maintenance":
            protein_help = "Maintenance: choose a reasonable protein target; adjust for training."
        else:
            protein_help = "Weight loss: common evidence-supported range 1.2–1.6 g/kg (higher can be ok)."

    # Carb cap UI (shows for all modes; A1C bands drive default ranges when diabetic guidance is on)
    carb_col, carb_q = st.columns([10, 1])
    with carb_col:
        enable_carb_cap = st.checkbox(
            "Enable carbohydrate target/cap (g/kg/day) and prioritize carbs → protein → fat remainder",
            value=bool(st.session_state["enable_carb_cap"]),
            help="If enabled, carbs are set first (g/kg cap), protein is set next, and fat becomes the remainder."
        )
        st.session_state["enable_carb_cap"] = bool(enable_carb_cap)

    with carb_q:
        with st.popover("❓"):
            insulin_resistance_reference()

    carbs_g_per_kg_cap: Optional[float] = None
    carb_cap_basis = st.session_state.get("carb_cap_basis", "Current")

    if enable_carb_cap:
        carb_cap_basis = st.selectbox(
            "Carb cap is based on which body weight?",
            options=["Current", "Macro weight"],
            index=0 if st.session_state.get("carb_cap_basis", "Current") == "Current" else 1,
            help="Many clinicians prefer carb caps based on CURRENT weight. Macro weight uses your selected macro weight source."
        )
        st.session_state["carb_cap_basis"] = carb_cap_basis

        recommended_cap = None
        if diet_pattern == "Diabetic" and bool(st.session_state.get("diabetic_carb_mode", True)):
            band = st.session_state.get("a1c_band", "T2DM near goal (A1C 6.5-6.9%)")
            band_cfg = IR_BANDS.get(band, {"min": 1.0, "max": 2.0, "default": 1.5, "cap": 2.0})
            cmin, cmax, cdef = float(band_cfg["min"]), float(band_cfg["max"]), float(band_cfg["default"])
            recommended_cap = float(band_cfg["cap"])

            st.caption(f"A1C band selected: **{band}**. Recommended carb cap: **{recommended_cap:.1f} g/kg/day**.")

            carbs_g_per_kg_cap = st.slider(
                "Carbohydrate cap (g/kg/day)",
                min_value=float(cmin),
                max_value=float(cmax),
                value=float(st.session_state.get("carbs_g_per_kg_cap", cdef)),
                step=0.1,
                help="Carbs are set first to this cap (based on selected weight basis)."
            )
        else:
            ir_level = st.selectbox(
                "Insulin resistance level (guides carb g/kg range)",
                options=["Low / none", "Moderate", "High", "Very high / uncontrolled"],
                index=1,
                help="Sets a sensible g/kg range; you can still choose the exact cap."
            )

            if ir_level == "Low / none":
                cmin, cmax, cdef = 1.5, 2.5, 2.0
            elif ir_level == "Moderate":
                cmin, cmax, cdef = 1.0, 1.8, 1.4
            elif ir_level == "High":
                cmin, cmax, cdef = 0.8, 1.2, 1.0
            else:
                cmin, cmax, cdef = 0.5, 1.0, 0.8

            if diet_pattern != "Diabetic":
                cmin = max(0.3, cmin)
                cmax = max(cmax, 2.5)

            carbs_g_per_kg_cap = st.slider(
                "Carbohydrate cap (g/kg/day)",
                min_value=float(cmin),
                max_value=float(cmax),
                value=float(st.session_state.get("carbs_g_per_kg_cap", cdef)),
                step=0.1,
                help="Carbs are set first to this cap (based on selected weight basis)."
            )

        st.session_state["carbs_g_per_kg_cap"] = float(carbs_g_per_kg_cap) if carbs_g_per_kg_cap is not None else st.session_state["carbs_g_per_kg_cap"]

        if recommended_cap is not None and carbs_g_per_kg_cap is not None and float(carbs_g_per_kg_cap) > float(recommended_cap):
            st.warning(f"Selected carbs exceed the recommended cap for this A1C band ({recommended_cap:.1f} g/kg/day).")

    # Fat mode toggle
    fat_mode = st.selectbox(
        "Fat setting",
        options=["Auto", "Manual"],
        index=0 if st.session_state.get("fat_mode", "Manual") == "Auto" else 1,
        help="Auto = fat becomes the remainder after carbs cap + protein. Manual = you set fat g/kg and carbs are remainder."
    )
    st.session_state["fat_mode"] = fat_mode

    # Protein + manual fat side-by-side
    col3, col4 = st.columns(2)
    with col3:
        protein_g_per_kg = st.number_input(
            "Protein (g/kg for macro weight)",
            min_value=float(protein_min),
            max_value=float(protein_max),
            value=float(st.session_state["protein_g_per_kg"]),
            step=0.1,
            help=protein_help,
        )
        st.session_state["protein_g_per_kg"] = float(protein_g_per_kg)

    fat_g_per_kg_manual = float(st.session_state.get("fat_g_per_kg", 0.7))
    with col4:
        if fat_mode == "Manual":
            fat_help = "Manual fat g/kg. Carbs become the remainder to hit calories."
            fat_g_per_kg_manual = st.number_input(
                "Fat (g/kg for macro weight)",
                min_value=0.3,
                max_value=1.5,
                value=float(st.session_state.get("fat_g_per_kg", 0.7)),
                step=0.1,
                help=fat_help,
            )
            st.session_state["fat_g_per_kg"] = float(fat_g_per_kg_manual)
        else:
            st.info("Fat is set automatically as the remainder after carbs + protein (with safety clamps).")

    # Weight gain carb target (still used; in Option A, it becomes a desired target but is capped)
    carbs_g_per_kg_gain: Optional[float] = None
    if goal_mode == "Weight gain":
        st.subheader("3b. Weight gain carb target (training volume)")
        training_volume: TrainingVolume = st.selectbox(
            "Training volume (for desired carb target)",
            options=["Moderate (3–4 days/week)", "High volume (5–6 days/week)"],
            index=0,
            help="Sets a DESIRED carbohydrate g/kg target to support training performance."
        )

        if training_volume.startswith("Moderate"):
            default_carbs_gkg = 3.5
            carbs_min, carbs_max = 3.0, 4.0
        else:
            default_carbs_gkg = 5.0
            carbs_min, carbs_max = 4.0, 6.0

        carbs_g_per_kg_gain = st.slider(
            "Desired carbohydrates (g/kg/day) for weight gain",
            min_value=float(carbs_min),
            max_value=float(carbs_max),
            value=float(default_carbs_gkg),
            step=0.1,
            help="If carb-cap priority mode is enabled, this becomes the desired target but will be capped."
        )

        if enable_carb_cap and carbs_g_per_kg_cap is not None:
            st.caption(
                "Option A active: In weight gain, carbs are set to MIN(desired gain carbs, carb cap). "
                "Fat becomes the remainder (with clamps), so weight gain can still happen even with capped carbs."
            )
        else:
            st.caption(
                "In weight gain without carb-cap priority, carbs are targeted first (desired g/kg) and fat becomes the remainder, "
                "clamped to 0.6–1.0 g/kg."
            )

    # 4) Preferences for AI Meal Plan
    st.subheader("4. Preferences for AI Meal Plan")

    allergies = st.text_input("Allergies (comma-separated)", placeholder="e.g., peanuts, shellfish")
    dislikes = st.text_input("Foods to avoid / dislikes", placeholder="e.g., mushrooms, cilantro")
    preferred_store = st.text_input("Preferred market/store", placeholder="e.g., H-E-B, Costco, Walmart")
    weekly_budget = st.number_input(
        "Weekly grocery budget (USD) for the household",
        min_value=10.0,
        max_value=1000.0,
        value=120.0,
        step=10.0,
    )

    language = st.selectbox(
        "Meal plan language",
        options=["English", "Spanish"],
        help="Choose the language for the generated meal plan."
    )

    st.info(
        "💲 **Price Disclaimer:** All grocery prices in the generated meal plan are estimates only.\n"
        "They do not reflect real-time pricing from Walmart, H-E-B, Costco, or any other retailer.\n"
        "Real-time pricing will be added once API access is approved."
    )

    # 6) Fast-food options
    st.subheader("6. Fast-food options (optional)")

    include_fast_food = st.checkbox(
        "Allow some meals from fast-food restaurants",
        value=False,
        help="The plan will still try to hit macros and any medical diet constraints."
    )

    fast_food_chains = []
    if include_fast_food:
        fast_food_chains = st.multiselect(
            "Allowed fast-food chains",
            options=[
                "McDonald's", "Chick-fil-A", "Taco Bell", "Subway", "Chipotle", "Wendy's", "Burger King",
                "Panera Bread", "Starbucks", "Five Guys", "Whataburger", "In-N-Out Burger", "Jack in the Box",
                "Sonic Drive-In", "Carl's Jr.", "Hardee's", "Culver's", "Smashburger", "Rally's / Checkers",
                "Freddy's Frozen Custard & Steakburgers", "Steak 'n Shake", "KFC", "Popeyes", "Raising Cane's",
                "Church's Chicken", "Zaxby's", "Wingstop", "Bojangles", "El Pollo Loco", "Domino's", "Pizza Hut",
                "Papa John's", "Little Caesars", "Marco's Pizza", "Papa Murphy's", "Blaze Pizza", "MOD Pizza",
                "Jimmy John's", "Jersey Mike's", "Firehouse Subs", "Which Wich", "Potbelly", "Qdoba", "Del Taco",
                "Taco Cabana", "Moe's Southwest Grill", "Baja Fresh", "Dunkin'", "Tim Hortons",
                "Einstein Bros Bagels", "Krispy Kreme", "Dutch Bros Coffee", "Long John Silver's", "Captain D's",
                "Wienerschnitzel", "Nathan's Famous", "Dairy Queen", "Baskin Robbins", "Cold Stone Creamery",
                "Ben & Jerry's", "Panda Express", "Arby's", "Shake Shack", "Noodles & Company", "Jollibee",
            ],
            help="The AI can substitute some meals with items from these places."
        )

    fast_food_percent = 0
    if include_fast_food:
        fast_food_percent = st.slider(
            "About what % of weekly meals can be fast-food / takeout?",
            min_value=10,
            max_value=80,
            step=10,
            value=20,
            help="This is a rough target; the plan will approximate this share of meals."
        )

    # 7) Meal timing
    st.subheader("7. Meal timing (optional)")
    big_meals_per_day = st.selectbox("Number of main meals per day", options=[1, 2, 3], index=2)
    snacks_per_day = st.selectbox("Number of snack times per day", options=[0, 1, 2, 3], index=2)

    # 8) Cooking vs premade balance
    st.subheader("8. Cooking vs premade balance")
    prep_style = st.selectbox(
        "Preferred style",
        options=[
            "Balanced: mix of cooking and premade",
            "Mostly premade / ready-to-eat from store",
            "Mostly home-cooked meals",
        ],
    )

    # 8b) Time available + skill level
    st.subheader("8b. Time available to meal prep (optional)")
    avg_prep_minutes = st.number_input(
        "Average time you can spend cooking/prepping per main meal (minutes)",
        min_value=0,
        max_value=120,
        value=20,
        step=10,
        help="Use 0 for 'no-cook / microwave only'. This is a guide, not a strict limit."
    )

    cooking_skill = st.selectbox(
        "Cooking skill level",
        options=["Beginner (needs instructions)", "Intermediate", "Advanced"],
        index=0,
        help="Beginner will trigger simple instructions for only the more complex meals at the end."
    )

    # 9) Household / family planning
    st.subheader("9. Household / family planning")
    household_size = st.number_input(
        "How many people will be eating most of these meals?",
        min_value=1, max_value=10, value=1, step=1,
        help="Macros/calorie targets remain for ONE primary individual; quantities scaled for household."
    )

    # 10) Variety vs meal prep style
    st.subheader("10. Variety vs meal prep style")
    meal_prep_style = st.selectbox(
        "How should the plan handle variety?",
        options=["Varied meals each day", "Bulk meal prep / repeat same meals for several days"],
        index=0,
    )

    submitted = st.button("Calculate macros and generate meal plan")

    if submitted:
        macros = calculate_macros(
            sex=sex,
            age=int(age),
            height_cm=float(height_cm),
            weight_current_kg=float(weight_current_kg),
            weight_goal_kg=float(weight_goal_kg),
            weight_source=weight_source,
            activity_factor=float(activity_factor),
            goal_mode=goal_mode,
            intensity=intensity,
            use_estimated_maintenance=use_estimated_maintenance,
            maintenance_kcal_known=maintenance_kcal_known,
            surplus_kcal=surplus_kcal if goal_mode == "Weight gain" else 0.0,

            protein_g_per_kg=float(st.session_state["protein_g_per_kg"]),

            carbs_g_per_kg_cap=float(carbs_g_per_kg_cap) if (enable_carb_cap and carbs_g_per_kg_cap is not None) else None,
            carb_cap_basis=carb_cap_basis,
            fat_mode="Auto" if fat_mode == "Auto" else "Manual",
            fat_g_per_kg_manual=float(fat_g_per_kg_manual),

            carbs_g_per_kg_gain=carbs_g_per_kg_gain,
        )

        st.success("Calculated macros successfully.")

        st.subheader("Calculated Energy and Macros")
        colA, colB = st.columns(2)
        with colA:
            st.metric("Resting Metabolic Rate (RMR)", f"{macros.rmr:.0f} kcal/day")
            st.metric("TDEE (estimated)", f"{macros.tdee:.0f} kcal/day")
            st.metric("Target calories", f"{macros.target_kcal:.0f} kcal/day")
        with colB:
            st.write("Daily macros")
            st.write(f"- Protein: {macros.protein_g:.0f} g ({macros.protein_pct:.1f}% kcal)")
            st.write(f"- Fat: {macros.fat_g:.0f} g ({macros.fat_pct:.1f}% kcal)")
            st.write(f"- Carbs: {macros.carbs_g:.0f} g ({macros.carbs_pct:.1f}% kcal)")

        st.divider()
        st.subheader("AI-Generated 14-Day Meal Plan")

        with st.spinner("Calling AI to generate your meal plan..."):
            try:
                plan_text = generate_meal_plan_with_ai(
                    macros=macros,
                    goal_mode=goal_mode,
                    using_glp1=using_glp1,
                    allergies=allergies,
                    dislikes=dislikes,
                    preferred_store=preferred_store,
                    weekly_budget=weekly_budget,
                    language=language,
                    diet_pattern=diet_pattern,
                    fluid_limit_l=fluid_limit_l,
                    fast_food_chains=fast_food_chains,
                    fast_food_percent=fast_food_percent,
                    big_meals_per_day=big_meals_per_day,
                    snacks_per_day=snacks_per_day,
                    prep_style=prep_style,
                    household_size=household_size,
                    meal_prep_style=meal_prep_style,
                    avg_prep_minutes=int(avg_prep_minutes),
                    cooking_skill=str(cooking_skill),
                )

                clean_text = normalize_text_for_parsing(plan_text)
                clean_text = add_section_spacing(clean_text)
                clean_text = add_recipe_spacing_and_dividers(clean_text, divider_len=48)
                clean_text = format_end_sections(clean_text)

                # FIX: clamp negatives + rebalance fat (and kcal) on every Approx line
                clean_text = sanitize_and_rebalance_macro_lines(clean_text)

                st.session_state["plan_text"] = clean_text
                st.session_state["plan_language"] = language
            except Exception as e:
                st.error(f"Error generating meal plan: {e}")

    plan_text = st.session_state.get("plan_text", "")
    plan_language = st.session_state.get("plan_language", "English")

    if plan_text:
        st.subheader("Current meal plan")
        st.text_area("Meal plan", plan_text, height=500)

        pdf_bytes = create_pdf_from_text(
            plan_text,
            title="Meal Plan (Spanish)" if plan_language == "Spanish" else "Meal Plan (English)",
        )

        st.download_button(
            label="Download meal plan as PDF",
            data=pdf_bytes,
            file_name="meal_plan.pdf",
            mime="application/pdf",
        )


if __name__ == "__main__":
    main()
