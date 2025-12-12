import os
from dataclasses import dataclass
from typing import Literal
from io import BytesIO

import streamlit as st
from openai import OpenAI
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics

MODEL_NAME = "gpt-4.1-mini"

# ---------- OPENAI CLIENT SETUP ----------

# Prefer Streamlit secrets in the cloud, fall back to local env
api_key = None

# Try Streamlit Cloud secrets
try:
    api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found in Streamlit secrets or environment variables")

client = OpenAI(api_key=api_key)


# ---------- DATA STRUCTURES ----------

Intensity = Literal["Gentle", "Moderate", "Aggressive"]


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
    intensity: Intensity,
    protein_g_per_kg: float = 1.4,
    fat_g_per_kg: float = 0.7,
) -> MacroResult:
    """Calculate RMR, TDEE, target calories and macros using Mifflin-St Jeor."""

    # 1) RMR using current weight
    if sex.upper() == "M":
        rmr = 10 * weight_current_kg + 6.25 * height_cm - 5 * age + 5
    else:
        rmr = 10 * weight_current_kg + 6.25 * height_cm - 5 * age - 161

    # 2) TDEE
    tdee = rmr * activity_factor

    # 3) Weight-loss target calories
    deficit_map = {
        "Gentle": 250,
        "Moderate": 500,
        "Aggressive": 750,
    }
    deficit = deficit_map[intensity]
    target_kcal = max(tdee - deficit, 1200)  # safety floor

    # 4) Macros based on chosen weight
    if weight_source == "Current":
        weight_for_macros = weight_current_kg
    else:
        weight_for_macros = weight_goal_kg

    protein_g = weight_for_macros * protein_g_per_kg
    fat_g = weight_for_macros * fat_g_per_kg

    kcal_protein = protein_g * 4
    kcal_fat = fat_g * 9

    kcal_carbs = max(target_kcal - (kcal_protein + kcal_fat), 0)
    carbs_g = kcal_carbs / 4 if kcal_carbs > 0 else 0

    # Percentages
    if target_kcal > 0:
        protein_pct = kcal_protein / target_kcal * 100
        fat_pct = kcal_fat / target_kcal * 100
        carbs_pct = kcal_carbs / target_kcal * 100
    else:
        protein_pct = fat_pct = carbs_pct = 0

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
):
    # Language instructions for the model
    if language == "Spanish":
        lang_note = (
            "IMPORTANT: Responde TODO en español, incluyendo encabezados, etiquetas y descripciones. "
            "Usa un estilo claro y fácil de entender para pacientes."
        )
    else:
        lang_note = (
            "IMPORTANT: Respond entirely in English. "
            "Use a clear, patient-friendly style."
        )

    # Clinical diet instructions
    clinical_note = ""
    if diet_pattern == "Cardiac (CHF / low sodium)":
        limit_txt = f"{fluid_limit_l:.1f} L/day" if fluid_limit_l else "1.5–2.0 L/day"
        clinical_note = f"""
CLINICAL DIET PATTERN: Cardiac diet for CHF with reduced ejection fraction.
- Sodium goal: generally < 2,000 mg/day.
- Emphasize high-fiber, low-sodium foods; minimize processed and canned foods.
- Avoid obviously salty foods (chips, fries, cured meats, canned soups, frozen dinners with high sodium).
- Fluid restriction: target total fluid intake of {limit_txt} per day (all beverages, soups, and liquid foods count).
  Standard CHF guidance is ~1.5 liters (1500 mL/day), unless otherwise individualized.
- Include a simple suggested fluid schedule over the day (for example morning / afternoon / evening allowances).
"""
    elif diet_pattern == "Diabetic":
        clinical_note = """
CLINICAL DIET PATTERN: Diabetic diet for active diabetes.
- Emphasize consistent, moderate carbohydrate intake spread throughout the day.
- Prefer low–glycemic index carbohydrates (beans, lentils, whole grains, non-starchy vegetables).
- Avoid sugary drinks, juice, desserts; minimize refined carbs and added sugars.
- Pair carbohydrates with protein and/or fat to reduce postprandial glucose spikes.
- Prioritize a zero carb first meal to avoid insulin spikes after first meal.
"""
    elif diet_pattern == "Renal (ESRD / CKD 4-5)":
        clinical_note = """
CLINICAL DIET PATTERN: Renal diet for ESRD or CKD stage 4–5 (general guidance, not individualized).
- Limit sodium and highly processed foods.
- Avoid very high potassium foods in large amounts (bananas, oranges, potatoes, tomatoes, spinach, avocados, etc.).
- Limit high phosphorus foods (colas, many processed foods, some dairy, organ meats).
- Use moderate portions of protein; avoid extremely high-protein fad diets unless on dialysis and advised otherwise.
- Prefer lower-potassium fruits and vegetables and simple home-cooked meals over restaurant / fast-food when possible.
"""

    # Fast-food instructions – real menu items + slightly high price estimates
    fast_food_note = ""
    if fast_food_chains and fast_food_percent > 0:
        chains_txt = ", ".join(fast_food_chains)
        fast_food_note = f"""
FAST-FOOD / TAKEOUT PATTERN (REAL MENU ITEMS ONLY):
- Patient is okay with using some meals from these fast-food chains: {chains_txt}.
- Aim for roughly {fast_food_percent}% of total weekly meals to be from fast-food or takeout.
- Use ONLY real menu items that actually exist or have existed on the standard menu at those chains
  (for example: grilled chicken sandwiches, burrito bowls, egg white breakfast sandwiches, salads from known chains).
- Prefer core, long-running menu items rather than limited-time specials to reduce error.
- For each fast-food meal, specify the restaurant and exact item name.
- For each item, provide approximate calories, protein, carbohydrates, fat, and sodium using best available knowledge.
- If you are unsure about a very specific item, choose a different well-known item from the same restaurant that you know better.
"""

    # Meal timing note
    meal_timing_note = f"""
MEAL TIMING PREFERENCES:
- Target {big_meals_per_day} main meal(s) and {snacks_per_day} snack time(s) per day.
- Main meals should contain the majority of daily calories and protein.
- Snacks should be lighter and help fill in remaining macros without overshooting daily targets.
"""

    # Cooking vs premade
    if prep_style == "Mostly premade / ready-to-eat from store":
        prep_note = """
COOKING VS PREMADE:
- Prioritize ready-to-eat or minimal-prep items (rotisserie chicken, pre-cooked grains, frozen vegetables, bagged salads, pre-made soups, etc.).
- Avoid complicated recipes; most meals should be assembly or reheat rather than full scratch cooking.
"""
    elif prep_style == "Mostly home-cooked meals":
        prep_note = """
COOKING VS PREMADE:
- Emphasize simple home-cooked meals using basic ingredients.
- Occasional premade or frozen items are okay, but most meals should involve basic cooking (stovetop, oven, or air fryer).
"""
    else:  # Balanced mix
        prep_note = """
COOKING VS PREMADE:
- Use a balanced mix of home-cooked meals, ready-to-eat items, and occasional fast-food or takeout.
- Reuse ingredients across cooked and premade meals to save time and reduce waste.
"""

    # Household / family-planning note
    household_note = ""
    if household_size and household_size > 1:
        household_note = f"""
HOUSEHOLD / FAMILY MEAL SCALING:
- The calorie and macro targets apply ONLY to the primary individual.
- Meals and recipes should be planned so they reasonably feed approximately {household_size} people.
- Portions may be scaled (for example: cook 4 chicken breasts instead of 1).
- Grocery quantities and total cost MUST reflect feeding about {household_size} people.
- Macro estimates MUST reflect ONLY the primary individual's portion.
- For fast-food meals, specify what the household orders, but macros apply to the primary individual's portion.
"""

    # Pricing note – estimates only, biased upward for restaurants
    two_week_budget = weekly_budget * 2.0
    pricing_note = f"""
PRICING AND GROCERY COST (ESTIMATES ONLY):
- All prices are approximate and must NOT use real-time data from any retailer or restaurant.
- Base grocery prices on typical U.S. supermarket averages.
- For restaurant / fast-food meals, estimate prices slightly higher than historical national averages
  (roughly 10–25 percent higher) to reflect current prices and regional variation.
- Weekly grocery budget is approximately ${weekly_budget:.2f} for the household (about {household_size} people).
- You are planning for 14 days (2 weeks), so try to keep the total 14-day food cost (groceries + fast-food)
  near ${two_week_budget:.2f} for the household.
- When multiple preferred stores are listed, you may choose one primary store and assume most items are purchased there.
- For each grocery list item, include an estimated unit price and a line total.
- Provide an estimated total grocery cost for all 14 days and a rough per-week average cost.
"""

    # Final prompt
    return f"""
{lang_note}

You are a registered dietitian and meal-planning assistant.

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
{household_note}
{pricing_note}

MEAL PLAN TASK:
Create a 14-day meal plan for a single adult (the primary individual) based on the macro targets and constraints above.
Plan meals and grocery quantities so they can reasonably feed the whole household (about {household_size} people),
but keep all calorie and macro estimates focused on the primary individual's portion only.

STRUCTURE:
- For each day, include exactly {big_meals_per_day} main meal(s) and {snacks_per_day} snack time(s).
- Label them clearly (for example: Breakfast, Lunch, Dinner, Snack 1, Snack 2).
- Distribute calories and macros so that totals for the day roughly match the macro targets for the primary individual.

Additional requirements:
- Keep recipes simple and realistic for a busy person.
- Reuse ingredients across meals to save cost and reduce waste.
- Keep daily totals reasonably close to the macro targets.
- Assume typical adult portion sizes; you may approximate macros.
- Respect the clinical diet pattern if one is specified.

OUTPUT FORMAT (plain text, no markdown tables):
IMPORTANT RESTRICTIONS (MANDATORY):
You MUST NOT include optional commentary, explanations, tips, recipe suggestions, or offers for additional help.
Do NOT include phrases such as "If you would like…" or similar.
Only output the required meal plan, macros, costs, summaries, and grocery list exactly in the format shown.

Day 1
- Main meal 1: ...
  Approx: X kcal, P: Y g, C: Z g, F: W g
- Snack 1: ...
- Main meal 2: ...
- Snack 2: ...
(Adjust labels and counts to match the number of main meals and snacks requested.)

Repeat for Days 1–14.

At the end of the 14 days, include:
1) A rough estimated daily calorie and macro summary per day for the primary individual.
2) A rough estimated daily and total 14-day cost (grocery + fast-food if used), plus a rough per-week average.
3) A combined grocery list grouped by category:
   - Produce
   - Protein
   - Dairy
   - Grains / Starches
   - Pantry
   - Frozen
   - Other
   For each item, include approximate unit price and line total.

PRICE DISCLAIMER:
All prices are estimates only and NOT real-time retailer data. Actual prices vary by store and region.
"""


def generate_meal_plan_with_ai(
    macros: MacroResult,
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
) -> str:
    prompt = build_mealplan_prompt(
        macros=macros,
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
    )

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "You are a precise, practical meal-planning assistant for evidence-based weight management.",
            },
            {"role": "user", "content": prompt},
        ],
    )

    return completion.choices[0].message.content


# ---------- PDF GENERATION ----------

def create_pdf_from_text(text: str, title: str = "Meal Plan") -> bytes:
    """
    Turn plain text into a multi-page PDF with real tables for the daily meals.
    It looks for patterns like:

    Day 1
    - Main meal 1: ...
      Approx: X kcal, P: Y g, C: Z g, F: W g

    and converts them into rows:
    Meal # | Meal description | Approx kcal & macros

    Any remaining text (summaries, grocery list, etc.) is rendered as
    plain wrapped text after the tables.
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

    # Fonts / sizes
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

    table_leading = 10  # line spacing inside table cells
    body_leading = 14   # line spacing for plain text

    # ---------- Helpers ----------

    def wrap_text(text_line: str, font_name: str, font_size: int, max_width: float):
        """Wrap a single line of text to fit within max_width using the given font."""
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
        """
        Parse the raw text into a structure:
        [
          {
            "day_title": "Day 1",
            "rows": [
              ("Meal 1", "Main meal 1: ...", "Approx: X kcal, P: ..."),
              ("Meal 2", "Snack 1: ...", "Approx: ..."),
              ...
            ]
          },
          ...
        ]

        Also returns a list of "leftover" lines (non-meal content) to print as
        regular text after the tables.
        """
        lines = text.splitlines()
        used_indices = set()

        days = []
        current_day = None
        current_rows = []

        i = 0
        while i < len(lines):
            raw = lines[i]
            line = raw.strip()

            # Detect "Day X" header
            if line.startswith("Day "):
                if current_day is not None:
                    days.append({"day_title": current_day, "rows": current_rows})
                    current_rows = []

                current_day = line
                used_indices.add(i)
                i += 1
                continue

            # Detect *meal* bullets, but ONLY if they have an "Approx:" line after them
            if current_day is not None and line.startswith("- "):
                desc = line[2:].strip()

                # Look ahead for an "Approx:" line
                macros_line = ""
                macros_idx = None
                j = i + 1
                while j < len(lines):
                    look_raw = lines[j]
                    look = look_raw.strip()
                    # Stop if we hit next meal or next day
                    if look.startswith("- ") or look.startswith("Day "):
                        break
                    if look.startswith("Approx:"):
                        macros_line = look
                        macros_idx = j
                        break
                    j += 1

                # If there is NO macros line, this bullet is not a meal row.
                if not macros_line:
                    i += 1
                    continue

                # Otherwise, treat as a meal row
                used_indices.add(i)
                if macros_idx is not None:
                    used_indices.add(macros_idx)

                meal_number = len(current_rows) + 1
                current_rows.append(
                    (f"Meal {meal_number}", desc, macros_line or "")
                )

                i += 1
                continue

            i += 1

        # Store last day
        if current_day is not None:
            days.append({"day_title": current_day, "rows": current_rows})

        # Leftover lines (non-table content)
        leftover_lines = [
            lines[idx] for idx in range(len(lines)) if idx not in used_indices
        ]

        return days, leftover_lines

    def new_page_with_title():
        """Start a new page and redraw the main title."""
        c.showPage()
        c.setFont(title_font, title_size)
        c.drawString(left_margin, page_height - top_margin, title)
        y = page_height - top_margin - 25
        return y

    # ---------- Start first page with main title ----------
    c.setFont(title_font, title_size)
    c.drawString(left_margin, page_height - top_margin, title)
    current_y = page_height - top_margin - 25

    # Parse text into structured days + leftover text
    days, leftover_lines = parse_days_and_meals(text)

    # ---------- Table column layout ----------
    col1_width = 70   # Meal #
    col3_width = 130  # Macros
    col2_width = usable_width - col1_width - col3_width  # Description

    col1_x = left_margin
    col2_x = col1_x + col1_width
    col3_x = col2_x + col2_width

    # ---------- Draw tables for each day ----------
    for day in days:
        day_title = day["day_title"]
        rows = day["rows"]

        if not rows:
            # If somehow no rows for this day, just print the day label as text
            if current_y <= bottom_margin + 40:
                current_y = new_page_with_title()
            c.setFont(day_font, day_size)
            c.drawString(left_margin, current_y, day_title)
            current_y -= 20
            continue

        # Ensure space for day title + header
        if current_y <= bottom_margin + 60:
            current_y = new_page_with_title()

        # Day title
        c.setFont(day_font, day_size)
        c.drawString(left_margin, current_y, day_title)
        current_y -= 25  # a bit more spacing after the day label

        # Table header
        header_height = table_leading + 4
        if current_y - header_height <= bottom_margin:
            current_y = new_page_with_title()
            c.setFont(day_font, day_size)
            c.drawString(left_margin, current_y, day_title + " (cont.)")
            current_y -= 25

        c.setFont(table_header_font, table_header_size)
        header_top_y = current_y
        header_bottom_y = current_y - header_height

        # Horizontal borders
        c.line(left_margin, header_top_y, page_width - right_margin, header_top_y)
        c.line(left_margin, header_bottom_y, page_width - right_margin, header_bottom_y)

        # Vertical column borders
        c.line(col1_x, header_top_y, col1_x, header_bottom_y)
        c.line(col2_x, header_top_y, col2_x, header_bottom_y)
        c.line(col3_x, header_top_y, col3_x, header_bottom_y)
        c.line(page_width - right_margin, header_top_y, page_width - right_margin, header_bottom_y)

        # Header labels (vertically centered-ish)
        baseline_offset = (header_height - table_header_size) / 2
        text_y = header_top_y - baseline_offset - 5
        c.drawString(col1_x + 2, text_y, "Meal #")
        c.drawString(col2_x + 2, text_y, "Meal description")
        c.drawString(col3_x + 2, text_y, "Approx kcal & macros")

        current_y = header_bottom_y

        # Table body rows
        c.setFont(table_body_font, table_body_size)
        for meal_no, desc, macros in rows:
            # Wrap each cell
            meal_lines = wrap_text(meal_no, table_body_font, table_body_size, col1_width - 4)
            desc_lines = wrap_text(desc, table_body_font, table_body_size, col2_width - 4)
            macros_lines = wrap_text(macros, table_body_font, table_body_size, col3_width - 4)

            num_lines = max(len(meal_lines), len(desc_lines), len(macros_lines))
            row_height = num_lines * table_leading + 4

            # New page if needed
            if current_y - row_height <= bottom_margin:
                current_y = new_page_with_title()
                # Redraw day title and header on new page (continuation)
                c.setFont(day_font, day_size)
                c.drawString(left_margin, current_y, f"{day_title} (cont.)")
                current_y -= 25

                # Header again
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

            # Draw row borders
            row_top_y = current_y
            row_bottom_y = current_y - row_height

            c.line(left_margin, row_top_y, page_width - right_margin, row_top_y)
            c.line(left_margin, row_bottom_y, page_width - right_margin, row_bottom_y)
            c.line(col1_x, row_top_y, col1_x, row_bottom_y)
            c.line(col2_x, row_top_y, col2_x, row_bottom_y)
            c.line(col3_x, row_top_y, col3_x, row_bottom_y)
            c.line(page_width - right_margin, row_top_y, page_width - right_margin, row_bottom_y)

            # Draw text inside row
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

        current_y -= 18  # Space after each day's table

    # ---------- Leftover text as plain wrapped text ----------
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
        # Sync current_y with where the leftover text actually ended
        current_y = text_obj.getY()

    # ---------- Disclaimer Section ----------
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

    # Add a little space before the disclaimer
    current_y -= 20
    if current_y <= bottom_margin + 40:
        current_y = new_page_with_title()

    c.setFont(body_font, body_size)
    text_obj = c.beginText()
    text_obj.setTextOrigin(left_margin, current_y)
    text_obj.setFont(body_font, body_size)
    text_obj.setLeading(body_leading)

    # Draw the disclaimer, wrapped to the page width
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

    # Do NOT add an extra showPage() here; just save the current page.
    c.save()

    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


# ---------- STREAMLIT UI ----------

def main():
    st.set_page_config(
        page_title="Evidence-Based Macro & Meal Planner",
        layout="centered"
    )

    st.title("Personalized Meal Planning for the busy clinician")
    st.write("Enter metrics and personal preferences below")

    # 1. Patient / User Info
    st.subheader("1. Patient / User Info")

    col1, col2 = st.columns(2)

    # ---- COLUMN 1: SEX, AGE, HEIGHT ----
    with col1:
        sex = st.selectbox("Sex", options=["M", "F"])
        age = st.number_input("Age (years)", min_value=12, max_value=100, value=30)

        height_unit = st.radio(
            "Height units",
            options=["cm", "ft/in"],
            index=1,
            horizontal=True,
        )

        if height_unit == "cm":
            height_cm = st.number_input(
                "Height (cm)",
                min_value=120.0,
                max_value=230.0,
                value=170.0,
            )
        else:
            height_ft = st.number_input(
                "Height (feet)",
                min_value=3,
                max_value=7,
                value=5,
            )
            height_in = st.number_input(
                "Height (inches)",
                min_value=0,
                max_value=11,
                value=6,
            )
            height_cm = height_ft * 30.48 + height_in * 2.54
            st.caption(f"Calculated height: {height_cm:.1f} cm")

    # ---- COLUMN 2: WEIGHT (KG OR LBS) ----
    with col2:
        weight_unit = st.radio(
            "Weight units",
            options=["kg", "lbs"],
            index=1,  # default to lbs
            horizontal=True,
        )

        if weight_unit == "kg":
            weight_current_kg = st.number_input(
                "Current weight (kg)",
                min_value=30.0,
                max_value=300.0,
                value=70.0,
            )
            weight_goal_kg = st.number_input(
                "Goal weight (kg)",
                min_value=30.0,
                max_value=300.0,
                value=65.0,
            )

        else:
            weight_current_lbs = st.number_input(
                "Current weight (lbs)",
                min_value=60.0,
                max_value=660.0,
                value=154.0,
            )
            weight_goal_lbs = st.number_input(
                "Goal weight (lbs)",
                min_value=60.0,
                max_value=660.0,
                value=143.0,
            )

            weight_current_kg = weight_current_lbs / 2.20462
            weight_goal_kg = weight_goal_lbs / 2.20462

            st.caption(
                f"Current weight: {weight_current_kg:.1f} kg\n"
                f"Goal weight: {weight_goal_kg:.1f} kg"
            )

        weight_source = st.selectbox(
            "Weight used for macros",
            options=["Current", "Goal"]
        )

    # 2. Activity & Weight-Loss Settings
    st.subheader("2. Activity & Weight-Loss Settings")
    activity_factor = st.number_input(
        "Activity factor",
        min_value=1.1,
        max_value=2.5,
        value=1.375,
        step=0.025,
        help="Typical: 1.2 sedentary, 1.375 light, 1.55 moderate, 1.725 very active."
    )

    intensity = st.selectbox(
        "Weight-loss intensity",
        options=["Gentle", "Moderate", "Aggressive"],
        help="Gentle ≈250 kcal/day deficit, Moderate ≈500, Aggressive ≈750."
    )

    # 3. Macro Settings
    st.subheader("3. Macro Settings (g/kg)")
    col3, col4 = st.columns(2)
    with col3:
        protein_g_per_kg = st.number_input(
            "Protein (g/kg for macro weight)",
            min_value=0.8,
            max_value=2.5,
            value=1.4,
            step=0.1,
            help="Evidence-supported weight loss range: 1.2–1.6 g/kg."
        )
    with col4:
        fat_g_per_kg = st.number_input(
            "Fat (g/kg for macro weight)",
            min_value=0.3,
            max_value=1.5,
            value=0.7,
            step=0.1,
            help="Common clinical range: 0.5–1.0 g/kg."
        )

    # 4. Preferences
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

    # 5. Clinical diet pattern
    st.subheader("5. Clinical diet pattern (optional)")

    diet_pattern = st.selectbox(
        "Apply a medical diet template",
        options=[
            "None",
            "Cardiac (CHF / low sodium)",
            "Diabetic",
            "Renal (ESRD / CKD 4-5)"
        ],
        help="Adds extra constraints to the meal plan but keeps your macros as a guide."
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

    # 6. Fast-food options
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
                "McDonald's",
                "Chick-fil-A",
                "Taco Bell",
                "Subway",
                "Chipotle",
                "Wendy's",
                "Burger King",
                "Panera Bread",
                "Starbucks",
                "Five Guys",
                "Whataburger",
                "In-N-Out Burger",
                "Jack in the Box",
                "Sonic Drive-In",
                "Carl's Jr.",
                "Hardee's",
                "Culver's",
                "Smashburger",
                "Rally's / Checkers",
                "Freddy's Frozen Custard & Steakburgers",
                "Steak 'n Shake",
                "KFC",
                "Popeyes",
                "Raising Cane's",
                "Church's Chicken",
                "Zaxby's",
                "Wingstop",
                "Bojangles",
                "El Pollo Loco",
                "Domino's",
                "Pizza Hut",
                "Papa John's",
                "Little Caesars",
                "Marco's Pizza",
                "Papa Murphy's",
                "Blaze Pizza",
                "MOD Pizza",
                "Jimmy John's",
                "Jersey Mike's",
                "Firehouse Subs",
                "Which Wich",
                "Potbelly",
                "Qdoba",
                "Del Taco",
                "Taco Cabana",
                "Moe's Southwest Grill",
                "Baja Fresh",
                "Dunkin'",
                "Tim Hortons",
                "Einstein Bros Bagels",
                "Krispy Kreme",
                "Dutch Bros Coffee",
                "Long John Silver's",
                "Captain D's",
                "Wienerschnitzel",
                "Nathan's Famous",
                "Dairy Queen",
                "Baskin Robbins",
                "Cold Stone Creamery",
                "Ben & Jerry's",
                "Panda Express",
                "Arby's",
                "Shake Shack",
                "Noodles & Company",
                "Jollibee",
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

    # 7. Meal timing
    st.subheader("7. Meal timing (optional)")

    big_meals_per_day = st.selectbox(
        "Number of main meals per day",
        options=[1, 2, 3],
        index=2,
        help="For example, choose 2 if they prefer something like brunch + dinner."
    )

    snacks_per_day = st.selectbox(
        "Number of snack times per day",
        options=[0, 1, 2, 3],
        index=2,
        help="Snacks will be lighter and used to fill in remaining macros."
    )

    # 8. Cooking vs premade balance
    st.subheader("8. Cooking vs premade balance")

    prep_style = st.selectbox(
        "Preferred style",
        options=[
            "Balanced: mix of cooking and premade",
            "Mostly premade / ready-to-eat from store",
            "Mostly home-cooked meals",
        ],
        help="Guides how many meals rely on cooking vs ready-to-eat store items."
    )

    # 9. Household / family planning
    st.subheader("9. Household / family planning")

    household_size = st.number_input(
        "How many people will be eating most of these meals?",
        min_value=1,
        max_value=10,
        value=1,
        step=1,
        help=(
            "Macros and calorie targets remain for ONE primary individual. "
            "Meals and grocery quantities will be scaled to feed this many people."
        ),
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
            intensity=intensity,  # type: ignore[arg-type]
            protein_g_per_kg=float(protein_g_per_kg),
            fat_g_per_kg=float(fat_g_per_kg),
        )

        st.success("Calculated macros successfully.")

        st.subheader("Calculated Energy and Macros")
        colA, colB = st.columns(2)
        with colA:
            st.metric("Resting Metabolic Rate (RMR)", f"{macros.rmr:.0f} kcal/day")
            st.metric("TDEE (maintenance)", f"{macros.tdee:.0f} kcal/day")
            st.metric("Weight-loss target", f"{macros.target_kcal:.0f} kcal/day")
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
                )
                st.session_state["plan_text"] = plan_text
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
