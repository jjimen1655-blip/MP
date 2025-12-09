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

    # Pricing note – estimates only, biased upward for restaurants
    pricing_note = f"""
PRICING AND GROCERY COST (ESTIMATES ONLY):
- All prices are approximate and must NOT use real-time data from any retailer or restaurant.
- Base grocery prices on typical U.S. supermarket averages.
- For restaurant / fast-food meals, estimate prices slightly higher than historical national averages
  (roughly 10–25 percent higher) to reflect current prices and regional variation.
- When multiple preferred stores are listed, you may choose one primary store and assume most items are purchased there.
- For each grocery list item, include an estimated unit price and a line total.
- Provide an estimated total grocery cost for the week and an overall weekly food cost including any fast-food meals.
- Try to keep the weekly total near ${weekly_budget:.2f}, but approximations are acceptable.
"""

    # Final prompt
    return f"""
{lang_note}

You are a registered dietitian and meal-planning assistant.

MACRO TARGETS (PER DAY):
- Daily calories: {macros.target_kcal:.0f} kcal
- Protein: {macros.protein_g:.0f} g/day
- Carbohydrates: {macros.carbs_g:.0f} g/day
- Fats: {macros.fat_g:.0f} g/day

PATIENT CONSTRAINTS:
- Allergies / must AVOID: {allergies or "none specified"}
- Foods to avoid / dislikes: {dislikes or "none specified"}
- Weekly grocery budget: ${weekly_budget:.2f} for all 7 days
- Preferred grocery store or market: {preferred_store or "generic US supermarket"}

{clinical_note}
{fast_food_note}
{meal_timing_note}
{prep_note}
{pricing_note}

MEAL PLAN TASK:
Create a 7-day meal plan for a single adult based on the macro targets and constraints above.

STRUCTURE:
- For each day, include exactly {big_meals_per_day} main meal(s) and {snacks_per_day} snack time(s).
- Label them clearly (for example: Breakfast, Lunch, Dinner, Snack 1, Snack 2).
- Distribute calories and macros so that totals for the day roughly match the macro targets.

Additional requirements:
- Keep recipes simple and realistic for a busy person.
- Reuse ingredients across meals to save cost and reduce waste.
- Keep daily totals reasonably close to the macro targets.
- Assume typical adult portion sizes; you may approximate macros.
- Respect the clinical diet pattern if one is specified.

OUTPUT FORMAT (plain text, no markdown tables):

Day 1
- Main meal 1: ...
  Approx: X kcal, P: Y g, C: Z g, F: W g
- Snack 1: ...
- Main meal 2: ...
- Snack 2: ...
(Adjust labels and counts to match the number of main meals and snacks requested.)

Repeat for Days 1–7.

At the end, include:
1) A rough estimated daily calorie and macro summary per day.
2) A rough estimated daily and weekly cost (grocery + fast-food if used).
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
    Turn plain text into a simple multi-page PDF and return it as bytes.
    Long lines are wrapped so they don't get cut off, and long plans
    continue on additional pages.
    """
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    left_margin = 40
    right_margin = 40
    top_margin = 40
    bottom_margin = 40

    font_name_title = "Helvetica-Bold"
    font_name_body = "Helvetica"
    font_size_title = 14
    font_size_body = 10
    line_leading = 14

    usable_width = width - left_margin - right_margin

    # ---- Helper: wrap a single logical line into multiple PDF lines ----
    def wrap_line(line: str, font_name: str, font_size: int, max_width: float):
        words = line.split(" ")
        if not words:
            return [""]

        wrapped = []
        current = words[0]
        for word in words[1:]:
            test = current + " " + word
            w = pdfmetrics.stringWidth(test, font_name, font_size)
            if w <= max_width:
                current = test
            else:
                wrapped.append(current)
                current = word
        wrapped.append(current)
        return wrapped

    # ---- First page title ----
    c.setFont(font_name_title, font_size_title)
    c.drawString(left_margin, height - top_margin, title)

    # Text object for body
    textobject = c.beginText()
    text_start_y = height - top_margin - 20  # a bit below the title
    textobject.setTextOrigin(left_margin, text_start_y)
    textobject.setFont(font_name_body, font_size_body)
    textobject.setLeading(line_leading)

    for raw_line in text.split("\n"):
        for line in wrap_line(raw_line, font_name_body, font_size_body, usable_width):
            if textobject.getY() <= bottom_margin:
                c.drawText(textobject)
                c.showPage()
                textobject = c.beginText()
                textobject.setTextOrigin(left_margin, height - top_margin)
                textobject.setFont(font_name_body, font_size_body)
                textobject.setLeading(line_leading)
            textobject.textLine(line)

    c.drawText(textobject)
    c.showPage()
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
        "Weekly grocery budget (USD)",
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
        st.subheader("AI-Generated 7-Day Meal Plan")

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

