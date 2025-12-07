import os
from dataclasses import dataclass
from typing import Literal
from io import BytesIO

import streamlit as st
from openai import OpenAI
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

MODEL_NAME = "gpt-4.1-mini"

api_key = None
try:
    api_key = st.secrets["OPENAI_API_KEY"]
except:
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

    return f"""
{lang_note}

You are a registered dietitian and meal-planning assistant.

Create a 7-day meal plan (breakfast, lunch, dinner, and 1–2 snacks per day)
for a single adult based on the following macro targets:

- Daily calories: {macros.target_kcal:.0f} kcal
- Protein: {macros.protein_g:.0f} g/day
- Carbohydrates: {macros.carbs_g:.0f} g/day
- Fats: {macros.fat_g:.0f} g/day

CONSTRAINTS:
- Allergies / must AVOID: {allergies or "none specified"}
- Foods to avoid / dislikes: {dislikes or "none specified"}
- Weekly grocery budget: ${weekly_budget:.2f} for all 7 days
- Preferred grocery store or market: {preferred_store or "generic US supermarket"}
- Use foods that are realistically available at that store.
- Keep recipes simple and realistic for a busy person.
- Reuse ingredients across meals to save cost and reduce waste.
- Keep daily totals reasonably close to the macro targets.
- Assume typical adult portion sizes; you may approximate macros.

OUTPUT FORMAT (plain text, no markdown tables):
Day 1
- Breakfast: ...
  Approx: X kcal, P: Y g, C: Z g, F: W g
- Snack: ...
- Lunch: ...
- Snack: ...
- Dinner: ...

Repeat for Days 1–7.

At the end, include:
1) A rough estimated daily calorie and macro summary per day.
2) A rough estimated daily and weekly cost.
3) One combined grocery list grouped by category:
   - Produce
   - Protein
   - Dairy
   - Grains / Starches
   - Pantry
   - Frozen
   - Other.
"""


def generate_meal_plan_with_ai(
    macros: MacroResult,
    allergies: str,
    dislikes: str,
    preferred_store: str,
    weekly_budget: float,
    language: str,
) -> str:
    prompt = build_mealplan_prompt(
        macros=macros,
        allergies=allergies,
        dislikes=dislikes,
        preferred_store=preferred_store,
        weekly_budget=weekly_budget,
        language=language,
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
    Turn plain text into a simple PDF and return it as bytes.
    """
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Optional title
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, height - 40, title)

    # Text body
    textobject = c.beginText()
    textobject.setTextOrigin(40, height - 60)
    textobject.setFont("Helvetica", 10)
    textobject.setLeading(14)

    for line in text.split("\n"):
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

    st.title("Evidence-Based BMR, Macros & AI Meal Planner")
    st.write("Prototype app using your macro logic plus an AI-generated 7-day meal plan.")

    with st.form("inputs"):
        st.subheader("1. Patient / User Info")

        col1, col2 = st.columns(2)
        with col1:
            sex = st.selectbox("Sex", options=["M", "F"])
            age = st.number_input("Age (years)", min_value=12, max_value=100, value=30)
            height_cm = st.number_input("Height (cm)", min_value=120.0, max_value=230.0, value=170.0)
        with col2:
            weight_current_kg = st.number_input("Current weight (kg)", min_value=30.0, max_value=300.0, value=70.0)
            weight_goal_kg = st.number_input("Goal weight (kg)", min_value=30.0, max_value=300.0, value=65.0)
            weight_source = st.selectbox("Weight used for macros", options=["Current", "Goal"])

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

        submitted = st.form_submit_button("Calculate macros and generate meal plan")

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
                )
                st.text_area("Meal plan", plan_text, height=500)

                pdf_bytes = create_pdf_from_text(
                    plan_text,
                    title="Meal Plan (Spanish)" if language == "Spanish" else "Meal Plan (English)",
                )

                st.download_button(
                    label="Download meal plan as PDF",
                    data=pdf_bytes,
                    file_name="meal_plan.pdf",
                    mime="application/pdf",
                )

            except Exception as e:
                st.error(f"Error generating meal plan: {e}")


if __name__ == "__main__":
    main()


