"""
Comprehensive Lifestyle Recommendation Engine
Provides personalized diet, exercise, and wellness recommendations based on health conditions
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DietRecommendation:
    """Represents a dietary recommendation"""
    category: str
    foods_to_include: List[str]
    foods_to_avoid: List[str]
    meal_suggestions: List[str]
    nutritional_goals: List[str]
    portion_guidelines: List[str]

@dataclass
class ExerciseRecommendation:
    """Represents an exercise recommendation"""
    category: str
    recommended_activities: List[str]
    frequency: str
    duration: str
    intensity: str
    precautions: List[str]
    progression_tips: List[str]

@dataclass
class LifestyleRecommendation:
    """Represents general lifestyle recommendations"""
    category: str
    recommendations: List[str]
    benefits: List[str]
    implementation_tips: List[str]

@dataclass
class ComprehensiveLifestylePlan:
    """Complete lifestyle plan for a patient"""
    diet_recommendations: List[DietRecommendation]
    exercise_recommendations: List[ExerciseRecommendation]
    lifestyle_recommendations: List[LifestyleRecommendation]
    priority_actions: List[str]
    monitoring_suggestions: List[str]
    professional_consultations: List[str]

class LifestyleRecommendationEngine:
    """Main engine for generating comprehensive lifestyle recommendations"""
    
    def __init__(self):
        self.setup_knowledge_base()
    
    def setup_knowledge_base(self):
        """Initialize comprehensive knowledge base for lifestyle recommendations"""
        
        # Diet recommendations by condition
        self.diet_recommendations = {
            'diabetes': DietRecommendation(
                category="Diabetes Management Diet",
                foods_to_include=[
                    "Non-starchy vegetables (broccoli, spinach, peppers)",
                    "Lean proteins (chicken, fish, tofu, legumes)",
                    "Whole grains (quinoa, brown rice, oats)",
                    "Healthy fats (avocado, nuts, olive oil)",
                    "Low-glycemic fruits (berries, apples, citrus)",
                    "High-fiber foods (beans, lentils, chia seeds)"
                ],
                foods_to_avoid=[
                    "Refined sugars and sweets",
                    "White bread and refined grains",
                    "Sugary beverages and fruit juices",
                    "Processed and packaged foods",
                    "High-sodium foods",
                    "Trans fats and fried foods"
                ],
                meal_suggestions=[
                    "Breakfast: Greek yogurt with berries and nuts",
                    "Lunch: Grilled chicken salad with olive oil dressing",
                    "Dinner: Baked salmon with quinoa and steamed vegetables",
                    "Snacks: Apple with almond butter, handful of nuts"
                ],
                nutritional_goals=[
                    "Maintain stable blood sugar levels",
                    "Aim for 25-35g fiber per day",
                    "Include protein in every meal",
                    "Limit carbohydrates to 45-60g per meal"
                ],
                portion_guidelines=[
                    "Use plate method: 1/2 vegetables, 1/4 protein, 1/4 whole grains",
                    "Monitor carbohydrate portions carefully",
                    "Eat regular, consistent meals"
                ]
            ),
            
            'hypertension': DietRecommendation(
                category="DASH Diet for Blood Pressure",
                foods_to_include=[
                    "Fruits and vegetables (8-10 servings daily)",
                    "Whole grains (6-8 servings daily)",
                    "Low-fat dairy products",
                    "Lean meats, poultry, and fish",
                    "Nuts, seeds, and legumes",
                    "Potassium-rich foods (bananas, oranges, spinach)"
                ],
                foods_to_avoid=[
                    "High-sodium processed foods",
                    "Canned soups and sauces",
                    "Deli meats and cured meats",
                    "Fast food and restaurant meals",
                    "Excessive alcohol",
                    "Added sugars and sweets"
                ],
                meal_suggestions=[
                    "Breakfast: Oatmeal with banana and low-fat milk",
                    "Lunch: Turkey and vegetable wrap with whole grain tortilla",
                    "Dinner: Grilled fish with roasted vegetables and brown rice",
                    "Snacks: Fresh fruit, unsalted nuts, low-fat yogurt"
                ],
                nutritional_goals=[
                    "Limit sodium to less than 2,300mg daily (ideally 1,500mg)",
                    "Increase potassium intake to 3,500-4,700mg daily",
                    "Maintain healthy weight",
                    "Limit alcohol consumption"
                ],
                portion_guidelines=[
                    "Read nutrition labels for sodium content",
                    "Use herbs and spices instead of salt",
                    "Choose fresh over processed foods"
                ]
            ),
            
            'heart_disease': DietRecommendation(
                category="Heart-Healthy Mediterranean Diet",
                foods_to_include=[
                    "Fatty fish (salmon, mackerel, sardines) 2-3 times weekly",
                    "Extra virgin olive oil as primary fat",
                    "Nuts and seeds (almonds, walnuts, flaxseeds)",
                    "Colorful vegetables and fruits",
                    "Whole grains and legumes",
                    "Moderate amounts of poultry and eggs"
                ],
                foods_to_avoid=[
                    "Saturated and trans fats",
                    "Red meat and processed meats",
                    "Refined sugars and processed foods",
                    "High-sodium foods",
                    "Excessive alcohol",
                    "Fried and fast foods"
                ],
                meal_suggestions=[
                    "Breakfast: Greek yogurt with walnuts and berries",
                    "Lunch: Mediterranean salad with olive oil and lemon",
                    "Dinner: Grilled salmon with vegetables and quinoa",
                    "Snacks: Handful of almonds, hummus with vegetables"
                ],
                nutritional_goals=[
                    "Reduce LDL cholesterol levels",
                    "Increase omega-3 fatty acids",
                    "Maintain healthy weight",
                    "Support overall cardiovascular health"
                ],
                portion_guidelines=[
                    "Focus on plant-based meals",
                    "Use olive oil in moderation",
                    "Include fish 2-3 times per week"
                ]
            ),
            
            'obesity': DietRecommendation(
                category="Weight Management Diet",
                foods_to_include=[
                    "High-fiber vegetables and fruits",
                    "Lean proteins (chicken breast, fish, legumes)",
                    "Whole grains in controlled portions",
                    "Low-fat dairy products",
                    "Healthy fats in small amounts",
                    "Water and herbal teas"
                ],
                foods_to_avoid=[
                    "Calorie-dense processed foods",
                    "Sugary beverages and alcohol",
                    "Large portion sizes",
                    "High-fat fried foods",
                    "Refined carbohydrates",
                    "Emotional eating triggers"
                ],
                meal_suggestions=[
                    "Breakfast: Vegetable omelet with whole grain toast",
                    "Lunch: Large salad with grilled chicken and vinaigrette",
                    "Dinner: Baked fish with steamed vegetables",
                    "Snacks: Raw vegetables, small portion of nuts"
                ],
                nutritional_goals=[
                    "Create sustainable caloric deficit",
                    "Aim for 1-2 pounds weight loss per week",
                    "Maintain muscle mass during weight loss",
                    "Develop healthy eating habits"
                ],
                portion_guidelines=[
                    "Use smaller plates and bowls",
                    "Measure portions initially",
                    "Eat slowly and mindfully"
                ]
            )
        }
        
        # Exercise recommendations by condition
        self.exercise_recommendations = {
            'diabetes': ExerciseRecommendation(
                category="Diabetes Exercise Program",
                recommended_activities=[
                    "Brisk walking (30-45 minutes daily)",
                    "Swimming or water aerobics",
                    "Cycling or stationary bike",
                    "Resistance training (2-3 times weekly)",
                    "Yoga or tai chi for flexibility",
                    "Dancing or recreational sports"
                ],
                frequency="5-7 days per week for aerobic, 2-3 days for strength",
                duration="150 minutes moderate aerobic activity weekly",
                intensity="Moderate (can talk while exercising)",
                precautions=[
                    "Monitor blood glucose before, during, and after exercise",
                    "Carry glucose tablets for hypoglycemia",
                    "Check feet daily for injuries",
                    "Stay hydrated",
                    "Start slowly and progress gradually"
                ],
                progression_tips=[
                    "Begin with 10-minute sessions if sedentary",
                    "Gradually increase duration before intensity",
                    "Add variety to prevent boredom",
                    "Track progress with activity log"
                ]
            ),
            
            'hypertension': ExerciseRecommendation(
                category="Blood Pressure Lowering Exercise",
                recommended_activities=[
                    "Regular brisk walking",
                    "Jogging or running (if appropriate)",
                    "Swimming and water exercises",
                    "Cycling",
                    "Light to moderate weight training",
                    "Yoga and stretching exercises"
                ],
                frequency="Most days of the week",
                duration="30-60 minutes daily",
                intensity="Moderate (50-70% max heart rate)",
                precautions=[
                    "Avoid heavy weightlifting or isometric exercises",
                    "Monitor blood pressure regularly",
                    "Stop if experiencing dizziness or chest pain",
                    "Warm up and cool down properly",
                    "Stay hydrated"
                ],
                progression_tips=[
                    "Start with 15-20 minutes if beginning",
                    "Focus on consistency over intensity",
                    "Include both aerobic and flexibility exercises",
                    "Consider group classes for motivation"
                ]
            ),
            
            'heart_disease': ExerciseRecommendation(
                category="Cardiac Rehabilitation Exercise",
                recommended_activities=[
                    "Supervised walking program",
                    "Stationary cycling",
                    "Low-impact aerobics",
                    "Light resistance training",
                    "Stretching and flexibility exercises",
                    "Breathing exercises"
                ],
                frequency="3-5 days per week initially",
                duration="20-40 minutes per session",
                intensity="Low to moderate (40-60% max heart rate)",
                precautions=[
                    "Must have physician clearance before starting",
                    "Monitor heart rate and symptoms",
                    "Stop immediately if chest pain occurs",
                    "Avoid exercising in extreme temperatures",
                    "Consider cardiac rehabilitation program"
                ],
                progression_tips=[
                    "Start very gradually under supervision",
                    "Progress slowly over weeks/months",
                    "Focus on consistency and safety",
                    "Regular medical monitoring required"
                ]
            ),
            
            'obesity': ExerciseRecommendation(
                category="Weight Loss Exercise Program",
                recommended_activities=[
                    "Walking (start with 10-15 minutes)",
                    "Swimming or water aerobics",
                    "Cycling or stationary bike",
                    "Low-impact aerobics",
                    "Strength training with light weights",
                    "Chair exercises if mobility limited"
                ],
                frequency="5-6 days per week",
                duration="45-60 minutes for weight loss",
                intensity="Moderate (can maintain conversation)",
                precautions=[
                    "Start slowly to avoid injury",
                    "Choose low-impact activities initially",
                    "Wear proper supportive footwear",
                    "Stay hydrated",
                    "Listen to your body"
                ],
                progression_tips=[
                    "Increase duration before intensity",
                    "Add 5 minutes weekly to sessions",
                    "Include both cardio and strength training",
                    "Track progress with measurements, not just weight"
                ]
            )
        }
        
        # General lifestyle recommendations
        self.lifestyle_recommendations = {
            'stress_management': LifestyleRecommendation(
                category="Stress Management",
                recommendations=[
                    "Practice daily meditation or mindfulness (10-20 minutes)",
                    "Maintain regular sleep schedule (7-9 hours nightly)",
                    "Engage in hobbies and enjoyable activities",
                    "Build strong social connections",
                    "Learn time management and prioritization skills",
                    "Consider professional counseling if needed"
                ],
                benefits=[
                    "Reduces cortisol levels",
                    "Improves immune function",
                    "Better sleep quality",
                    "Enhanced mental clarity",
                    "Reduced risk of chronic diseases"
                ],
                implementation_tips=[
                    "Start with 5-minute meditation sessions",
                    "Use smartphone apps for guided meditation",
                    "Schedule regular social activities",
                    "Create a relaxing bedtime routine"
                ]
            ),
            
            'sleep_hygiene': LifestyleRecommendation(
                category="Sleep Optimization",
                recommendations=[
                    "Maintain consistent sleep-wake schedule",
                    "Create dark, cool, quiet sleep environment",
                    "Avoid screens 1 hour before bedtime",
                    "Limit caffeine after 2 PM",
                    "Establish relaxing bedtime routine",
                    "Get morning sunlight exposure"
                ],
                benefits=[
                    "Improved glucose metabolism",
                    "Better blood pressure control",
                    "Enhanced immune function",
                    "Improved mood and cognitive function",
                    "Reduced inflammation"
                ],
                implementation_tips=[
                    "Use blackout curtains or eye mask",
                    "Keep bedroom temperature 65-68°F",
                    "Try reading or gentle stretching before bed",
                    "Avoid large meals close to bedtime"
                ]
            ),
            
            'smoking_cessation': LifestyleRecommendation(
                category="Smoking Cessation",
                recommendations=[
                    "Set a quit date and prepare mentally",
                    "Consider nicotine replacement therapy",
                    "Identify and avoid smoking triggers",
                    "Find healthy alternatives to smoking",
                    "Seek support from family, friends, or support groups",
                    "Consult healthcare provider for cessation aids"
                ],
                benefits=[
                    "Dramatically reduced heart disease risk",
                    "Improved lung function",
                    "Better circulation",
                    "Reduced cancer risk",
                    "Improved sense of taste and smell"
                ],
                implementation_tips=[
                    "Remove all smoking materials from environment",
                    "Replace smoking breaks with short walks",
                    "Use stress management techniques",
                    "Reward yourself for milestones"
                ]
            ),
            
            'hydration': LifestyleRecommendation(
                category="Proper Hydration",
                recommendations=[
                    "Drink 8-10 glasses of water daily",
                    "Start day with a glass of water",
                    "Carry water bottle throughout the day",
                    "Eat water-rich foods (fruits, vegetables)",
                    "Monitor urine color for hydration status",
                    "Increase intake during exercise or hot weather"
                ],
                benefits=[
                    "Supports kidney function",
                    "Helps maintain healthy blood pressure",
                    "Improves energy levels",
                    "Supports healthy skin",
                    "Aids in weight management"
                ],
                implementation_tips=[
                    "Set hourly reminders to drink water",
                    "Flavor water with lemon or cucumber",
                    "Drink water before, during, and after meals",
                    "Replace sugary drinks with water"
                ]
            )
        }
    
    def analyze_conditions(self, conditions: List[str], lab_values: List[str] = None) -> List[str]:
        """Analyze medical conditions and lab values to identify relevant health issues"""
        identified_conditions = []
        
        # Convert to lowercase for matching
        condition_text = ' '.join(conditions).lower() if conditions else ''
        lab_text = ' '.join(lab_values).lower() if lab_values else ''
        combined_text = f"{condition_text} {lab_text}"
        
        # Check for diabetes
        diabetes_keywords = ['diabetes', 'diabetic', 'hyperglycemia', 'high blood sugar', 'hba1c', 'glucose']
        if any(keyword in combined_text for keyword in diabetes_keywords):
            identified_conditions.append('diabetes')
        
        # Check for hypertension
        hypertension_keywords = ['hypertension', 'high blood pressure', 'elevated bp', 'systolic', 'diastolic']
        if any(keyword in combined_text for keyword in hypertension_keywords):
            identified_conditions.append('hypertension')
        
        # Check for heart disease
        heart_keywords = ['heart disease', 'cardiac', 'coronary', 'myocardial', 'angina', 'heart attack', 'cardiovascular']
        if any(keyword in combined_text for keyword in heart_keywords):
            identified_conditions.append('heart_disease')
        
        # Check for obesity/weight issues
        weight_keywords = ['obesity', 'overweight', 'high bmi', 'weight management']
        if any(keyword in combined_text for keyword in weight_keywords):
            identified_conditions.append('obesity')
        
        # Check for high cholesterol
        cholesterol_keywords = ['cholesterol', 'hyperlipidemia', 'ldl', 'triglycerides']
        if any(keyword in combined_text for keyword in cholesterol_keywords):
            if 'heart_disease' not in identified_conditions:
                identified_conditions.append('heart_disease')  # Treat high cholesterol as heart disease risk
        
        return identified_conditions
    
    def generate_comprehensive_plan(
        self, 
        conditions: List[str], 
        lab_values: List[str] = None,
        patient_age: int = None,
        patient_sex: str = None,
        current_medications: List[str] = None
    ) -> ComprehensiveLifestylePlan:
        """Generate a comprehensive lifestyle plan based on health conditions"""
        
        # Analyze conditions to identify relevant health issues
        identified_conditions = self.analyze_conditions(conditions, lab_values)
        
        # Generate diet recommendations
        diet_recs = []
        for condition in identified_conditions:
            if condition in self.diet_recommendations:
                diet_recs.append(self.diet_recommendations[condition])
        
        # Generate exercise recommendations
        exercise_recs = []
        for condition in identified_conditions:
            if condition in self.exercise_recommendations:
                exercise_recs.append(self.exercise_recommendations[condition])
        
        # Always include general lifestyle recommendations
        lifestyle_recs = [
            self.lifestyle_recommendations['stress_management'],
            self.lifestyle_recommendations['sleep_hygiene'],
            self.lifestyle_recommendations['hydration']
        ]
        
        # Add smoking cessation if relevant
        if any('smok' in condition.lower() for condition in conditions):
            lifestyle_recs.append(self.lifestyle_recommendations['smoking_cessation'])
        
        # Generate priority actions based on conditions
        priority_actions = self._generate_priority_actions(identified_conditions, patient_age)
        
        # Generate monitoring suggestions
        monitoring_suggestions = self._generate_monitoring_suggestions(identified_conditions)
        
        # Generate professional consultation recommendations
        professional_consultations = self._generate_consultation_recommendations(identified_conditions)
        
        return ComprehensiveLifestylePlan(
            diet_recommendations=diet_recs,
            exercise_recommendations=exercise_recs,
            lifestyle_recommendations=lifestyle_recs,
            priority_actions=priority_actions,
            monitoring_suggestions=monitoring_suggestions,
            professional_consultations=professional_consultations
        )
    
    def _generate_priority_actions(self, conditions: List[str], patient_age: int = None) -> List[str]:
        """Generate priority actions based on conditions"""
        actions = []
        
        if 'diabetes' in conditions:
            actions.extend([
                "Start blood glucose monitoring routine",
                "Schedule appointment with endocrinologist or diabetes educator",
                "Begin carbohydrate counting and meal planning"
            ])
        
        if 'hypertension' in conditions:
            actions.extend([
                "Begin daily blood pressure monitoring",
                "Implement DASH diet principles immediately",
                "Start gentle exercise program with walking"
            ])
        
        if 'heart_disease' in conditions:
            actions.extend([
                "Obtain medical clearance before starting exercise",
                "Schedule cardiology follow-up",
                "Begin heart-healthy Mediterranean diet"
            ])
        
        if 'obesity' in conditions:
            actions.extend([
                "Set realistic weight loss goals (1-2 lbs/week)",
                "Start food diary to track eating patterns",
                "Begin low-impact exercise program"
            ])
        
        # Age-specific priorities
        if patient_age and patient_age > 65:
            actions.append("Focus on fall prevention and balance exercises")
        
        return actions
    
    def _generate_monitoring_suggestions(self, conditions: List[str]) -> List[str]:
        """Generate monitoring suggestions based on conditions"""
        monitoring = []
        
        if 'diabetes' in conditions:
            monitoring.extend([
                "Daily blood glucose monitoring",
                "Weekly weight checks",
                "Quarterly HbA1c testing",
                "Annual eye and foot examinations"
            ])
        
        if 'hypertension' in conditions:
            monitoring.extend([
                "Daily blood pressure readings",
                "Weekly weight monitoring",
                "Monthly medication review",
                "Regular kidney function tests"
            ])
        
        if 'heart_disease' in conditions:
            monitoring.extend([
                "Regular blood pressure and heart rate monitoring",
                "Cholesterol level checks every 3-6 months",
                "Weight monitoring for fluid retention",
                "Symptom tracking (chest pain, shortness of breath)"
            ])
        
        if 'obesity' in conditions:
            monitoring.extend([
                "Weekly weight and measurement tracking",
                "Monthly progress photos",
                "Food diary maintenance",
                "Exercise log keeping"
            ])
        
        return monitoring
    
    def _generate_consultation_recommendations(self, conditions: List[str]) -> List[str]:
        """Generate professional consultation recommendations"""
        consultations = []
        
        if 'diabetes' in conditions:
            consultations.extend([
                "Endocrinologist for diabetes management",
                "Certified diabetes educator",
                "Registered dietitian for meal planning",
                "Podiatrist for foot care"
            ])
        
        if 'hypertension' in conditions:
            consultations.extend([
                "Cardiologist for blood pressure management",
                "Registered dietitian for DASH diet guidance",
                "Exercise physiologist for safe exercise program"
            ])
        
        if 'heart_disease' in conditions:
            consultations.extend([
                "Cardiologist for ongoing care",
                "Cardiac rehabilitation specialist",
                "Registered dietitian for heart-healthy nutrition",
                "Mental health counselor for stress management"
            ])
        
        if 'obesity' in conditions:
            consultations.extend([
                "Registered dietitian for weight management",
                "Exercise physiologist or personal trainer",
                "Behavioral therapist for eating habits",
                "Bariatric specialist if BMI > 40"
            ])
        
        return consultations

# Create singleton instance
lifestyle_engine = LifestyleRecommendationEngine()
