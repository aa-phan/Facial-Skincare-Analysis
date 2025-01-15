import os
from openai import OpenAI
from dotenv import load_dotenv

def rag_ingredients(prediction=None):
    if prediction is None or not isinstance(prediction, int) or prediction < 0 or prediction > 3:
        return -1 #invalid input
    else:
        load_dotenv()
        LLM_key = os.getenv('NEBIUS_API_KEY')
        client = OpenAI(
            base_url="https://api.studio.nebius.ai/v1/",
            api_key = LLM_key
        )
        analysis_text = """
        Find me one ingredient that is good at treating acne based on the condition severity that I provide. 
        Note these interpretations of scores for what severity of acne they indicate:
        0: Mild severity
        1: Moderate severity
        2: High severity
        3: Extreme severity
        """
        analysis_text+=str(prediction)
        
        try:
            completion = client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-70B-Instruct-fast",
                messages=[
                    {"role": "system", "content": "You are a skincare expert looking for three ingredients that can best treat the level of acne severity provided to you. Please only return a single ingredient like this: ingredient. There should be no additional text, whitespace, commas, or numbers."},
                    {"role": "user", "content": analysis_text}
                ],
                temperature=1.0,
                max_tokens=512,
                top_p=1.0,
                presence_penalty=0.7
                
            )

            ingredients_string = completion.choices[0].message.content.strip()
            ingredients_array = [ingredient.strip() for ingredient in ingredients_string.split(',')]

            #print("\nIngredients list:")
            #print("-" * 50)
            print(ingredients_array)

            return ingredients_array

        except Exception as e:
            print(f"Error getting LLM analysis: {str(e)}")
            return None
        
def analyze_pareto_products(pareto_results):
    """
    Analyze Pareto-optimal products using Nebius AI to provide insights about
    which products are optimized for different objectives.
    
    Args:
        pareto_results: List of products with their scores from pareto_front()
    """
    load_dotenv()
    LLM_key = os.getenv('NEBIUS_API_KEY')
    client = OpenAI(
        base_url="https://api.studio.nebius.ai/v1/",
        api_key = LLM_key
    )
    
    # Prepare the analysis prompt
    analysis_text = "Analyze these Pareto-optimal skincare products:\n\n"
    for product in pareto_results:
        analysis_text += f"Product: {product['name']}\n"
        analysis_text += "Scores (higher is better, scale 0-1):\n"
        for objective, score in product['scores'].items():
            analysis_text += f"- {objective}: {score:.2f}\n"
        analysis_text += "\n"
    
    analysis_text += """Based on the provided scores, please provide key recommendations for different user priorities in the following format, where only text between the /// and ||| limiters indicate the format of your response. Do not include those limiters in the final output:
    ///
    **Best Overall Product:** [Product Name] (List top 2-3 scores) - Brief explanation of why it excels.

    **Best Value for Money:** [Product Name] (List top 2-3 scores) - Brief explanation of its value proposition.

    **Best Brand Alignment:** [Product Name] (List top 2-3 scores) - Explanation of brand preference match.

    **Most Compatible with Skin:** [Product Name] (List top 2-3 scores) - Explanation of skin compatibility.

    **Best Rated By Customers:** [Product Name] (List top 2-3 scores) - Explanation of positive customer sentiment.
    |||
    
    IMPORTANT: Interpret the scores as follows:

    1. Brand Score: Higher scores indicate better alignment with user's brand preferences, not necessarily stronger brand reputation.

    2. Skin Type Score: Higher scores show better compatibility with user's skin type preferences. A score of 1.0 means perfect match with requested skin types, while 0.5 might indicate compatibility with half of the requested types.

    3. Rating Score: Higher scores indicate more positive customer sentiment.
    
    4. Price Score: Lower price scores indicate worse affordability relative to the preferred price, while higher price scores indicate better affordability relative to preferred price. 
    A price score of 0.5 indicates the price is equal to the preferred price.

    Please provide your analysis keeping these interpretations in mind."""
    
    try:
        completion = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-70B-Instruct-fast",
            messages=[
                {"role": "system", "content": "You are a skincare expert analyzing Pareto-optimal product recommendations. Interpret scores where 1.0 is optimal and 0.0 is worst."},
                {"role": "user", "content": analysis_text}
            ],
            temperature=0.6,
            max_tokens=512,
            top_p=0.9
        )
        
        print("\nAI Analysis of Pareto-Optimal Products:")
        print("Note: Scores range from 0 to 1, where a higher score indicates better performance with a given trait.")
        print("-" * 50)
        print(completion.choices[0].message.content)
        
    except Exception as e:
        print(f"Error getting LLM analysis: {str(e)}")
        return None
