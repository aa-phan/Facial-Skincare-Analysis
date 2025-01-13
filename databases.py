import os
from dotenv import load_dotenv
from supabase import create_client, Client


def initialize_supabase():
    load_dotenv()
    url = os.getenv('SUPABASE_URL')
    key = os.getenv('SUPABASE_API_KEY')
    return create_client(url, key)

def find_products(ingredients=None, exclude_ingredients=None, max_price=None, combination=None, dry=None, normal=None, oily=None, sensitive=None, product_type=None):
    """
    Query skincare products based on multiple filter criteria.
    
    Parameters:
    - ingredients (str): Ingredient to search for
    - combination (bool): True for combination skin products
    - dry (bool): True for dry skin products
    - normal (bool): True for normal skin products
    - oily (bool): True for oily skin products
    - sensitive (bool): True for sensitive skin products
    - product_type (str): Type of product to search for
    
    Returns:
    - list: Matching skincare products
    """
    # Query data from the combined_skincare_products view
    supabase = initialize_supabase()
    query = supabase.table("combined_skincare_products").select("*")
    if ingredients:
        query = query.ilike("ingredients", f"%{ingredients}%")
    if exclude_ingredients:
        query = query.not_.ilike("ingredients", f"%{exclude_ingredients}%")
    if max_price is not None:
        query=query.lte("price", max_price)
    if combination is not None:
        query = query.eq("combination", combination)
    if dry is not None:
        query = query.eq("dry", dry)
    if normal is not None:
        query = query.eq("normal", normal)
    if oily is not None:
        query = query.eq("oily", oily)
    if sensitive is not None:
        query = query.eq("sensitive", sensitive)
    if product_type:
        query = query.ilike("product_type", f"%{product_type}%")
        
    response = query.execute()
    return response.data



