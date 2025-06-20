from langchain_core.tools import tool
from langchain_groq import ChatGroq

llm = ChatGroq(model="llama-3.1-8b-instant")

@tool
def calculate_discount(price: float, discount_percentage: float) -> float:
    """
    The BaseTool interface used by the tool decorator
        LangChain uses a class called BaseTool as the blueprint for all tools. It defines the core attributes and methods that all tools should have:
        
        name: A descriptive name for the tool (e.g., “weather_lookup”, “multiply_numbers”).
        
        description: A natural language explanation of what the tool does (e.g., “Retrieves the current weather conditions for a given city.”).
        
        args: A definition of the expected inputs for the tool (e.g., a “city” string).
        
        invoke(args): This method is called to execute the tool.
    Calculates the final price after applying a discount.

    Args:
        price (float): The original price of the item.
        discount_percentage (float): The discount percentage (e.g., 20 for 20%).

    Returns:
        float: The final price after the discount is applied.
    """
    if not (0 <= discount_percentage <= 100):
        raise ValueError("Discount percentage must be between 0 and 100")

    discount_amount = price * (discount_percentage / 100)
    final_price = price - discount_amount
    return final_price
    
print(calculate_discount.name)
print(calculate_discount.description)
print(calculate_discount.args)
print(calculate_discount.invoke({"price":100, "discount_percentage": 15}))

llm_with_tools = llm.bind_tools([calculate_discount])

hello_world = llm_with_tools.invoke("Hello world!")
print("Content:", hello_world.content,'\n')

# content attribute is empty and the tool_calls attribute has been set.
result = llm_with_tools.invoke("What is the price of an item that costs $100 after a 20% discount?")
print("Content:", result.content)

# Assuming we have made the call like we did before
print(result.tool_calls)
# This would output something like:
{'name': 'calculate_discount', 'args': {'price': 100.0, 'discount_percentage': 20.0}, 'id': 'abcdef', 'type': 'tool_call'}

args = result.tool_calls[0]['args']
print(calculate_discount.invoke(args))
