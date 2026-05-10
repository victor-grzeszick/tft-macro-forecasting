import os 
import dotenv 
import fredapi 

dotenv.load_dotenv()

api_key = os.getenv("FRED_API_KEY")

fred = fredapi.Fred(api_key=api_key)

result = fred.get_series("GDPC1", observation_start="2020-01-01")

print(result.head())