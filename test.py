import os
from dotenv import load_dotenv
load_dotenv('.env_template')
print(os.environ.get('DB_DESTINATION_HOST'))
print("Yes")