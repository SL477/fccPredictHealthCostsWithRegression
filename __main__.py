#python3 -m pip install requirements.txt
import sys 
from streamlit import cli as stcli
if __name__ == "__main__":
    print("Hello")
    sys.argv = ["streamlit", "run","fccPredictHealthCostsWithRegression/app.py"]
    sys.exit(stcli.main())
