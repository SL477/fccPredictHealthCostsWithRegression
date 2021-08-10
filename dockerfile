#FROM python:3.8-slim-buster
FROM python:3.7
#FROM marcskovmadsen/awesome-streamlit:latest

#WORKDIR /app
#COPY requirements.txt requirements.txt
#RUN pip3 install --user -r requirements.txt

#COPY . .

# Run the app
#CMD ["python3","-m","streamlit.cli", "run", "app.py"]
#,"--server.port=3000"]
#EXPOSE 3000

# For StreamLit
#RUN mkdir -p /root/.streamlit
EXPOSE 8501
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip3 install --user -r requirements.txt
#RUN pip3 install --index-url https://google-coral.github.io/py-repo/ tflite_runtime

COPY . .

#Run
#CMD [ "streamlit", "run", "app.py" ]
CMD [ "python3", "-m", "streamlit.cli", "run", "app.py" ]