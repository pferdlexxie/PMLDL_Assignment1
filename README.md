# PMLDL_Assignment1

This project provides a model for smile classification. It predicts if the person in the provided photo is smiling or not.

#### Start with 
````
git clone https://github.com/pferdlexxie/PMLDL_Assignment1
````
````
cd PMLDL_Assignment1
````

####  To run docker navigate to the directory where docker-compose.yaml is lockated 
````
cd code/deployment
````

####  Build virtual environment if necessary 
````
python -m venv venv
````
````
venv\Scripts\activate   
````

####  Run docker 
````
docker-compose build
```` 
````
docker-compose up
````

### Access the model
FastAPI: \
http://localhost:8000/docs to view the API doc.

Streamlit: \
http://localhost:8501 to access web app.
