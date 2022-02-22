import requests

row = {
    "age": 47,
    "workclass": "Private",
    "fnlwgt": 51835,
    "education": "Prof-school",
    "education_num": 15,
    "marital_status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Wife",
    "race": "White",
    "sex": "Female",
    "capital_gain": 0,
    "capital_loss": 1902,
    "hours_per_week": 60,
    "native_country": "Honduras"
}

response = requests.post(
    url='https://udacity-rushikesh-project-3.herokuapp.com/inference',
    json=row
)

print(response.status_code)
print(response.json())