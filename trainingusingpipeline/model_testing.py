from model_training import train_model
import numpy as np

def prediction():

    model = train_model()

    ask_experience = input('Enter Your Experience: ').lower()
    total_score = int(input('test_score(out of 10): '))
    interview_score = int(input('interview_score(out of 10): '))



    value = np.array([ask_experience, total_score, interview_score], dtype=object).reshape(1, 3)
    predict = model.predict(value)


    print(f'Your Expected Salary: {round(predict[0])}$')


prediction()