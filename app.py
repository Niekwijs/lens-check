import random
import string
import os

from flask import Flask, render_template, request

import numpy as np
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model("my_model")


class model_input:
    pass




_final_input = [0] * 7

questions =[
    [
        "Age",
        "Age of the patient: (1) young, (2) pre-presbyopic, (3) presbyopic",
        33

    ],
    [
        "Spectacle",
        "Spectacle prescription: (1) myope, (2) hypermetrope",
        66

    ],

    [
        "Tear production",
        "Tear production rate: (1) reduced, (2) normal",
        99

    ]
]

_q_input = [None] * 3

answers =[
    [
        "young",
        "pre-presbyopic",
        "presbyopic"
     ],
    [
        "myope",
        "hypermetrope"
     ],
    [
        "reduced",
        "normal"
    ]
]



@app.get('/')
def index():
    return render_template('index.html')


def handle_answer(session_id, question_id, answer):
    if question_id == 1:
        _q_input[0] = answer
    elif question_id == 2:
        _q_input[1] = answer
    elif question_id == 3:
        _q_input[2] = answer
    print(_q_input)

def create_final_input():
    _final_input = [0, 0, 0, 0, 0, 0, 0]
    if _q_input[0] == "young":
        _final_input[0] = 1
    elif _q_input[0] == "pre-presbyopic":
        _final_input[1] = 1
    elif _q_input[0] == "presbyopic":
        _final_input[2] = 1
    if _q_input[1] == "myope":
        _final_input[3] = 1
    elif _q_input[1] == "hypermetrope":
        _final_input[4] = 1
    if _q_input[2] == "reduced":
        _final_input[5] = 1
    elif _q_input[2] == "normal":
        _final_input[6] = 1
    print(_final_input)

@app.get("/prediction")
def prediction_get():
    # ['young', 'pre-presbyopic', 'presbyopic', 'myope', 'hypermetrope', 'tear production rate reduced',
    #  'tear production rate normal']
    create_final_input()
    pred =(100 * model.predict([_final_input]))
    pred = pred.round(2)
    return pred


@app.get('/question')
def answer_question():
    question_id = request.args.get("question_id", default=0, type=int)
    # Not truely super duper random, but same as above
    _session_id_code = "".join(
        random.choice(string.ascii_uppercase + string.digits) for _ in range(16))
    session_id = request.args.get("session_id", type=str, default=_session_id_code)

    if question_id != 0:
        handle_answer(session_id, question_id, request.args.get("radio_answer", type=str, default=""))
    if question_id >= len(questions):
        pred = prediction_get()
        return render_template("prediction-page.html",
                               prediction_hard= pred[0][0] ,
                               prediction_soft= pred[0][1] ,
                               prediction_glasses= pred[0][2])

    question = questions[question_id]
    print(f"New req: {question_id}")
    return render_template("genetic-page.html",
                           question=question,
                           question_id=question_id,
                           questions=questions,
                           answers=answers,
                           session_id=session_id,
                           len=len,
                           **{fun. __name__ :fun for fun in [enumerate, str] })



if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))