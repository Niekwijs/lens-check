import base64
import io
import random
import string
import os
import IPython
import matplotlib as plt
import shap
import pandas as pd
from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

app = Flask(__name__)
model = tf.keras.models.load_model("my_model")


class model_input:
    pass


_final_input = [0] * 7

prev_image = None

questions = [
    [
        "Age",
        "Please select your age below (young 0-40 , pre-presbyopic  40-50, presbyopic 50> .",
        33

    ],
    [
        "Spectacle",
        "Please select you Spectacle prescription below",
        66

    ],

    [
        "Tear production",
        "Please select your Tear production rate below.",
        99

    ]
]

_q_input = [None] * 3

answers = [
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


@app.get('/about')
def about_get():
    return render_template('about-page.html')


@app.get('/collection')
def collection_get():
    return render_template('collection-page.html')


def prediction_get():
    # ['young', 'pre-presbyopic', 'presbyopic', 'myope', 'hypermetrope', 'tear production rate reduced',
    #  'tear production rate normal']
    create_final_input()
    y_prob = model.predict([_final_input])

    # b64_img = render_shap_explainer(y_prob, [_final_input], explainer)

    pred = y_prob * 100
    pred = pred.round(2)

    return pred


@app.get("/about-prediction")
def about_prediction_get():
    return render_template('about-prediction.html', image=prev_image)


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
                               prediction_hard=pred[0][0],
                               prediction_soft=pred[0][1],
                               prediction_glasses=pred[0][2])

    question = questions[question_id]
    print(f"New req: {question_id}")
    return render_template("genetic-page.html",
                           question=question,
                           question_id=question_id,
                           questions=questions,
                           answers=answers,
                           session_id=session_id,
                           len=len,
                           **{fun.__name__: fun for fun in [enumerate, str]})


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


# def render_shap_explainer(y_proba, X_questions, explainer):
#     plt.pyplot.ioff()
#     fig = plt.pyplot.figure()
#     fig.legend(["soft lenses", "hard lenses", "glasses"])
#
#     df_pred = pd.DataFrame(X_questions)
#     shap_values = explainer.shap_values(df_pred)
#
#     shap.summary_plot(shap_values, df_pred, plot_type="bar", show=None,
#                       feature_names=['young', 'pre-presbyopic', 'presbyopic', 'myope', 'hypermetrope',
#                                      'tear production rate reduced', 'tear production rate normal'])
#
#     inmem_file = io.BytesIO()
#     fig.savefig(inmem_file, format="png")
#     return base64.b64encode(inmem_file.getbuffer()).decode("ascii")


# def load_dataset(dataset_path='./dataset/lenses-comp.csv'):
#     df = pd.read_csv(dataset_path)
#
#     X, y = df[['young', 'pre-presbyopic', 'presbyopic', 'myope', 'hypermetrope', 'tear production rate reduced',
#                'tear production rate normal']], df["result"]
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#
#     return df, X_train, X_test, y_train, y_test


if __name__ == '__main__':
    # df, X_train, X_test, y_train, y_test = load_dataset()
    # summary = shap.kmeans(X_test, 20)
    # print('hallo')
    # explainer = shap.KernelExplainer(model.predict, summary)

    app.run(debug=True, port=os.getenv("PORT", default=5000))
