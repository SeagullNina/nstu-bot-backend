from flask import Flask
from flask import request
from flask_cors import CORS
import random
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from IPython import get_ipython
from flask import jsonify

BOT_CONFIG = {
    'intents': {
        'hello': {
            'examples': ['Привет', 'Добрый день', 'Хай', 'Хеллоу', 'Йоу', 'Добрый вечер', 'Вечер добрый', 'Здравствуйте', 'Доброй ночи', 'Доброе утро'],
            'responses': ['Здравствуйте! Я - чат-бот НГТУ. Готов ответить на Ваши вопросы']
        },
        'bye': {
             'examples': ['Пока', 'До свидания', 'Бывай'],
            'responses': ['Если что, я буду ждать Вас здесь', 'До скорой встречи!', 'Было приятно поговорить. Приходите ещё']
         },
        'documents': {
            'examples': ['Какие документы требуются для поступления?', 'Документы для поступления', 'Какие документы принести', 'Какие документы нужны', 'Документы'],
            'responses': ['При поступлении в вуз вам необходимо будет заполнить заявление о поступлении. К заявлению нужно приложить: паспорт или другой документ, удостоверяющий личность и гражданство абитуриента; документ о предыдущем полученном образовании: аттестат об окончании школы, диплом о начальном, среднем или высшем профессиональном образовании; информацию о результатах ЕГЭ, если вы его сдавали; 2 фотографии, если при поступлении вы будете проходить дополнительные вступительные испытания; приписное свидетельство или военный билет (при наличии); медицинскую справку формы 086/у — для медицинских, педагогических и некоторых других специальностей и направлений; если вместо вас документы будет подавать ваш представитель, дополнительно понадобится нотариально заверенная доверенность и документ, удостоверяющий его личность; если на момент подачи документов вам будет меньше 18 лет, возьмите с собой форму согласия на обработку ваших персональных данных, подписанную родителем или опекуном, — без нее документы не примут. Скачайте форму на сайте вуза или попросите сотрудников приемной комиссии выслать ее вам по электронной почте; документы, подтверждающие индивидуальные достижения; документы, подтверждающие особые права и льготы.']
        },
        'wave': {
            'examples': ['Что такое волна зачисления', 'Волна зачисления'],
            'responses': ['Зачисление на бюджет происходит в первую и вторую волну. На первом этапе заполняется – 80 % от общего количества бесплатных мест, на втором – оставшиеся 20 %.']
        },
        'lhot_places': {
            'examples': ['Что такое льготные места', 'Поступление вне конкурса'],
            'responses': [' Вне конкурса могут быть зачислены сироты, инвалиды при представлении в Приёмную комиссию соответствующих подтверждающих документов.']
        },
        'army': {
            'examples': ['Военная кафедра', 'У вас есть военная кафедра'],
            'responses': ['У нас нет военной кафедры.']
        },
        'howManyYearsEGE': {
            'examples': ['Сколько лет действуют результаты ЕГЭ?'],
            'responses': ['При приеме на обучение по программам бакалавриата и программам специалитета, результаты ЕГЭ действительны четыре года, следующих за годом получения таких результатов.']
        },
        'alternative EGE': {
            'examples': ['Есть ли альтернатива ЕГЭ?', 'Могу ли я сдавать внутренние экзамены вместо ЕГЭ?', 'Можно ли не сдавать ЕГЭ', 'Не сдавать ЕГЭ'],
            'responses': ['В порядке исключения это могут разрешить пяти категориям абитуриентов: иностранным гражданам; абитуриентам, имеющим инвалидность; тем, кто проходил итоговую государственную аттестацию не в форме ЕГЭ, либо проходил аттестационные процедуры в иностранных образовательных учреждениях (в течение не более чем 1 года до дня приема документов); абитуриентам, проходившим обучение на базе профессионального образования; абитуриентам с высшим образованием. В остальных случаях прием ведется только по результатам ЕГЭ. ']
        },
        'numberDekanatAvtf': {
            'examples': ['Номер деканата АВТФ', 'Как позвонить в деканат АВТФ'],
            'responses': ['Номер деканата АВТФ - 8 383 346-11-53']
        },
        'minBallsRussian': {
            'examples': ['Минимальные баллы русский язык', 'Минимальные баллы по русскому языку'],
            'responses': ['Минимальный балл по русскому языку в 2021 году - 40 баллов']
        },
        'minBallsMath': {
            'examples': ['Минимальные баллы математика', 'Минимальные баллы по математике'],
            'responses': ['Минимальный балл по математике в 2021 году - 39 баллов']
        },
        'minBallsIKT': {
            'examples': ['Минимальные баллы информатика', 'Минимальные баллы по информатике'],
            'responses': ['Минимальный балл по информатике в 2021 году - 44 балла']
        },
        'minBallsPhysics': {
            'examples': ['Минимальные баллы физика', 'Минимальные баллы по физике'],
            'responses': ['Минимальный балл по физике в 2021 году - 39 баллов']
        },
        'minBallsChemistry': {
            'examples': ['Минимальные баллы химия', 'Минимальные баллы по химии'],
            'responses': ['Минимальный балл по химии в 2021 году - 39 баллов']
        }
        },
    'failure_phrases': [
        'Непонятно. Перефразируй, пожалуйста',
        'Я еще только учусь. Не умею на такое отвечать'
    ]
}

texts = []
intent_names = []

for intent, intent_data in BOT_CONFIG['intents'].items():
    for example in intent_data['examples']:
        texts.append(example)
        intent_names.append(intent)


vectorizer = TfidfVectorizer(ngram_range=(2, 4), analyzer='char')
X = vectorizer.fit_transform(texts)
clf = LinearSVC()
clf.fit(X, intent_names)


def classify_intent(replica):
    intent = clf.predict(vectorizer.transform([replica]))[0]

    examples = BOT_CONFIG['intents'][intent]['examples']
    for example in examples:
        example = clear_text(example)
        if len(example) > 0:
            if abs(len(example) - len(replica)) / len(example) < 0.7:
                distance = nltk.edit_distance(replica, example)
                if len(example) and distance / len(example) < 0.7:
                    return intent


def get_answer_by_intent(intent):
    if intent in BOT_CONFIG['intents']:
        responses = BOT_CONFIG['intents'][intent]['responses']
        return random.choice(responses)


with open('dialogues.txt', encoding="utf-8") as dialogues_file:
    dialogues_text = dialogues_file.read()
dialogues = dialogues_text.split('\n\n')


def clear_text(text):
    text = text.lower()
    text = ''.join(char for char in text if char in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя -')
    return text


dataset = []
questions = set()

for dialogue in dialogues:
    replicas = dialogue.split('\n')
    replicas = replicas[:2]

    if len(replicas) == 2:
        question, answer = replicas
        question = clear_text(question[2:])
        answer = answer[2:]

        if len(question) > 0 and question not in questions:
            questions.add(question)
            dataset.append([question, answer])


dataset_by_word = {}

for question, answer in dataset:
    words = question.split(' ')
    for word in words:
        if word not in dataset_by_word:
            dataset_by_word[word] = []
        dataset_by_word[word].append([question, answer])

dataset_by_word_filtered = {}
for word, word_dataset in dataset_by_word.items():
    word_dataset.sort(key=lambda pair: len(pair[0]))
    dataset_by_word_filtered[word] = word_dataset[:1000]

def generate_answer(replica):
    replica = clear_text(replica)
    if not replica:
        return

    words = set(replica.split(' '))
    words_dataset = []
    for word in words:
        if word in dataset_by_word_filtered:
            word_dataset = dataset_by_word_filtered[word]
            words_dataset += word_dataset

    results = []  # [[question, answer, distance], ...]
    for question, answer in words_dataset:
        if abs(len(question) - len(replica)) / len(question) < 0.2:
            distance = nltk.edit_distance(replica, question)
            if distance / len(question) < 0.2:
                results.append([question, answer, distance])

    question, answer, distance = min(results, key=lambda three: three[2])
    return answer


def get_stub():
    failure_phrases = BOT_CONFIG['failure_phrases']
    return random.choice(failure_phrases)


stats = {'intents': 0, 'generative': 0, 'stubs': 0}


def bot(replica):
    # NLU
    intent = classify_intent(replica)

    # Получение ответа

    # правила
    if intent:
        answer = get_answer_by_intent(intent)
        if answer:
            stats['intents'] += 1
            return answer

    # генеративная модель
    answer = generate_answer(replica)
    if answer:
        stats['generative'] += 1
        return answer

    # заглушка
    answer = get_stub()
    stats['stubs'] += 1
    return answer

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.route('/api/messages', methods=['POST'])
def message():
    messageText = request.json['message']
    answer = bot(messageText)
    return jsonify(message=answer)
