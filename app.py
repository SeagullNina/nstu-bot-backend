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

bot_state = {
    "next_step": None,
    "variables": {
        "result": 0
    },
    "saveReplicaToVariable": None,
}


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
        'numbeDekanatAvtf': {
            'examples': ['Номер', 'Как позвонить в деканат АВТФ'],
            'responses': ['Номе деканата АВТФ - 8 383 346-11-53']
        },
        'firstQuestion': {
            'examples': ['1', 'Тест'],
            'responses': ['Если бы на свете существовали только две профессии, какую работу вы бы предпочли из двух? 1. Ухаживать за животными. 2. Обслуживать машины, приборы.'],
            'next_step': 'testSummary',
            "save_variable": 'firstQuestion',
        },
        'secondQuestion': {
            'examples': ['1', 'Тест'],
            'responses': ['Какую работу вы предпочтёте? 1. Помогать больным людям. 2. Составлять таблицы, схемы, программы вычислительных машин.'],
            'next_step': 'thirdQuestion',
            "save_variable": 'secondQuestion',
        },
        'thirdQuestion': {
            'responses': ['Какую работу вы предпочтёте? 1. Следить за качеством книжных иллюстраций, плакатов, художественных открыток. 2. Следить за состоянием, развитием растений.'],
            'next_step': 'fourthQuestion',
            "save_variable": 'thirdQuestion',
        },
        'fourthQuestion': {
            'responses': ['Какую работу вы предпочтёте? 1. Обрабатывать материалы (дерево, ткань, металл, пластмассу) 2. Доводить товары до потребителя (рекламировать, продавать)'],
            'next_step': 'fifthQuestion',
            "save_variable": 'fourthQuestion',
        },
        'fifthQuestion': {
            'responses': ['Какую работу вы предпочтёте? 1. Обсуждать научно популярные книги, статьи 2. Обсуждать художественные книги'],
            'next_step': 'sixthQuestion',
            "save_variable": 'fifthQuestion',
        },
        'sixthQuestion': {
            'responses': ['Какую работу вы предпочтёте? 1. Выращивать молодняк животных какой-либо породы 2. Тренировать сверстников (или младших) в выполнении каких-либо действий (трудовых, учебных, спортивных)'],
            'next_step': 'seventhQuestion',
            "save_variable": 'sixthQuestion',
        },
        'seventhQuestion': {
            'responses': ['Какую работу вы предпочтёте? 1. Копировать рисунки, изображения, настраивать музыкальные инструменты 2. Управлять каким-либо грузовым, подъемным транспортным средством (подъемным краном, трактором, тепловозом и др.)'],
            'next_step': 'eighthQuestion',
            "save_variable": 'seventhQuestion',
        },
        'eighthQuestion': {
            'responses': ['Какую работу вы предпочтёте? 1. Сообщать, разъяснять людям нужные им сведения (в справочном бюро, на экскурсии и т.п.) 2. Художественно оформлять выставки, витрины, участвовать в подготовке пьес, концертов'],
            'next_step': 'ninethQuestion',
            "save_variable": 'eightQuestion',
        },
        'ninethQuestion': {
            'responses': ['Какую работу вы предпочтёте? 1. Ремонтировать изделия, вещи (одежду, технику), жилище 2. Искать и исправлять ошибки в текстах, таблицах, рисунках'],
            'next_step': 'tenthQuestion',
            "save_variable": 'ninethQuestion',
        },
        'tenthQuestion': {
            'responses': ['Какую работу вы предпочтёте? 1. Лечить животных 2. Выполнять вычисления, расчеты'],
            'next_step': 'testSummary',
            "save_variable": 'tenthQuestion',
        },
        'testSummary': {
            "get_variables": ["resultText"],
            "responses": ["{resultText}"],
        }
        },
    'failure_phrases': [
        'Непонятно. Перефразируй, пожалуйста',
        'Я еще только учусь. Не умею на такое отвечать'
    ]
}

def clear_text(text):
    text = text.lower()
    text = "".join(char for char in text if char in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя -')
    return text

def classify_intent(replica):
    replica = clear_text(replica)

    for intent, intent_data in BOT_CONFIG['intents'].items():
        if 'examples' not in intent_data:
            continue

        for example in intent_data['examples']:
            example = clear_text(example)

            distance = nltk.edit_distance(replica, example)

            if (len(example) != 0 and distance / len(example)) < 0.3:
                return intent

def get_answer_by_intent(intent):
    if intent in BOT_CONFIG["intents"]:
        responses = BOT_CONFIG["intents"][intent]["responses"]

        if "next_step" in BOT_CONFIG["intents"][intent]:
            bot_state["next_step"] = BOT_CONFIG["intents"][intent]["next_step"]

        # Если в кейсе есть запрос на сохранение(save_variable) записываем имя переменной что бы
        # сохранить след реплику в хранилище
        if 'save_variable' in BOT_CONFIG["intents"][intent]:
            bot_state["saveReplicaToVariable"] = BOT_CONFIG["intents"][intent]["save_variable"]
            global result
            result = BOT_CONFIG["intents"][intent]["save_variable"]

        # Если в кейсе есть запрос на получение(get_variables), проходимся по строке и заменяем в строке
        # плейсхолдеры на значение переменных
        if 'get_variables' in BOT_CONFIG["intents"][intent]:
            response = random.choice(responses)

            for variable in BOT_CONFIG["intents"][intent]["get_variables"]:
                response = response.replace("{" + variable + "}", bot_state["variables"][variable])


            return response

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
    failure_phrases = BOT_CONFIG["failure_phrases"]
    return random.choice(failure_phrases)

def summator(ball):
    ball += ball
    return ball

def bot(replica):

    global result
    result = replica
    if bot_state["saveReplicaToVariable"] != None:
        # Сохраняем переменную
        var_name = bot_state["saveReplicaToVariable"]
        bot_state["variables"][var_name] = replica
        bot_state["variables"]["result"] = summator(bot_state["variables"]["result"])
        if bot_state["variables"]["result"] < 5:
            bot_state["variables"]["resultText"] = "Вы машина"
        if bot_state["variables"]["result"] >= 5:
            bot_state["variables"]["resultText"] = "Вы пес"

        bot_state["saveReplicaToVariable"] = None

        # Загружаем след шаг если он есть
    if bot_state["next_step"] != None:
        intent = bot_state["next_step"]
        # очищаем next_step чтобы не попасть в цикличность
        bot_state["next_step"] = None
        return get_answer_by_intent(intent)

        # NLU
    intent = classify_intent(replica)

    # Получение ответа

    # Правила
    if intent:
        answer = get_answer_by_intent(intent)

        if answer:
            return answer

    # генеративная модель
    answer = generate_answer(replica)
    if answer:
        return answer

    # заглушка
    answer = get_stub()
    return answer

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.route('/api/messages', methods=['POST'])
def message():
    messageText = request.json['message']
    answer = bot(messageText)
    return jsonify(message=answer)
