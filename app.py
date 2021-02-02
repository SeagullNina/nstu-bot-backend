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
            'examples': ['Привет', 'Добрый день', 'Хай', 'Хеллоу', 'Добрый вечер', 'Вечер добрый', 'Вечер в хату', 'Здравствуйте', 'Доброй ночи', 'Доброе утро'],
            'responses': ['Здравствуйте! Я - Коннор. Прислан Киберлайф', 'Добрый день. Моё имя Коннор. Я из Киберлайф']
        },
        'bye': {
             'examples': ['Пока', 'До свидания', 'Пока, Коннор', 'До свидания, Коннор', 'Бывай', 'Досвидос'],
            'responses': ['Если что, я буду ждать Вас здесь', 'До скорой встречи', 'Было приятно поговорить. Приходите ещё']
         },
        'personalLikes': {
            'examples': ['Ты любишь читать?', 'Ты любишь музыку?', 'Ты любишь кино?', 'Ты любишь играть?', 'Ты любишь рок?', 'Ты любишь компьютерные игры?'],
            'responses': ['Да, люблю. В этом есть какая-то... жизнь', 'Да', 'Мне нравится заниматься этим в свободное от расследований время']
        },
        'professionalDeeds': {
            'examples': ['Как дела на работе?', 'Как дела с расследованиями?', 'Нашел девиантов?'],
            'responses': ['Дела идут хорошо, спасибо', 'Мы в шаге от раскрытия дела', 'Скоро все девианты будут обнаружены']
        },
        'cyberlife': {
            'examples': ['Расскажи про Киберлайф', 'Что такое Киберлайф', 'Киберлайф это who', 'Киберлайф это'],
            'responses': ['«Киберлайф» — компания, производящая андроидов в широком масштабе. Основана Элайджей Камски в 2018 году.', 'Не поверю, что вы не знаете о Киберлайф', 'Киберлайф - компания, которая произвела меня. И многих других андроидов']
        },
        'whoAreYou': {
            'examples': ['Кто ты?', 'Кто ты такой?'],
            'responses': ['Я - андроид из Киберлайф', 'Я - андроид модели RK800']
        },
        'Hank': {
            'examples': ['Как у тебя с Хэнком?', 'Как ты относишься к Хэнку?', 'Вы с Хэнком друзья?', 'Как дела у Хэнка?', 'Что ты можешь сказать про Хэнка', 'Как дела у мистера Андерсона?'],
            'responses': ['Мы с мистером Андерсоном хорошие коллеги... я бы даже сказал, что мы можем стать друзьями', 'Мы сейчас расследуем новое дело. Мы отличная команда']
        },
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
            if abs(len(example) - len(replica)) / len(example) < 0.5:
                distance = nltk.edit_distance(replica, example)
                if len(example) and distance / len(example) < 0.5:
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
