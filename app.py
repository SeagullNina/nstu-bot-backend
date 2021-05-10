from flask import Flask
from flask import request
from flask_cors import CORS
import random
import nltk
import json
import numbers
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from IPython import get_ipython
from flask import jsonify
import requests

nstu_data = requests.get('https://api.ciu.nstu.ru/v1.0/api/abit_bot/get_plan_data').content
nstu_data_json = json.loads(nstu_data)

def find_profile(search_profile):
    for direction in nstu_data_json:
        for profile in direction['prof_list']:
            if profile['SPEC'] == search_profile:
                return direction['B_MEST']

def all_faculties():
    i = 1;
    resultWithDuplicates = []
    result = []
    res = ""
    for data in nstu_data_json:
        resultWithDuplicates.append(data['FACULTET'])
    result = set(resultWithDuplicates)
    for fac in result:
        res += str(i) + '.' + str(fac) + ';' + '\n'
        i += 1
    return res

def all_directions():
    i = 1;
    result = ""
    for data in nstu_data_json:
        for profile in data['prof_list']:
            result += str(i) + '.' + str(profile['SPEC']) + ';' + '\n'
            i += 1
    return result

def faculty(fac):
    res = ""
    if fac == 'РЭФ':
        res = "Факультет начал свою работу в 1953 году, когда НЭТИ только открыл свои двери для студентов. Сегодня, на факультете ведется разноплановая подготовка по самым передовым направлениям математики и физики, в том числе в области нанотехнологий и радиоэлектронных информационных технологий. В учебном процессе используется уникальная лабораторная база, которая не имеет аналогов за Уралом. Это позволяет студентам РЭФ в совершенстве освоить навыки работы на современном радиоэлектронном оборудовании: производственном, исследовательском и бытовом. На факультете реализуются программы для ТОП-5 мировых проектов СКИФ и С-тау фабрика, в рамках проекта Академгородок 2.0. Студенты факультета проходят практику на ведущих предприятиях России и к выпуску, большинство из них уже имеют стаж и место работы. Все профили факультета: "
    if fac == 'ФГО':
        res = "ФГО был создан в 1990 году. Сегодня это один из самых больших факультетов в университете: здесь одновременно обучается более 1000 студентов. Студенты ФГО получают фундаментальное гуманитарное образование, которое служит отличной базой для специализации в области социологии, журналистики, филологии, рекламы, психологии и зарубежного регионоведения. Большое внимание при обучении на ФГО уделяется развитию личностных компетенций и подготовке по иностранным языкам: на большинстве направлений студенты изучают как минимум два языка, а некоторые занятия ведут преподаватели-носители языка из разных стран."
    if fac == 'ФМА':
        res = 'Факультет был основан в 1953 году, когда Новосибирский электро-технический институт только открыл свои двери для студентов. На протяжении многих лет ФМА выпускает специалистов в области электроэнергетики, электротехники, автоматизациии технологических процессов и производств, робототехники, а также специалистов нефтегазовой отрасли. При подготовке специалистов особое внимание уделяется мехатронике, области знаний и компетенций, основанных на объединении узлов и технологических модулей: механики, электротехники и программного обеспечения.'
    if fac == 'ИСТ':
        res = 'Институт социальных технологий появился в НГТУ в 1995 году и сейчас здесь готовят специалистов социальной сферы: социальных работников, юристов, конфликтологов. Программа среднего профессионального образования ведет подготовку по следующим направлениям: адаптивная физкультура, программирование в социальной сфере, декоративно-прикладное искусство. Кроме того, в ИСТ есть магистратура и аспирантура в области социологических наук. Институт располагается в отдельном учебном корпусе, при строительстве которого учитывались все современные требования доступности, так как среди студентов института обучаются лица с ограниченными возможностями здоровья. На базе института расположено огромное количество лабораторий, различных творческих, художественных и спортивных студий.'
    if fac == 'ФЭН':
        res = 'На факультете реализуются практико-ориентированные программы обучения в тесном контакте с предприятиями-партнерами. Помимо сильного профессорско-преподавательского состава, обучение ведут квалифицированные специалисты в области энергетики. Студенты проходят практику на настоящих энергетических объектах, как регионального, так и федерального уровня. Представители предприятий принимают активное участие во всех образовательных и научных мероприятиях, организованных факультетов энергетики, в формировании тем выпускных квалификационных работ и полностью сопровождают их выполнение с последующим внедрением в свою деятельность.'
    if fac == 'АВТФ':
        res = 'Здесь готовят специалистов по разработке программного обеспечения, информационной безопасности, робототехнике, биотехнологиям, автоматике и другие. Более 40 учебных и исследовательских лабораторий, оснащенных современным оборудованием и новейшей вычислительной техникой — в отдельном восьмиэтажном корпусе. АВТФ — самый большой факультет НГТУ НЭТИ.'
    if fac == 'ФЛА':
        res = 'Факультет основан в 1959 году и на протяжении и уже больше полувека готовит специалистов высокого класса в области аэрокосмической, оборонной, машиностроительной промышленности, безопасности и охраны окружающей среды. Во время обучения студенты проходят практику на ведущих авиационных и машиностроительных предприятиях, в институтах СО РАН, отраслевых Научно-исследовательских институтах и конструкторских бюро. Кроме того, в студенческом конструкторском бюро ФЛА проходит конструирование и постройка реальных летательных аппаратов.'
    if fac == 'ИДО':
        res = 'В  Институте дистанционного обучения прошли обучение 6336 студентов по 40 учебным дистанционным специальностям. При дистанционной форме обучения нужно приезжать на сессию всего один раз в год. ИДО поможет вам провести любое учебное мероприятие, семинар или лекцию. К вашим услугам: смарт-доска, проектор, HD-камеры и прочее необходимое мультимедиа оборудование.'
    if fac == 'МТФ':
        res = 'Факультет был основан в 1956 году и с тех пор осуществляет подготовку специалистов в области автоматизации, исследований и разработки новых материалов и технологических процессов их производства, технологий художественной обработки материалов, а также в области разработки и эксплуатации оборудования, в том числе и для пищевого производства. Образовательные программы ориентированы на получение практических навыков, факультет активно сотрудничает с ведущими российскими и зарубежными организациями машиностроительными предприятиями. На МТФ развита общественно-научная жизнь: ежегодно проходят десятки мероприятий, а также две летние школы, организованные совместно с ведущим университетом Германии.'
    if fac == 'ФБ':
        res = 'С 1991 года факультет бизнеса ведет подготовку специалистов по менеджменту, экономике, бизнес-информатике, экономической безопасности, организации общественного питания, которые способны обеспечить системный, комплексный подход к решению проблем управления компаниями и содействовать успехам их деятельности. В рамках обучения на ФБ применяются проектный подход, кейсы, бизнес-симуляторы, обеспечивается интеграция инженерно-экономических знаний. Помимо сильного профессорско-преподавательского состава, обучение ведут топ-менеджеры, профессиональные бухгалтеры, аттестованные аудиторы и бизнес-тренеры.'
    if fac == 'ФПМИ':
        res = 'Факультет прикладной математики и информатики основан в 1993 году. На факультете ведется подготовка высококвалифицированных специалистов в области математического моделирования физических процессов, а также анализа больших данных методами машинного обучения. Цель образовательных программ – подготовка специалистов, способных к решению технических задач любой сложности с помощью современных IT-технологий.'
    if fac == 'ФТФ':
        res = 'Факультет основан в 1966 году. С тех пор ФТФ активно развивается и проводит подготовку специалистов в области ядерной физики и ядерных технологий, лазерных и квантовых технологий, фотоники, нефтегазового дела и геофизики. Одной из главных особенностей факультета является углубленная языковая подготовка: ФТФ - единственный технический факультет НГТУ, где студенты изучают иностранный язык на протяжении всего периода обучения. Здесь, как и во всем университете уделяется внимание практико-ориентированному обучению. Уже с 3-го курса студенты получают опыт научно-исследовательской работы на уникальном оборудовании в реально действующих лабораториях НИИ СО РАН и в технопарке Академгородка.'
    for data in nstu_data_json:
        if data['FACULTET'] == fac:
            directions = str(data['PROFILES']) + ';' + '\n'
            res += str(directions)
    return res.replace("профиль:", "").replace(")", "").replace("(", "").replace("(профили:", "").replace("специализация:", "")

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
        'listOfDirections': {
            'examples': ['Список направлений', 'Все направления', 'Направления'],
            'responses': [f"{all_directions()} Чтобы узнать подробнее о направлении, введите его название"]
        },
        'listOfFaculties': {
            'examples': ['Список факультетов', 'Все факультеты', 'Факультеты'],
            'responses': [f"{all_faculties()} Чтобы узнать подробнее о факультете, введите его аббревиатуру"]
        },
        'REF': {
            'examples': ['РЭФ'],
            'responses': [f"{faculty('РЭФ')}"]
        },
        'FGO': {
            'examples': ['ФГО'],
            'responses': [f"{faculty('ФГО')}"]
        },
        'FMA': {
            'examples': ['ФМА'],
            'responses': [f"{faculty('ФМА')}"]
        },
        'IST': {
            'examples': ['ИСТ'],
            'responses': [f"{faculty('ИСТ')}"]
        },
        'FEN': {
            'examples': ['ФЭН'],
            'responses': [f"{faculty('ФЭН')}"]
        },
        'AVTF': {
            'examples': ['АВТФ'],
            'responses': [f"{faculty('АВТФ')}"]
        },
        'FLA': {
            'examples': ['ФЛА'],
            'responses': [f"{faculty('ФЛА')}"]
        },
        'IDO': {
            'examples': ['ИДО'],
            'responses': [f"{faculty('ИДО')}"]
        },
        'MTF': {
            'examples': ['МТФ'],
            'responses': [f"{faculty('МТФ')}"]
        },
        'FB': {
            'examples': ['ФБ'],
            'responses': [f"{faculty('ФБ')}"]
        },
        'FPMI': {
            'examples': ['ФПМИ'],
            'responses': [f"{faculty('ФПМИ')}"]
        },
        'FTF': {
            'examples': ['ФТФ'],
            'responses': [f"{faculty('ФТФ')}"]
        },
        'firstQuestion': {
        'examples': ['Тест', 'Начнем', 'Пройти тест'],
        'responses': ['Если бы на свете существовали только две профессии, какую работу вы бы предпочли из двух? 1. Ухаживать за животными. 2. Обслуживать машины, приборы.'],
        'next_step': 'secondQuestion',
        "save_variable": 'firstQuestion',
        },
        'secondQuestion': {
        'examples': ['1', 'Тест'],
        'responses': ['Какую работу вы предпочтёте? 1. Помогать больным людям. 2. Составлять таблицы, схемы, программы вычислительных машин.'],
        'next_step': 'thirdQuestion',
        "save_variable": 'secondQuestion',
        },
        'thirdQuestion': {
        'responses': ['Какую работу вы предпочтёте? 1. Следить за состоянием, развитием растений. 2. Следить за качеством книжных иллюстраций, плакатов, художественных открыток.'],
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
        'responses': ['Какую работу вы предпочтёте? 1. Управлять каким-либо грузовым, подъемным транспортным средством (подъемным краном, трактором, тепловозом и др.) 2. Копировать рисунки, изображения, настраивать музыкальные инструменты '],
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
        },
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

            if (len(example) != 0 and distance / len(example)) < 0.1:
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
            file = open("file.txt", "r")
            number_in_file = file.read() # number_in_file = '0'
            file.close()
            if str(result).isnumeric():
                res = int(number_in_file) + int(result)
            else:
                res = int(number_in_file)
            file = open("file.txt", "w")
            file.write(str(res))
            bot_state["variables"]["result"] = res
            file.close()


        # Если в кейсе есть запрос на получение(get_variables), проходимся по строке и заменяем в строке
        # плейсхолдеры на значение переменных
        if 'get_variables' in BOT_CONFIG["intents"][intent]:

            #file = open("file.txt", "w")
            #file.write('0')
            #file.close()

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
    global res
    res += ball
    return res

def bot(replica):

    global result
    result = replica
    if bot_state["saveReplicaToVariable"] != None:
        # Сохраняем переменную
        var_name = bot_state["saveReplicaToVariable"]
        bot_state["variables"][var_name] = replica
        if bot_state["variables"]["result"] <= 12:
            bot_state["variables"]["resultText"] = "Ваш результат: человек-природа. Это профессии, в которых человек работает с живой и неживой природой." \
            "Главное - объект труда данного специалиста - какое-либо природное явление или объект природы." \
            "В НЭТИ такому результату соответствуют следующие направления: 12.03.04. Биотехнические системы и технологии;" \
            "18.03.01. Химическая технология; " \
            "03.03.02. Физика"
        elif bot_state["variables"]["result"] >= 13 and bot_state["variables"]["result"] <= 14:
            bot_state["variables"]["resultText"] = "Ваш результат: человек-человек. Это профессии, в которых человек работает с человеком или несколькими людьми. Он может работать как с телом человека, так и с его душой, сознанием, поведением. Он может лечить людей, обучать их, продавать им что-то, управлять ими, выступать перед ними, судить их, помогать им, выслушивать, проповедовать им, изучать их, развлекать их и т. д." \
            "В НЭТИ такому результату соответствуют следующие направления: 38.03.02 Менеджмент;" \
            "37.03.01 Психология; " \
            "42.03.01 Реклама и связи с общественностью"
        elif bot_state["variables"]["result"] >= 15 and bot_state["variables"]["result"] <= 16:
            bot_state["variables"]["resultText"] = "Ваш результат: человек-техника. Это профессии, в которых человек работает с техникой. Он технику разрабатывает, чинит, обслуживает, эксплуатирует. Это может быть инженер любого рода: энергетик, робототехник, самолётостроитель, автомобилестроитель, строитель поездов, лифтов, холодильников, станков, трубопроводов, медицинской техники, звукотехники и видеотехники, компьютерной техники" \
            "В НЭТИ такому результату соответствуют следующие направления: 15.03.06 Мехатроника и робототехника;" \
            "28.03.02 Наноинженерия; " \
            "09.03.01 Информатика и вычислительная техника;" \
            "15.03.03 Прикладная механика"
        elif bot_state["variables"]["result"] >= 17 and bot_state["variables"]["result"] <= 18:
            bot_state["variables"]["resultText"] = "Ваш результат: человек-знаковая система. Это профессии, в которых человек работает с числами, таблицами, графиками, схемами, буквами, шифрами." \
            "В НЭТИ такому результату соответствуют следующие направления: 10.03.01 Информационная безопасность" \
            "24.03.04. Авиастроение;" \
            "38.04.01 Экономика"
        elif bot_state["variables"]["result"] <= 20:
            bot_state["variables"]["resultText"] = "Ваш результат: человек-художественный образ. Это профессии, в которых человек работает с художественными образами, создавая их визуальное изображение, звуковой образ, литературный образ. " \
            "В НЭТИ такому результату соответствуют следующее направление: 54.02.02 Декоративно-прикладное искусство и народные промыслы"
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
