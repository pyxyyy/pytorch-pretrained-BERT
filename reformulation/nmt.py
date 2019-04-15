"""
nmt based query reformulation
"""

from googletrans import Translator
import csv
import os
from google.cloud import translate_v3beta1 as translate


def google_translate_batch(questions, output):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/peixuan/Downloads/green-chalice-237310-5fa1959eb742.json'
    client = translate.TranslationServiceClient()
    project_id = 'green-chalice-237310'
    location = 'us-central1'

    parent = client.location_path(project_id, location)

    operation = client.batch_translate_text(
        parent=parent,
        source_language_code='en',
        target_language_codes=['zh-CN'])

    result = operation.result(90)

    print('Total Characters: {}'.format(result.total_characters))
    print('Translated Characters: {}'.format(result.translated_characters))


def google_translate(questions, output):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/peixuan/Downloads/green-chalice-237310-5fa1959eb742.json'
    client = translate.TranslationServiceClient()
    project_id = 'green-chalice-237310'
    location = 'us-central1'

    parent = client.location_path(project_id, location)

    with open(output, 'w') as fout:

        while len(questions) > 0:

            subset_questions = questions[:5]
            questions = questions[5:]

            cn_response = client.translate_text(
                parent=parent,
                contents=subset_questions,
                source_language_code='en',
                target_language_code='zh-CN')
            cn_questions = []
            for cn_question in cn_response.translations:
                cn_questions.append(cn_question.translated_text)
            # print('cn: {}'.format(cn_response.translations[0].translated_text))
            # print(cn_questions)
            en_response = client.translate_text(
                parent=parent,
                contents=cn_questions,
                mime_type='text/plain',
                source_language_code='zh-CN',
                target_language_code='en'
            )
            # print('en: {}'.format(en_response.translations[0].translated_text))
            for en_question in en_response.translations:
                # print(en_question.translated_text)
                fout.write(en_question.translated_text + '\n')


def googletrans_csv(questions, metadata, header, output, src, dst):
    print(questions[0:3])
    generated_questions = []
    translator = Translator()
    for question in questions:
        if len(question) < 15000:  # api limit: num of characters < 15k
            print('translating {}'.format(question))
            dst_obj = translator.translate(question, dest=dst, src=src)
            src_obj = translator.translate(dst_obj.text, dest=src, src=dst)
            generated_questions.append(src_obj.text)
    with open(output, 'w') as fout:
        fout.write(header)
        for q, m in zip(generated_questions, metadata):
            m.insert(1, q)
            fout.write(','.join(m) + '\n')


def googletrans_text(questions, output, src, dst):
    translator = Translator()
    with open(output, 'w') as fout:
        for question in questions:
            if len(question) < 15000:  # api limit: num of characters < 15k
                print('translating {}'.format(question))
                dst_obj = translator.translate(question, dest=dst, src=src)
                src_obj = translator.translate(dst_obj.text, dest=src, src=dst)
                fout.write(src_obj.text + '\n')


def extract_questions_csv(csv_file):
    with open(csv_file, 'r') as fin:
        questions = []
        metadata = []
        header = []
        reader = csv.reader(fin, delimiter=',')
        for i, row in enumerate(reader):
            if i == 0:
                header = row
            else:
                question = row.pop(1)
                questions.append(question)
                metadata.append(row)
        return questions, metadata, header


def extract_questions_text(txt):
    questions = []
    with open(txt, 'r') as fin:
        for line in fin:
            line = line.strip()
            if 0 < len(line) < 500:
                questions.append(line.strip())
    return questions


def run_csv():
    csv_newsqa = 'newsqa-data-v1.csv'
    csv_output = 'zhcn-newsqa-data-v1.csv'
    questions, metadata, header = extract_questions_csv(csv_newsqa)
    googletrans_csv(questions, metadata, header, csv_output, 'en', 'zh-CN')


def run_text():
    text_newsqa = 'question_090.txt'
    text_output = 'zhcn-question_090.txt'
    questions = extract_questions_text(text_newsqa)
    num_chars = 0
    for question in questions:
        num_chars += len(question)
    print(num_chars)
    # google_translate_text(questions, text_output, 'en', 'zh-CN')


def clean(questions, metadata):
    startwords = ['what', 'who', 'when', 'how', 'why', 'where']
    clean_questions = []
    clean_metadata = []
    for i, question in enumerate(questions):
        question = question.lower()
        words = question.split()
        if words[0] in startwords and words[-1][-1] == '?':
            clean_questions.append(question)
            clean_metadata.append(metadata[i])
    return clean_questions, clean_metadata


def main():
    # run_text()
    # run_csv()
    questions = extract_questions_text('from35541_question_090.txt')
    google_translate(questions, 'zhcn_from35541_question_090.txt')


if __name__ == "__main__":
    main()
