from __future__ import unicode_literals, print_function

import plac
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding

def preprocess(file_name="pnr.train.txt"):
    file_path = file_name
    train_data = []
    sentence = []
    with open(file_path) as f:
        for line in f:
            if line == '\n':
                if len(sentence) != 0:
                    train_data.append(sentence)
                sentence = []
            else: 
                if line != '\n' and len(line.split(' ')) !=3:
                    line = line.strip()
                    sentence.append(line)
    return(train_data)

def extractEntities(train_data):
    batch = []
    for item in train_data:
        sentence = ''
        entities = {'entities':[]}
        for line_itr in range(0,len(item)):
            person_words = []
            splitted = item[line_itr].split(' ')
            if len(splitted) == 4:
                word=splitted[0]
                # pos=splitted[1]
                # print(splitted)
                entity=splitted[3]
                if entity == 'I':
                    person_words.append(word)
                sentence+=word+' '
                # sentence=sentence.rstrip()
                single_entity = []
                for person in person_words:
                    start_ind = sentence.index(person)
                    end_ind = start_ind+len(person)
                    single_entity.append(start_ind)
                    single_entity.append(end_ind)
                    single_entity.append('PERSON')
                    single_entity=tuple(single_entity)
                    entities['entities'].append(single_entity)
                    single_entity = []
        temp = []
        temp.append(sentence)
        temp.append(entities)
        batch.append(temp)

    return(batch)

TRAIN_DATA = extractEntities(preprocess(file_name="pnr.train.txt"))

def main(model=None, output_dir=None, n_iter=10):
    """Load the model, set up the pipeline and train the entity recognizer."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        # print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        # print("Created blank 'en' model")
        
    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        print('else')
        ner = nlp.get_pipe("ner")

    ner.add_label('PERSON')
    # add labels
    # for _, annotations in TRAIN_DATA:
    #     for ent in annotations.get("entities"):
    #         ner.add_label(ent[2])
            
    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        # reset and initialize the weights randomly – but only if we're
        # training a new model
        if model is None:
            nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    drop=0.5,  # dropout - make it harder to memorise data
                    losses=losses,
                )
#             print("Losses", losses)
    # test the trained model
    for text, _ in TRAIN_DATA:
        doc = nlp(text)
        # print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
        # print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)
        
    # # test the saved model
    # print("Loading from", output_dir)
    # nlp2 = spacy.load(output_dir)
    # for text in TEST_DATA:
    #     doc = nlp2(text)
    #     print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
    #     print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])

def findAllPersons(file_name="pnr.test.txt"):
    file_path = file_name
    persons = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if len(line.split(' ')) == 4:
                splitted = line.split(' ')
                word=splitted[0]
                entity=splitted[3]
                if entity == 'I':
                    persons.append(word)
    return(persons)

def getPersonInLine():
    test_data = extractEntities(preprocess(file_name="pnr.test.txt"))
    list_test_persons=findAllPersons()
    result_person = []
    for text, _ in test_data:
        personInLine = []
        for person in list_test_persons:
            if person in text:
                personInLine.append(person)
        result_person.append(personInLine)
        personInLine=[]
    return result_person

def predict(output_dir="saved_model3"):
    # test the saved model
    nlp2 = spacy.load(output_dir)
    list_test_persons=getPersonInLine()
    length_test_person = len(list_test_persons)
    correct = 0
    not_correct = 0
    test_data = extractEntities(preprocess(file_name="pnr.test.txt"))
    for text,_ in test_data:
        wrong = False
        doc = nlp2(text)
        for ent in doc.ents:
            for person in list_test_persons:
                if ent.text not in person:
                    wrong = True
                    break
            if wrong:
                not_correct += 1
            else:
                correct += 1
    print('Accuracy: ')
    print((correct/len(test_data))*100)


predict(output_dir="saved_model3")
# main(output_dir="saved_model3")