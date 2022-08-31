import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, BertTokenizer 
from transformers import BertForQuestionAnswering

#Squad Training

from datasets import load_dataset
raw_datasets = load_dataset('squad')



tokenizer_ajai = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")

inputs = tokenizer_ajai(
    raw_datasets["train"][2:6]["question"],
    raw_datasets["train"][2:6]["context"],
    max_length=100,
    truncation="only_second",
    stride=50,
    return_overflowing_tokens=True,
    return_offsets_mapping=True,
)

from tqdm.auto import tqdm 

def add_end_idx(answers, contexts):
    new_answers = []
    # loop through each answer-context pair
    for answer, context in tqdm(zip(answers, contexts)):
        # quick reformating to remove lists
        answer['text'] = answer['text'][0]
        answer['answer_start'] = answer['answer_start'][0]
        # gold_text refers to the answer we are expecting to find in context
        gold_text = answer['text']
        # we already know the start index
        start_idx = answer['answer_start']
        # and ideally this would be the end index...
        end_idx = start_idx + len(gold_text)

        # ...however, sometimes squad answers are off by a character or two
        if context[start_idx:end_idx] == gold_text:
            # if the answer is not off :)
            answer['answer_end'] = end_idx
        else:
            # this means the answer is off by 1-2 tokens
            for n in [1, 2]:
                if context[start_idx-n:end_idx-n] == gold_text:
                    answer['answer_start'] = start_idx - n
                    answer['answer_end'] = end_idx - n
        new_answers.append(answer)
    return new_answers

def prep_data(dataset):
    questions = dataset['question']
    contexts = dataset['context']
    answers = add_end_idx(
        dataset['answers'],
        contexts
    )
    return {
        'question': questions,
        'context': contexts,
        'answers': answers
    }

dataset = prep_data(raw_datasets['train'][:10000])


train = tokenizer_ajai(dataset['context'], dataset['question'],
                  truncation=True, padding='max_length',
                  max_length=512, return_tensors='pt')


def add_token_positions(encodings, answers):
    # initialize lists to contain the token indices of answer start/end
    start_positions = []
    end_positions = []
    for i in tqdm(range(len(answers))):
        # append start/end token position using char_to_token method
        start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
        end_positions.append(encodings.char_to_token(i, answers[i]['answer_end']))

        # if start position is None, the answer passage has been truncated
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer_ajai.model_max_length
        # end position cannot be found, char_to_token found space, so shift position until found
        shift = 1
        while end_positions[-1] is None:
            end_positions[-1] = encodings.char_to_token(i, answers[i]['answer_end'] - shift)
            shift += 1
    # update our encodings object with the new token-based start/end positions
    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

    add_token_positions(train, dataset['answers'])

    train.keys()

    train['start_positions'][:5], train['end_positions'][:5]

    import torch

class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

# build datasets for both our training data
train_dataset = SquadDataset(train)

loader = torch.utils.data.DataLoader(train_dataset,
                                     batch_size=16,
                                     shuffle=True)

m = BertForQuestionAnswering.from_pretrained('/Users/ajai_devanathan/Desktop/project_ir/covidbert_last_layer_training/')

from transformers import AdamW

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
m.to(device)
m.train()
optim = AdamW(m.parameters(), lr=5e-5)

for epoch in range(1):
    loop = tqdm(loader)
    for batch in loop:
        optim.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)

        outputs = m(input_ids, attention_mask=attention_mask,
                        start_positions=start_positions,
                        end_positions=end_positions)
        
        loss = outputs[0]
        loss.backward()
        optim.step()

        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())


m.save_pretrained('/Users/ajai_devanathan/Desktop/project_ir/covbert_squad_10000/')
covbert_squad_10000=BertForQuestionAnswering.from_pretrained('/Users/ajai_devanathan/Desktop/project_ir/covbert_squad_10000/')
#Testing the bot
paragraph1 = '''Coronavirus disease (COVID-19) is an infectious disease caused by the SARS-CoV-2 virus.Most people infected with the virus will experience mild to moderate respiratory illness and recover without requiring special treatment. However, some will become seriously ill and require medical attention. Older people and those with underlying medical conditions like cardiovascular disease, diabetes, chronic respiratory disease, or cancer are more likely to develop serious illness. Anyone can get sick with COVID-19 and become seriously ill or die at any age. The best way to prevent and slow down transmission is to be well informed about the disease and how the virus spreads. Protect yourself and others from infection by staying at least 1 metre apart from others, wearing a properly fitted mask, and washing your hands or using an alcohol-based rub frequently. Get vaccinated when it’s your turn and follow local guidance.The virus can spread from an infected person’s mouth or nose in small liquid particles when they cough, sneeze, speak, sing or breathe. These particles range from larger respiratory droplets to smaller aerosols. It is important to practice respiratory etiquette, for example by coughing into a flexed elbow, and to stay home and self-isolate until you recover if you feel unwell.To prevent infection and to slow transmission of COVID-19, do the following: Get vaccinated when a vaccine is available to you.Stay at least 1 metre apart from others, even if they don’t appear to be sick.Wear a properly fitted mask when physical distancing is not possible or when in poorly ventilated settings.Choose open, well-ventilated spaces over closed ones. Open a window if indoors.Wash your hands regularly with soap and water. clean them with alcohol-based hand rub.Cover your mouth and nose when coughing or sneezing.If you feel unwell, stay home and self-isolate until you recover.'''
question1 = 'What is the best way to prevent and slow down transmission of COVID-19?'

encoding = tokenizer_ajai.encode_plus(text=question1,text_pair=paragraph1)
inputs = encoding['input_ids']  #Token embeddings
sentence_embedding = encoding['token_type_ids']  #Segment embeddings
tokens = tokenizer_ajai.convert_ids_to_tokens(inputs)
start_scores = covbert_squad_10000(input_ids=torch.tensor([inputs]), token_type_ids=torch.tensor([sentence_embedding]))[0]
end_scores =  covbert_squad_10000(input_ids=torch.tensor([inputs]), token_type_ids=torch.tensor([sentence_embedding]))[1]
start_index = torch.argmax(start_scores)
end_index = torch.argmax(end_scores)
answer = ' '.join(tokens[start_index:end_index+1])
answer 

paragraph2 = '''Coronavirus disease (COVID-19) is an infectious disease caused by the SARS-CoV-2 virus.Most people infected with the virus will experience mild to moderate respiratory illness and recover without requiring special treatment. However, some will become seriously ill and require medical attention. Older people and those with underlying medical conditions like cardiovascular disease, diabetes, chronic respiratory disease, or cancer are more likely to develop serious illness. Anyone can get sick with COVID-19 and become seriously ill or die at any age. The best way to prevent and slow down transmission is to be well informed about the disease and how the virus spreads. Protect yourself and others from infection by staying at least 1 metre apart from others, wearing a properly fitted mask, and washing your hands or using an alcohol-based rub frequently. Get vaccinated when it’s your turn and follow local guidance.The virus can spread from an infected person’s mouth or nose in small liquid particles when they cough, sneeze, speak, sing or breathe. These particles range from larger respiratory droplets to smaller aerosols. It is important to practice respiratory etiquette, for example by coughing into a flexed elbow, and to stay home and self-isolate until you recover if you feel unwell.To prevent infection and to slow transmission of COVID-19, do the following: Get vaccinated when a vaccine is available to you.Stay at least 1 metre apart from others, even if they don’t appear to be sick.Wear a properly fitted mask when physical distancing is not possible or when in poorly ventilated settings.Choose open, well-ventilated spaces over closed ones. Open a window if indoors.Wash your hands regularly with soap and water. clean them with alcohol-based hand rub.Cover your mouth and nose when coughing or sneezing.If you feel unwell, stay home and self-isolate until you recover.'''
question2 = '''can corona give me respiratory illness?'''

encoding = tokenizer_ajai.encode_plus(text=question2,text_pair=paragraph2)
inputs = encoding['input_ids']  #Token embeddings
sentence_embedding = encoding['token_type_ids']  #Segment embeddings
tokens = tokenizer.convert_ids_to_tokens(inputs)
start_scores = covbert_squad_10000(input_ids=torch.tensor([inputs]), token_type_ids=torch.tensor([sentence_embedding]))[0]
end_scores =  covbert_squad_10000(input_ids=torch.tensor([inputs]), token_type_ids=torch.tensor([sentence_embedding]))[1]
start_index = torch.argmax(start_scores)
end_index = torch.argmax(end_scores)
answer2 = ' '.join(tokens[start_index:end_index+1])
answer2

paragraph = '''Coronavirus disease (COVID-19) is an infectious disease caused by the SARS-CoV-2 virus.Most people infected with the virus will experience mild to moderate respiratory illness and recover without requiring special treatment. However, some will become seriously ill and require medical attention. Older people and those with underlying medical conditions like cardiovascular disease, diabetes, chronic respiratory disease, or cancer are more likely to develop serious illness. Anyone can get sick with COVID-19 and become seriously ill or die at any age. The best way to prevent and slow down transmission is to be well informed about the disease and how the virus spreads. Protect yourself and others from infection by staying at least 1 metre apart from others, wearing a properly fitted mask, and washing your hands or using an alcohol-based rub frequently. Get vaccinated when it’s your turn and follow local guidance.The virus can spread from an infected person’s mouth or nose in small liquid particles when they cough, sneeze, speak, sing or breathe. These particles range from larger respiratory droplets to smaller aerosols. It is important to practice respiratory etiquette, for example by coughing into a flexed elbow, and to stay home and self-isolate until you recover if you feel unwell.To prevent infection and to slow transmission of COVID-19, do the following: Get vaccinated when a vaccine is available to you.Stay at least 1 metre apart from others, even if they don’t appear to be sick.Wear a properly fitted mask when physical distancing is not possible or when in poorly ventilated settings.Choose open, well-ventilated spaces over closed ones. Open a window if indoors.Wash your hands regularly with soap and water. clean them with alcohol-based hand rub.Cover your mouth and nose when coughing or sneezing.If you feel unwell, stay home and self-isolate until you recover.'''
question = '''hands can be washed with?'''
encoding = tokenizer_ajai.encode_plus(text=question,text_pair=paragraph)
inputs = encoding['input_ids']  #Token embeddings
sentence_embedding = encoding['token_type_ids']  #Segment embeddings
tokens = tokenizer.convert_ids_to_tokens(inputs)
start_scores = covbert_squad_10000(input_ids=torch.tensor([inputs]), token_type_ids=torch.tensor([sentence_embedding]))[0]
end_scores =  covbert_squad_10000(input_ids=torch.tensor([inputs]), token_type_ids=torch.tensor([sentence_embedding]))[1]
start_index = torch.argmax(start_scores)
end_index = torch.argmax(end_scores)
answer = ' '.join(tokens[start_index:end_index+1])
#answer

corrected_answer = ''

for word in answer.split():
    
    #If it's a subword token
    if word[0:2] == '##':
        corrected_answer += word[2:]
    else:
        corrected_answer += ' ' + word

print(corrected_answer)




# THE NEXT 10,000-20,000

dataset1 = prep_data(raw_datasets['train'][10001:20000])
train1 = tokenizer_ajai(dataset1['context'], dataset1['question'],
                  truncation=True, padding='max_length',
                  max_length=512, return_tensors='pt')
train_dataset1 = SquadDataset(train1)
loader1 = torch.utils.data.DataLoader(train_dataset1,
                                     batch_size=16,
                                     shuffle=True)
m1 = BertForQuestionAnswering.from_pretrained('/Users/ajai_devanathan/Desktop/project_ir/covbert_squad_10000')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
m1.to(device)
m1.train()
optim = AdamW(m.parameters(), lr=5e-5)

for epoch in range(1):
    loop = tqdm(loader)
    for batch in loop:
        optim.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)

        outputs = m1(input_ids, attention_mask=attention_mask,
                        start_positions=start_positions,
                        end_positions=end_positions)
        
        loss = outputs[0]
        loss.backward()
        optim.step()

        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())


m1.save_pretrained('/Users/ajai_devanathan/Desktop/project_ir/covbert_squad_10000-20000/')
covbert_squad_10000_20000=BertForQuestionAnswering.from_pretrained('/Users/ajai_devanathan/Desktop/project_ir/covbert_squad_10000-20000/')


def queryme(text,question):
    encoding = tokenizer_ajai.encode_plus(text=query,text_pair=text)
    inputs = encoding['input_ids']  #Token embeddings
    sentence_embedding = encoding['token_type_ids']  #Segment embeddings
    tokens = tokenizer_ajai.convert_ids_to_tokens(inputs)
    start_scores = covbert_squad_10000_20000(input_ids=torch.tensor([inputs]), token_type_ids=torch.tensor([sentence_embedding]))[0]
    end_scores =  covbert_squad_10000_20000(input_ids=torch.tensor([inputs]), token_type_ids=torch.tensor([sentence_embedding]))[1]
    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores)
    answer = ' '.join(tokens[start_index:end_index+1])
    corrected_answer = ''

    for word in answer.split():
     
    #If it's a subword token
        if word[0:2] == '##':
            corrected_answer += word[2:]
        else:
            corrected_answer += ' ' + word

    return corrected_answer

text = 'corona virus is a infectious disease. It can kill anybody. Mostly old peope die because of this. please wash hands with soap and water'
query = 'hands can be should be washed with what?'
queryme(text,query)
with open('results.txt','a+') as f:
    print(text,file=f)
    print(query,file=f)
    print(queryme(text,query),file=f)

query = 'who die in covid?'
with open('results.txt','a+') as f:
    print(text,file=f)
    print(query,file=f)
    print(queryme(text,query),file=f)

text = 'Coronavirus disease (COVID-19) is an infectious disease caused by the SARS-CoV-2 virus.Most people infected with the virus will experience mild to moderate respiratory illness and recover without requiring special treatment. However, some will become seriously ill and require medical attention. Older people and those with underlying medical conditions like cardiovascular disease, diabetes, chronic respiratory disease, or cancer are more likely to develop serious illness. Anyone can get sick with COVID-19 and become seriously ill or die at any age. The best way to prevent and slow down transmission is to be well informed about the disease and how the virus spreads. Protect yourself and others from infection by staying at least 1 metre apart from others, wearing a properly fitted mask, and washing your hands or using an alcohol-based rub frequently. Get vaccinated when it’s your turn and follow local guidance.The virus can spread from an infected person’s mouth or nose in small liquid particles when they cough, sneeze, speak, sing or breathe. These particles range from larger respiratory droplets to smaller aerosols. It is important to practice respiratory etiquette, for example by coughing into a flexed elbow, and to stay home and self-isolate until you recover if you feel unwell.To prevent infection and to slow transmission of COVID-19, do the following: Get vaccinated when a vaccine is available to you.Stay at least 1 metre apart from others, even if they don’t appear to be sick.Wear a properly fitted mask when physical distancing is not possible or when in poorly ventilated settings.Choose open, well-ventilated spaces over closed ones. Open a window if indoors.Wash your hands regularly with soap and water. clean them with alcohol-based hand rub.Cover your mouth and nose when coughing or sneezing.If you feel unwell, stay home and self-isolate until you recover.'
query = 'hands can be washed with what?'
with open('results.txt','a+') as f:
    print(text,file=f)
    print(query,file=f)
    print(queryme(text,query),file=f)

query = 'what will people experience?'
with open('results.txt','a+') as f:
    print(text,file=f)
    print(query,file=f)
    print(queryme(text,query),file=f)

query = 'what age can we die?'
with open('results.txt','a+') as f:
    print(text,file=f)
    print(query,file=f)
    print(queryme(text,query),file=f)

text = "The Coronavirus (CoV) is a large family of viruses known to cause illnesses ranging from the common cold to acute respiratory tract infection. The severity of the infection may be visible as pneumonia, acute respiratory syndrome, and even death. Until the outbreak of SARS, this group of viruses was greatly overlooked. However, since the SARS and MERS outbreaks, these viruses have been studied in greater detail, propelling the vaccine research. On December 31, 2019, mysterious cases of pneumonia were detected in the city of Wuhan in China's Hubei Province. On January 7, 2020, the causative agent was identified as a new coronavirus (2019-nCoV), and the disease was later named as COVID-19 by the WHO. The virus spread extensively in the Wuhan region of China and has gained entry to over 210 countries and territories."
query = 'where was pneumonia detected?'
with open('results.txt','a+') as f:
    print(text,file=f)
    print(query,file=f)
    print(queryme(text,query),file=f)

text ="Though experts suspected that the virus is transmitted from animals to humans, there are mixed reports on the origin of the virus. There are no treatment options available for the virus as such, limited to the use of anti-HIV drugs and/or other antivirals such as Remdesivir and Galidesivir. For the containment of the virus, it is recommended to quarantine the infected and to follow good hygiene practices. The virus has had a significant socio-economic impact globally. Economically, China is likely to experience a greater setback than other countries from the pandemic due to added trade war pressure, which have been discussed in this paper."
query = 'is there any treatment option?'
with open('results.txt','a+') as f:
    print(text,file=f)
    print(query,file=f)
    print(queryme(text,query),file=f)

text ="Coronaviridae is a family of viruses with a positive-sense RNA that possess an outer viral coat. When looked at with the help of an electron microscope, there appears to be a unique corona around it. This family of viruses mainly cause respiratory diseases in humans, in the forms of common cold or pneumonia as well as respiratory infections. These viruses can infect animals as well (1, 2). Up until the year 2003, coronavirus (CoV) had attracted limited interest from researchers. However, after the SARS (severe acute respiratory syndrome) outbreak caused by the SARS-CoV, the coronavirus was looked at with renewed interest (3, 4). This also happened to be the first epidemic of the 21st century originating in the Guangdong province of China. Almost 10 years later, there was a MERS (Middle East respiratory syndrome) outbreak in 2012, which was caused by the MERS-CoV (5, 6). Both SARS and MERS have a zoonotic origin and originated from bats. A unique feature of these viruses is the ability to mutate rapidly and adapt to a new host. The zoonotic origin of these viruses allows them to jump from host to host. Coronaviruses are known to use the angiotensin-converting enzyme-2 (ACE-2) receptor or the dipeptidyl peptidase IV (DPP-4) protein to gain entry into cells for replication (7–10).In December 2019, almost seven years after the MERS 2012 outbreak, a novel Coronavirus (2019-nCoV) surfaced in Wuhan in the Hubei region of China. The outbreak rapidly grew and spread to neighboring countries. However, rapid communication of information and the increasing scale of events led to quick quarantine and screening of travelers, thus containing the spread of the infection. The major part of the infection was restricted to China, and a second cluster was found on a cruise ship called the Diamond Princess docked in Japan (11, 12)."
query = 'what is the origin of the virus?'
queryme(text,query)

query = 'what did we find in the electron microscope?'
queryme(text,query)
with open('results.txt','a+') as f:
    print(text,file=f)
    print(query,file=f)
    print(queryme(text,query),file=f)




