import sys, torch, argparse, os.path
import numpy as np
from transformers import AutoTokenizer
from datasets import Dataset, load_metric
from transformers import AutoModelForSequenceClassification, DistilBertForSequenceClassification,  TrainingArguments, Trainer,  DataCollatorWithPadding



#suggestions for code improvement:
# - add an argument to specify which model for HF hub we want to fine-tune, for now it's hard coded "bert-base-uncased"
# - add check that the model provided exists

parser = argparse.ArgumentParser()
parser.add_argument('--train', nargs=3, type=str, help='Provide the paths for the 2 training files: a first file containing the fluent utterances and another a second file containing the disfluent utterances. These files should contain one utterance per line. The third argument is the output directory where the training checkpoints will be saved.')
parser.add_argument('--dev',  nargs=2, type=str, help='Provide the paths for the 2 validating files: a first file containing the fluent utterances and another a second file containing the disfluent utterances. These files should contain one utterance per line.')
parser.add_argument('--test',  nargs=2, type=str, help='Provide the paths for the 2 testing files: a first file containing the fluent utterances and another a second file containing the disfluent utterances. These files should contain one utterance per line. Note: This is intended to be used only for evaluation purpose.')
parser.add_argument('--model', type=str, help='Path to the model directory to be used for inference.')
args = parser.parse_args()

if (args.train == None and args.dev != None) or (args.train != None and args.dev == None):
    sys.stderr.write('You have to provide both --dev and --train arguments if you wish to train the classifier.\n')
    sys.exit(0)
if args.model != None and args.test == None:
    sys.stderr.write('You provide a model but do not provide a test set file so I will read from stdin.\n')
if args.train == None and args.dev == None  and args.model == None:
    sys.stderr.write('You have to provide at least one option: train and dev, or model.\n')
    sys.exit(0) 

ERROR_MSG = dict()
ERROR_MSG['NOTFOUND'] = "not found!"

#dict to convert the labels into meaningful words
LABELS = dict()
LABELS[0] = 'DISFLUENT'
LABELS[1] = 'FLUENT'

#Loading the tokenizer used by BERT

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

#function called during training to compute the accuracy during training.
metric = load_metric("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

#preprocessing function to prepare the dataset
def preprocess_function(examples):
    #tokenize and pad the text values of the data. Padding is set to max length and if the sequence is too long the tokenizer truncates it.
    return tokenizer(examples["text"], padding="max_length", truncation=True)

#function for adding the data to the data structure
def add2data(label, file, target):
    for i in file:
        target['label'].append(label)
        label_counter[label] += 1.0
        target['text'].append(i.strip())
   

#check that files exists
def file_exist(f, max_i=2):
    for i in range(max_i):
        if not os.path.isfile(f[i]):
            sys.stderr.write(f[i]+" "+ERROR_MSG['NOTFOUND']+"\n")
            sys.exit(0)



data_labeled = dict()
#count instances for each class. This may be used for balancing class.
label_counter = dict()
label_counter[0] = 0
label_counter[1] = 0




#If we do training
if (args.train != None) :
    file_exist(args.train)
    file_exist(args.dev)

    #Reading the training data
    train_fluent = open(args.train[0]).readlines()
    train_disfluent = open(args.train[1]).readlines()

    #Reading the validating data
    dev_fluent = open(args.dev[0]).readlines()
    dev_disfluent = open(args.dev[1]).readlines()

    #initialize the data structure
    data_labeled['train'] = {'label':[],'text':[]}
    data_labeled['dev'] = {'label':[],'text':[]}

    #assign label 1 to fluent utterances and 0 to disfluent ones
    add2data(1,train_fluent,data_labeled['train'])
    add2data(0,train_disfluent,data_labeled['train'])
    add2data(1,dev_fluent,data_labeled['dev'])
    add2data(0,dev_disfluent,data_labeled['dev']) 
    #compute the weights for each class and add it to a tensor
    total_example = label_counter[0] + label_counter[1] 
    weights = torch.tensor([label_counter[0]/label_counter[1],1.0]).cuda()

    #small modification to the trainer from HF Transformer to add our custom loss with balanced weights
    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")

            # forward pass
            outputs = model(**inputs)            
            logits = outputs[1]

            # compute custom loss 
            loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    #preprocess the datasets and make batches
    data_labeled['train'] = Dataset.from_dict(data_labeled['train'])
    data_labeled['train'] = data_labeled['train'].map(preprocess_function, batched=True)
    data_labeled['dev'] = Dataset.from_dict(data_labeled['dev'])
    data_labeled['dev'] = data_labeled['dev'].map(preprocess_function, batched=True)

    #Get the pre-trained BERT model from HF hub   
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)

    training_args = TrainingArguments(
        output_dir=args.train[2],
        overwrite_output_dir=True,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        save_steps=10000,
        evaluation_strategy="epoch",
        warmup_steps=10000,
        do_predict=True,
        do_train=True
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=data_labeled["train"],
        eval_dataset=data_labeled["dev"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics        
    )

    trainer.train()

#if we do inference from stdin
if args.model != None and args.test == None:
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2, problem_type="multi_label_classification")
    for line in sys.stdin:
        inputs = tokenizer(line.strip(), padding="max_length", truncation=True, return_tensors="pt")
        #This next line is there only to correct some specific bugs with distilBERT, maybe removal with next version of HF Transformers (>4.13)
        inputs =  {key:value for (key,value) in inputs.items() if key in model.forward.__code__.co_varnames}
        logits = model(**inputs).logits
        predicted_class_id = logits.argmax().item()
        print(LABELS[int(model.config.id2label[predicted_class_id].split('_')[1])])

if (args.test != None and args.model != None) :


    data_labeled['test_fluent'] = {'label':[],'text':[]}
    data_labeled['test_disfluent'] = {'label':[],'text':[]}

    
    file_exist(args.test)
    test_fluent = open(args.test[0]).readlines()
    test_disfluent = open(args.test[1]).readlines()

    add2data(1,test_fluent,data_labeled['test_fluent'])
    add2data(0,test_disfluent,data_labeled['test_disfluent'])

    data_labeled['test_fluent'] = Dataset.from_dict(data_labeled['test_fluent'])
    data_labeled['test_fluent'] = data_labeled['test_fluent'].map(preprocess_function, batched=True)
    data_labeled['test_disfluent'] = Dataset.from_dict(data_labeled['test_disfluent'])
    data_labeled['test_disfluent'] = data_labeled['test_disfluent'].map(preprocess_function, batched=True)

    if (args.train == None):
        model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)

        #The following training and trainer object instanciation should be cleaner
        training_args = TrainingArguments(
            output_dir="./",
            overwrite_output_dir=False,
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=5,
            weight_decay=0.01,
            save_steps=10000,
            evaluation_strategy="epoch",
            warmup_steps=10000,
            do_predict=True,
            do_train=True
        )


        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=data_labeled["test_fluent"],
            eval_dataset=data_labeled["test_fluent"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )


    test_predictions = trainer.predict(data_labeled['test_fluent'])
    print('Accuracy for fluent utterances')
    print(test_predictions)

    test_predictions = trainer.predict(data_labeled['test_disfluent'])
    print('Accuracy for disfluent utterances')
    print(test_predictions)


    

