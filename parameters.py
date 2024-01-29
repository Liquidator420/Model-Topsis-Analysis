import textstat 
import nltk
from nltk.translate import meteor_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def calculate_readability(text):
    # Flesch-Kincaid Grade Level
    fk_grade = textstat.flesch_kincaid_grade(text)


    return fk_grade



def calculate_bleu(generated_summary, reference_summary):
    # Tokenize the sentences
    gen_tokens = nltk.word_tokenize(generated_summary.lower())
    ref_tokens = nltk.word_tokenize(reference_summary.lower())

    # Calculate BLEU score for a single sentence
    bleu_score = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=SmoothingFunction().method1)

    return bleu_score



def calculate_perplexity(generated_summary, model_name="gpt2"):
    # Load pre-trained GPT-2 model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Tokenize the input text
    input_ids = tokenizer.encode(generated_summary, return_tensors="pt")

    # Get the model's output logits
    with torch.no_grad():
        logits = model(input_ids)[0]

    # Calculate perplexity
    perplexity = torch.exp(torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), input_ids.view(-1)))

    return perplexity.item()



def calculate_meteor(generated_summary, reference_summary):
    # Tokenize the sentences
    gen_tokens = nltk.word_tokenize(generated_summary.lower())
    ref_tokens = nltk.word_tokenize(reference_summary.lower())

    # Calculate METEOR score
    meteor_score_value = meteor_score.meteor_score([ref_tokens], gen_tokens)

    return meteor_score_value




def para(gensum, refsum):

    bleu_score = calculate_bleu(gensum, refsum)
    fk_grade= calculate_readability(gensum)
    perplexity = calculate_perplexity(gensum)
    meteor_score = calculate_meteor(gensum, refsum)
    return fk_grade, bleu_score, perplexity, meteor_score
