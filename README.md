
---

# Model Analysis Report

## By:Sachin Sushil Singh Roll no: 102103575

## Executive Summary

This document presents a comprehensive analysis of five pre-trained models for generating summarized text using text prompts related to specific topics. The models considered are as follows:

1. 'facebook/bart-large-cnn'
2. 'philschmid/bart-large-cnn-samsum'
3. 'sshleifer/distilbart-cnn-12-6'
4. 'Falconsai/text_summarization'
5. 'pszemraj/led-large-book-summary'

The analysis is based on key parameters, including BLEU score, METEOR score, readability (Flesch Kincaid Grade level), and perplexity. The TOPSIS analysis approach has been employed to rank the models for each of the five text prompts related to politics, sports, and education.

## Model Evaluation Criteria

### 1. BLEU Score

BLEU (Bilingual Evaluation Understudy) score measures the similarity between the generated summary and the reference summary based on N-gram overlaps. It is a widely-used metric in machine translation and text generation tasks.

### 2. METEOR Score

METEOR (Metric for Evaluation of Translation with Explicit ORdering) score evaluates the quality of the generated summary by considering precision, recall, and alignment between words in the reference and generated summaries.

### 3. Readability (Flesch Kincaid Grade Level)

Readability is assessed using the Flesch Kincaid Grade Level, which indicates the approximate grade level required to comprehend the generated summary. Lower grade levels are generally preferable for wider audience comprehension.

### 4. Perplexity

Perplexity measures how well a language model predicts a sample. Lower perplexity values indicate better model performance.

## TOPSIS Analysis Results

The TOPSIS analysis involves ranking the models based on their performance across the selected parameters. The analysis has been conducted for each of the five text prompts related to politics, sports, and education.

### Results Summary

| Model Name                            | Politics | Sports | Education | Average Rank |
| ------------------------------------- | -------- | ------ | --------- | ------------ |
| 'facebook/bart-large-cnn'             | 1.6      | 1.2    | 1.6       | 1.47         |
| 'philschmid/bart-large-cnn-samsum'    | 4.0      | 4.2    | 3.8       | 3.67         |
| 'sshleifer/distilbart-cnn-12-6'       | 1.4      | 1.8    | 1.4       | 1.53         |
| 'Falconsai/text_summarization'        | 4.8      | 4.2    | 4.6       | 4.53         |
| 'pszemraj/led-large-book-summary'     | 3.2      | 3.6    | 3.6       | 4.47         |

## Conclusion

Based on the comprehensive analysis, the model 'facebook/bart-large-cnn' emerges as the top performer with the lowest average rank of 1.47 across the three topics. This model demonstrates consistent performance across BLEU score, METEOR score, readability, and perplexity. Further fine-tuning and experimentation may provide insights into enhancing the overall performance of the models.

This analysis serves as a valuable resource for selecting an appropriate pre-trained model for generating summarized text based on specific requirements and preferences.

--- 
