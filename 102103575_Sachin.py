from transformers import pipeline
from parameters import para
import pandas as pd
from topsis import topsis
from prompts import text, rs

models=['facebook/bart-large-cnn','philschmid/bart-large-cnn-samsum','sshleifer/distilbart-cnn-12-6', 'Falconsai/text_summarization', 'pszemraj/led-large-book-summary']
def main():
    for run in range(len(text)):
        results=[]
        for i in range(len(models)):
            summarization_pipeline = pipeline("summarization", model=models[i])

            prompt = text[run]
            refsum=rs[run]
            sum = summarization_pipeline(prompt, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
            gensum = sum[0]['summary_text']
            metrics = para(gensum, refsum)
            
            
            fk_grade, bleu_score, perplexity, meteor_score = metrics
            
            results.append({
                'Model': models[i],
                'Flesch-Kincaid Grade Level': fk_grade,
                'BLEU Score': bleu_score,
                'Perplexity': perplexity,
                'METEOR Score': meteor_score
            })

            # Print or use the metrics as needed
            print(f"\nResults for Model {i} ({models[i]}):")
            print(f"Flesch-Kincaid Grade Level: {fk_grade}")
            print(f"BLEU Score: {bleu_score}")
            print(f"Perplexity: {perplexity}")
            print(f"METEOR Score: {meteor_score}")

        
        df = pd.DataFrame(results)
        print(df)
        d2f=df.copy()
        criteria = ['Flesch-Kincaid Grade Level', 'BLEU Score', 'Perplexity', 'METEOR Score']
        weights = [1, 1, 1, 1]  
        impacts = ['-', '+', '-', '+'] 

        
        
        print(df)
        d2f['TOPSIS Score'] = topsis(d2f[criteria].values, weights, impacts)
        print(df)
        print(d2f)
        df['TOPSIS Score'] = d2f['TOPSIS Score']
        print(df)
        df['Rank'] = df['TOPSIS Score'].rank(ascending=False)
        print(df)

        df.to_csv(f'model_scores_topsis_{run}.csv', index=False)
        
        print(f"\nResults with TOPSIS scores and ranks saved to 'model_scores_topsis_{run}.csv'")

if __name__ == "__main__":
    main()
