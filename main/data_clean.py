import pandas as pd

faq = pd.read_excel(r"faq/tuc_faq.xlsx")
faq.sort_values(by='web-scraper-order', ascending=True)
faq_questions = faq[['question']].dropna().reset_index(drop=True)
faq_responses = faq[['response']].dropna().reset_index(drop=True)
faq_resul = pd.concat([faq_questions, faq_responses], axis=1)