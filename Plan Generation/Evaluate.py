import os
from my_llmcall import call_groq
import json
from transformers import BertModel, BertTokenizer
import torch
from torch.optim import AdamW
import torch.nn as nn
from rouge import Rouge
from evaluate import load
import evaluate
from tqdm import tqdm

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# get the answer from the chatgpt
# def chatgpt_answer(question, gpt_model="gpt-3.5-turbo"):
#     os.environ["OPENAI_API_KEY"] = "..."
#     openai.api_key = os.environ["OPENAI_API_KEY"]

#     client = OpenAI()
#     completion = client.chat.completions.create(
#         model=gpt_model,
#         messages=[
#             {"role": "user", "content": question}
#         ]
#     )
#     return completion.choices[0].message.content

def chatgpt_answer(question, gpt_model="llama-3.1-8b-instant"):
    messages=[
            {"role": "user", "content": question}
    ]
    context = call_groq(messages, gpt_model)
    
    if context is None:
        exception_msg = f"No context found for question: {context}"
        raise Exception(exception_msg)
    return context

# input all the previous memory, and format them in the way "question-answer", store them in a list
def allSentences(memorybase_file_path):
    IO_list = []  # Initialize the empty list
    with open(memorybase_file_path, 'r', encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)  # Load JSON object from each line
            # Check if there are at least two messages to combine
            if len(data["messages"]) >= 2:
                # Extract sentences from the first and second messages' "content" and combine them
                combined_sentence = data["messages"][0]["content"] + "->" + data["messages"][1]["content"]
                IO_list.append(combined_sentence)  # Add the combined sentence to the list A
    return IO_list


# Model Definition with a classification layer
class SentenceSimilarityModel(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super(SentenceSimilarityModel, self).__init__()
        # 用于将输入的句子编码为向量
        self.encoder = BertModel.from_pretrained(model_name)
        # 分词器将句子分成词，转化为BERT可以处理的输入
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        # 分类器，将句子的向量拼接后输入分类器，输出一个分数
        self.classifier = nn.Linear(self.encoder.config.hidden_size * 2, 1)

    def forward(self, sentence_pairs_s):
        flattened_sentences = [sentence_e for pair in sentence_pairs_s for sentence_e in pair]
        inputs = self.tokenizer(flattened_sentences, padding=True, truncation=True, return_tensors="pt", max_length=512)
        outputs = self.encoder(**inputs)
        sentence_embeddings = outputs.pooler_output.view(-1, 2, self.encoder.config.hidden_size)
        combined_embeddings = torch.cat((sentence_embeddings[:, 0], sentence_embeddings[:, 1]), dim=1)
        similarity_scores = self.classifier(combined_embeddings).squeeze(-1)
        return similarity_scores, sentence_embeddings


def save_model_and_optimizer(trained_model, trained_optimizer, model_path="model.pth", optimizer_path="optimizer.pth"):
    torch.save(trained_model.state_dict(), model_path)
    torch.save(trained_optimizer.state_dict(), optimizer_path)
    #print(f"Saved model to {model_path} and optimizer state to {optimizer_path}")


def load_model_and_optimizer(trained_model, trained_optimizer, model_path="model.pth", optimizer_path="optimizer.pth"):
    trained_model.load_state_dict(torch.load(model_path))
    trained_optimizer.load_state_dict(torch.load(optimizer_path))
    #print(f"Loaded model from {model_path} and optimizer state from {optimizer_path}")


def find_most_similar_sentences(test_sentence_eee, dataset_sentences, model_lll, tokenizer, top_k=2):
    model_lll.eval()
    scores = []

    with torch.no_grad():   # 在该代码块内不计算梯度，节省资源
        test_embedding = None
        # First, get the embedding of the test sentence
        # 首先，将待检索的句子编码为向量
        inputs = tokenizer(test_sentence_eee, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = model_lll.encoder(**inputs)
        test_embedding = outputs.pooler_output

        # Then, compare it to each sentence in the dataset
        # 然后比较它与数据集中的每个句子的向量，计算相似度
        for dataset_sentence in dataset_sentences:
            inputs = tokenizer(dataset_sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
            outputs = model_lll.encoder(**inputs)
            dataset_embedding = outputs.pooler_output

            # Compute cosine similarity
            # 计算余弦相似度
            cos_sim = torch.nn.functional.cosine_similarity(test_embedding, dataset_embedding)
            scores.append(cos_sim.item())

    # Rank the sentences based on the scores
    # 根据分数从高到底排序，选出最相似的top_k个句子
    ranked_sentences_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    top_sentences_sss = [dataset_sentences[i] for i in ranked_sentences_indices[:top_k]]
    top_scores_sss = [scores[i] for i in ranked_sentences_indices[:top_k]]

    return top_sentences_sss, top_scores_sss


def final_prompt(question, model_pth, optimizer_pth, dataset_sentences):
    # 句子相似度模型
    m_model = SentenceSimilarityModel()
    # 优化器
    o_optimizer = AdamW(m_model.parameters(), lr=5e-5)
    # Load the saved states
    # 加载已保存的模型和优化器参数
    load_model_and_optimizer(m_model, o_optimizer, model_pth, optimizer_pth)
    #m_model = m_model.to(device)
    # 找到最相似的句子
    t_top_sentences, t_top_scores = find_most_similar_sentences(question, dataset_sentences, m_model, m_model.tokenizer,
                                                                top_k=1)
    # 根据找到的句子，完成字符串拼接
    finalQuestion = ""
    for s_sentence in t_top_sentences:
        finalQuestion = finalQuestion + s_sentence + " | "
    return finalQuestion + "| Now, based on these question and answer, what is the answer of question:" + question


def generate_answer_afterRetrieval(questions, model_pth, optimizer_pth, dataset_sentences):
    final_questions = []
    for question in tqdm(questions):
        final_questions.append(final_prompt(question, model_pth, optimizer_pth, dataset_sentences))
    generated_answers = []
    for final_question in tqdm(final_questions):
        generated_answers.append(chatgpt_answer(final_question))
    return generated_answers


def calculate_bertScores(generated_answers, standard_answers):
    bertscore = load("bertscore")
    results = bertscore.compute(predictions=generated_answers, references=standard_answers,
                                model_type="microsoft/deberta-xlarge-mnli")
    if results is None:
        raise Exception("No BERTScore results found")
    return sum(results['precision'])/len(results['precision'])


def calculate_aggregated_rouge_score(generated_answers, standard_answers):
    rouge = evaluate.load('rouge')
    results = rouge.compute(predictions=generated_answers, references=standard_answers, use_aggregator=True)
    return results


"""
def calculate_aggregated_rouge_score(generated_answers, standard_answers):
    # Ensure the 'rouge' package is installed: pip install rouge
    rouge = Rouge()

    # Calculate ROUGE scores for each generated-standard answer pair
    scores = [rouge.get_scores(gen_ans, std_ans)[0] for gen_ans, std_ans in zip(generated_answers, standard_answers)]

    # Calculate the average of the F1 scores for ROUGE-1, ROUGE-2, and ROUGE-L across all answers
    avg_rouge_f1 = sum(score['rouge-1']['f'] + score['rouge-2']['f'] + score['rouge-l']['f'] for score in scores) / (
            3 * len(scores))

    return avg_rouge_f1
"""


def calculate_aggregated_rouge_score_withoutRetrieval(questions, standard_answers):
    # Ensure the 'rouge' package is installed: pip install rouge
    rouge = Rouge()
    generated_answers = []
    for final_question in questions:
        generated_answers.append(chatgpt_answer(final_question))
    # Calculate ROUGE scores for each generated-standard answer pair
    scores = [rouge.get_scores(gen_ans, std_ans)[0] for gen_ans, std_ans in zip(generated_answers, standard_answers)]
    if scores is None:
        raise Exception("No ROUGE scores found")
    
    # Calculate the average of the F1 scores for ROUGE-1, ROUGE-2, and ROUGE-L across all answers
    avg_rouge_f1 = sum(score['rouge-1']['f'] + score['rouge-2']['f'] + score['rouge-l']['f'] for score in scores) / (
            3 * len(scores))

    return avg_rouge_f1


def main(file_name, test_file):
    # file_name 参数作为记忆池，用于检索和辅助生成答案
    # test_file 参数作为测试集，用于计算评价指标
    all_sentences = allSentences(file_name)
    test_sentences = allSentences(test_file)
    questions = []
    standard_answers = []
    for testSentence in test_sentences:
        dash_before, dash_after = testSentence.split('->')
        questions.append(dash_before)
        standard_answers.append(dash_after)
    answers = generate_answer_afterRetrieval(questions, "model.pth", "optimizer.pth", all_sentences)
    print(calculate_aggregated_rouge_score(answers, standard_answers))
    print(calculate_bertScores(answers, standard_answers))


if __name__ == '__main__':
    main("PlanPool.jsonl", "studyTest_reduce.jsonl")
